"""
extract_features.py — Compute cochlear and cortical representations.

Reads the single continuous MP3 from generate_stimuli.py, uses the
metadata to slice it into trials, then computes per-trial:
  1. Auditory spectrogram (cochlear)  via wav2aud  — shape (T, 128)
  2. Cortical representation          via aud2cor  — shape (S, R*2, T, 128)

and saves everything as .npy / .npz archives.

Usage:
    python extract_features.py
"""

import os
import time
import numpy as np
import soundfile as sf

from nsl_toolbox import wav2aud, aud2cor, unitseq, get_cf
from config import (
    MODEL_SR,
    WAV2AUD_PARAMS,
    RATE_VECTOR,
    SCALE_VECTOR,
    FEATURES_DIR,
    OUTPUT_DIR,
)


def load_experiment(output_dir):
    """Load the single continuous MP3 and trial metadata.

    Returns
    -------
    audio : ndarray, 1-D
        Full continuous waveform at MODEL_SR.
    meta : dict
        Metadata from experiment_metadata.npz.
    """
    mp3_path = os.path.join(output_dir, "experiment_audio.mp3")
    audio, sr = sf.read(mp3_path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MODEL_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SR)
        # Recompute onsets for the new sample rate
        scale = MODEL_SR / sr
    else:
        scale = 1.0

    meta_path = os.path.join(output_dir, "experiment_metadata.npz")
    meta = dict(np.load(meta_path, allow_pickle=False))

    # Scale onsets if resampled
    if scale != 1.0:
        meta["trial_onsets_samples"] = np.round(
            meta["trial_onsets_samples"] * scale
        ).astype(int)

    print(f"Loaded {mp3_path}: {len(audio)/MODEL_SR:.1f} s @ {MODEL_SR} Hz")
    print(f"  {len(meta['trial_order'])} trials")
    return audio, meta


def slice_trials(audio, meta):
    """Slice the continuous audio into individual trial segments.

    Returns
    -------
    trial_audios : list of ndarray
        Each entry is one trial's audio (DUR_FULL seconds).
    """
    fs = MODEL_SR
    dur_full = float(meta["dur_full"])
    L_trial = round(dur_full * fs)
    onsets = meta["trial_onsets_samples"]

    trial_audios = []
    for onset in onsets:
        start = int(onset)
        end = min(start + L_trial, len(audio))
        segment = audio[start:end]
        # Pad if segment is slightly short (edge of file)
        if len(segment) < L_trial:
            segment = np.pad(segment, (0, L_trial - len(segment)))
        trial_audios.append(segment)

    return trial_audios


def process_all_trials(trial_audios, labels, paras, rv, sv,
                       compute_cortical_flag=True):
    """Compute cochlear + cortical features for all trials."""
    label_names = {0: "N", 1: "RN", 2: "RefRN"}
    n_trials = len(trial_audios)
    cochlear_list = []
    cortical_list = []

    t0 = time.time()
    for i, audio in enumerate(trial_audios):
        x = unitseq(audio)
        aud = wav2aud(x, paras)
        cochlear_list.append(aud)

        cr = None
        if compute_cortical_flag:
            cr = aud2cor(aud, paras, rv, sv)
            cortical_list.append(cr)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_trials - i - 1)
        lbl = label_names.get(int(labels[i]), "?")
        info = f"cochlear={aud.shape}"
        if cr is not None:
            info += f"  cortical={cr.shape}"
        print(f"  [{i+1:3d}/{n_trials}] {lbl:5s}  {info}  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    return cochlear_list, cortical_list


def save_features(cochlear_list, cortical_list, meta):
    """Save all features to disk."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    labels = meta["trial_order"]

    # --- Stacked cochlear: (n_trials, T, 128) ---
    cochlear_stacked = np.stack(cochlear_list, axis=0)
    coch_path = os.path.join(FEATURES_DIR, "cochlear.npy")
    np.save(coch_path, cochlear_stacked)
    print(f"Cochlear {cochlear_stacked.shape} -> {coch_path}")

    # --- Stacked cortical magnitude + phase: (n_trials, S, R*2, T, 128) ---
    if cortical_list:
        cr_mag = np.stack(
            [np.abs(cr).astype(np.float32) for cr in cortical_list], axis=0
        )
        cr_phase = np.stack(
            [np.angle(cr).astype(np.float32) for cr in cortical_list], axis=0
        )
        mag_path = os.path.join(FEATURES_DIR, "cortical_magnitude.npy")
        phase_path = os.path.join(FEATURES_DIR, "cortical_phase.npy")
        np.save(mag_path, cr_mag)
        np.save(phase_path, cr_phase)
        print(f"Cortical magnitude {cr_mag.shape} -> {mag_path}")
        print(f"Cortical phase     {cr_phase.shape} -> {phase_path}")

    # --- Labels ---
    labels_path = os.path.join(FEATURES_DIR, "labels.npy")
    np.save(labels_path, labels)
    print(f"Labels {labels.shape} -> {labels_path}")

    # --- Model parameters ---
    np.savez(
        os.path.join(FEATURES_DIR, "model_params.npz"),
        wav2aud_params=np.array(WAV2AUD_PARAMS),
        rate_vector=RATE_VECTOR,
        scale_vector=SCALE_VECTOR,
        model_sr=MODEL_SR,
        cf=get_cf(shft=WAV2AUD_PARAMS[3]),
    )


def main():
    print("=" * 60)
    print("  Auditory Feature Extraction")
    print("=" * 60)

    # 1. Load the single continuous MP3 + metadata
    audio, meta = load_experiment(OUTPUT_DIR)

    # 2. Slice into trials
    trial_audios = slice_trials(audio, meta)
    print(f"Sliced into {len(trial_audios)} trials, "
          f"each {len(trial_audios[0])/MODEL_SR:.3f} s")

    # 3. Compute features
    print(f"\nwav2aud params: frmlen={WAV2AUD_PARAMS[0]} ms, "
          f"tc={WAV2AUD_PARAMS[1]} ms, fac={WAV2AUD_PARAMS[2]}, "
          f"shft={WAV2AUD_PARAMS[3]}")
    print(f"Cortical rates:  {RATE_VECTOR} Hz")
    print(f"Cortical scales: {SCALE_VECTOR} cyc/oct\n")

    cochlear_list, cortical_list = process_all_trials(
        trial_audios, meta["trial_order"],
        WAV2AUD_PARAMS, RATE_VECTOR, SCALE_VECTOR,
        compute_cortical_flag=True,
    )

    # 4. Save
    print(f"\nSaving to {FEATURES_DIR}/ ...")
    save_features(cochlear_list, cortical_list, meta)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Trials:            {len(cochlear_list)}")
    print(f"  Cochlear shape:    {cochlear_list[0].shape}  per trial")
    print(f"  Cortical shape:    {cortical_list[0].shape}  per trial")
    print(f"  Frame resolution:  {WAV2AUD_PARAMS[0]} ms")
    cf = get_cf(WAV2AUD_PARAMS[3])
    print(f"  CF range:          {cf[0]:.0f} - {cf[-1]:.0f} Hz")
    print("Done.")


if __name__ == "__main__":
    main()
