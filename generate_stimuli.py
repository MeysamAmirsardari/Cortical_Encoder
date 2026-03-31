"""
generate_stimuli.py — Generate the Agus et al. (2010) stimulus set.

Creates a full experimental block:
  - 100 N  trials  (1 s of fresh Gaussian noise)
  - 50  RN trials  (0.5 s fresh noise repeated twice)
  - 50  RefRN trials (same frozen 0.5 s segment repeated twice)

Saves:
  - One continuous MP3 file (trials separated by silence gaps)
  - Trial metadata (.npz + .json) with onset info for slicing

Usage:
    python generate_stimuli.py
"""

import os
import json
import numpy as np
import soundfile as sf

from config import (
    GENERATION_SR,
    DUR_HALF,
    DUR_FULL,
    INTER_TRIAL_GAP,
    N_NOISE,
    N_RN,
    N_REFRN,
    FROZEN_SEED,
    OUTPUT_DIR,
)


def generate_block(seed_frozen=FROZEN_SEED):
    """Generate one experimental block following Agus et al. (2010).

    Returns
    -------
    trials : list of dict
        Each entry has keys: 'audio' (ndarray), 'label' (int), 'label_name' (str).
        Labels: 0 = N, 1 = RN, 2 = RefRN.
    frozen_half : ndarray
        The frozen 0.5 s noise segment used for all RefRN trials.
    trial_order : ndarray
        Shuffled label sequence (ints).
    """
    fs = GENERATION_SR
    L_half = round(DUR_HALF * fs)
    L_full = round(DUR_FULL * fs)

    # --- Generate the frozen RefRN segment (deterministic seed) ---
    rng_frozen = np.random.default_rng(seed_frozen)
    frozen_half = rng_frozen.standard_normal(L_half)
    refrn_audio = np.concatenate([frozen_half, frozen_half])

    # --- Build label pool and shuffle ---
    n_total = N_NOISE + N_RN + N_REFRN
    labels_pool = np.concatenate([
        np.zeros(N_NOISE, dtype=int),
        np.ones(N_RN, dtype=int),
        2 * np.ones(N_REFRN, dtype=int),
    ])

    # Shuffle until RefRN (label=2) is never on consecutive trials
    rng_shuffle = np.random.default_rng()
    while True:
        perm = rng_shuffle.permutation(n_total)
        trial_order = labels_pool[perm]
        refrn_positions = np.where(trial_order == 2)[0]
        if len(refrn_positions) < 2 or np.all(np.diff(refrn_positions) > 1):
            break

    # --- Generate audio for each trial ---
    rng_trial = np.random.default_rng()
    label_names = {0: "N", 1: "RN", 2: "RefRN"}
    trials = []

    for label in trial_order:
        if label == 0:  # N — fresh 1 s noise
            audio = rng_trial.standard_normal(L_full)
        elif label == 1:  # RN — fresh 0.5 s repeated
            half = rng_trial.standard_normal(L_half)
            audio = np.concatenate([half, half])
        else:  # RefRN — frozen segment repeated
            audio = refrn_audio.copy()

        trials.append({
            "audio": audio,
            "label": int(label),
            "label_name": label_names[label],
        })

    return trials, frozen_half, trial_order


def save_block(trials, trial_order, frozen_half):
    """Save one continuous MP3 and metadata."""
    fs = GENERATION_SR
    L_gap = round(INTER_TRIAL_GAP * fs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Build single continuous audio stream ---
    segments = []
    trial_onsets_samples = []
    current_sample = 0

    for trial in trials:
        trial_onsets_samples.append(current_sample)
        segments.append(trial["audio"])
        segments.append(np.zeros(L_gap))
        current_sample += len(trial["audio"]) + L_gap

    continuous = np.concatenate(segments)
    continuous = continuous / (np.max(np.abs(continuous)) + 1e-12)

    # --- Save the single MP3 ---
    mp3_path = os.path.join(OUTPUT_DIR, "experiment_audio.mp3")
    sf.write(mp3_path, continuous, fs)
    print(f"Saved continuous block: {len(continuous)/fs:.1f} s -> {mp3_path}")

    # --- Save metadata ---
    trial_onsets_samples = np.array(trial_onsets_samples, dtype=int)
    trial_onsets_sec = trial_onsets_samples / fs

    np.savez(
        os.path.join(OUTPUT_DIR, "experiment_metadata.npz"),
        trial_order=trial_order,
        trial_onsets_samples=trial_onsets_samples,
        trial_onsets_sec=trial_onsets_sec,
        frozen_half=frozen_half,
        fs=fs,
        dur_half=DUR_HALF,
        dur_full=DUR_FULL,
        inter_trial_gap=INTER_TRIAL_GAP,
    )

    summary = {
        "n_trials": len(trials),
        "fs": fs,
        "dur_full_s": DUR_FULL,
        "dur_half_s": DUR_HALF,
        "inter_trial_gap_s": INTER_TRIAL_GAP,
        "n_N": int(np.sum(trial_order == 0)),
        "n_RN": int(np.sum(trial_order == 1)),
        "n_RefRN": int(np.sum(trial_order == 2)),
        "trial_labels": trial_order.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metadata saved to {OUTPUT_DIR}/experiment_metadata.npz")


def main():
    print("Generating Agus et al. (2010) Experiment 1 stimuli...")
    trials, frozen_half, trial_order = generate_block()

    label_counts = {
        "N": int(np.sum(trial_order == 0)),
        "RN": int(np.sum(trial_order == 1)),
        "RefRN": int(np.sum(trial_order == 2)),
    }
    print(f"Trial counts: {label_counts}")

    save_block(trials, trial_order, frozen_half)
    print("Done.")


if __name__ == "__main__":
    main()
