"""
config.py — Experiment and model parameters for the Agus et al. (2010) paradigm.

Reference:
    Agus, T. R., Thorpe, S. J., & Pressnitzer, D. (2010).
    Rapid formation of robust auditory memories: Insights from noise.
    Neuron, 66(4), 610-618.
"""

import numpy as np

# ============================================================================
#  Experiment parameters  (Agus 2010, Experiment 1)
# ============================================================================

# Audio generation sample rate (high quality for saving)
GENERATION_SR = 16000

# Trial structure
DUR_HALF = 0.5          # seconds — half-segment duration for repetition
DUR_FULL = 1.0          # seconds — total trial duration (2 × DUR_HALF)
INTER_TRIAL_GAP = 1.5   # seconds — silence between trials

# Trial counts per block
N_NOISE = 100           # N   — plain noise (no repetition), correct = "No"
N_RN = 50               # RN  — repeated noise (fresh each trial), correct = "Yes"
N_REFRN = 50            # RefRN — reference repeated noise (frozen), correct = "Yes"

# Random seeds
FROZEN_SEED = 42        # seed for generating the frozen RefRN segment

# ============================================================================
#  NSL cochlear model parameters
# ============================================================================

# Internal sample rate for feature extraction
MODEL_SR = 16000        # Hz — octave shift = 0

# wav2aud parameters: [frmlen, tc, fac, shft]
FRMLEN = 1              # frame length in ms → 1 ms resolution
TC = 8                  # time constant in ms (leaky integrator)
FAC = -2                # nonlinear factor: -2 = linear
OCTAVE_SHIFT = 0        # 0 for 16 kHz  (log2(MODEL_SR / 16000))

WAV2AUD_PARAMS = [FRMLEN, TC, FAC, OCTAVE_SHIFT]

# ============================================================================
#  NSL cortical model parameters
# ============================================================================

# Rate vector (temporal modulation frequencies, Hz)
RATE_VECTOR = 2.0 ** np.arange(1, 6)       # [2, 4, 8, 16, 32] Hz

# Scale vector (spectral modulation densities, cyc/oct)
SCALE_VECTOR = 2.0 ** np.arange(-2, 4)     # [0.25, 0.5, 1, 2, 4, 8] cyc/oct

# ============================================================================
#  Output paths
# ============================================================================

OUTPUT_DIR = "output"
AUDIO_DIR = f"{OUTPUT_DIR}/audio"
FEATURES_DIR = f"{OUTPUT_DIR}/features"
