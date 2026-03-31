"""
nsl_toolbox - Python implementation of the NSL Auditory Cortical Model
======================================================================

Provides ``wav2aud`` (waveform -> auditory spectrogram) and ``aud2cor``
(auditory spectrogram -> cortical rate-scale representation), faithful
translations of the original MATLAB NSL toolbox by Powen Ru, Taishih Chi,
and Shihab Shamma at the University of Maryland.
"""

from .core import (
    wav2aud,
    aud2cor,
    sigmoid,
    gen_cort,
    gen_corf,
    unitseq,
    halfregu,
    get_cf,
)

__all__ = [
    "wav2aud",
    "aud2cor",
    "sigmoid",
    "gen_cort",
    "gen_corf",
    "unitseq",
    "halfregu",
    "get_cf",
]
