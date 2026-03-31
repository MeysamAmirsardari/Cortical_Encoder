"""
NSL Toolbox - Python implementation
====================================
Python translation of the NSL (Neural Systems Laboratory) auditory cortical
model toolbox, originally written in MATLAB by Powen Ru and Taishih Chi at
the University of Maryland.

Reference:
    Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal
    analysis of complex sounds. J. Acoust. Soc. Am., 118(2), 887-906.

This module provides:
    - wav2aud : acoustic waveform -> auditory spectrogram
    - aud2cor : auditory spectrogram -> cortical representation (rate-scale)
    - Supporting functions: sigmoid, gen_cort, gen_corf, unitseq, halfregu
"""

import os
import math
import numpy as np
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Module-level COCHBA filter bank (loaded lazily)
# ---------------------------------------------------------------------------
_COCHBA = None
_COCHBA_OLD = None

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_cochba(filt='p'):
    """Load the cochlear filter bank coefficients.

    Parameters
    ----------
    filt : str
        'p'   -> Powen's IIR filter (default)
        'p_o' -> Powen's old IIR filter (steeper group delay)

    Returns
    -------
    COCHBA : ndarray, shape (L, M), complex128
        Row 0 stores filter order (real) and characteristic frequency (imag).
        Rows 1.. store IIR coefficients: B = real part, A = imag part.
    """
    global _COCHBA, _COCHBA_OLD
    if filt == 'p_o':
        if _COCHBA_OLD is None:
            path = os.path.join(_DATA_DIR, 'aud24_old.npy')
            _COCHBA_OLD = np.load(path, allow_pickle=False)
        return _COCHBA_OLD
    else:
        if _COCHBA is None:
            path = os.path.join(_DATA_DIR, 'aud24.npy')
            _COCHBA = np.load(path, allow_pickle=False)
        return _COCHBA


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def unitseq(x):
    """Normalise a sequence to zero-mean, unit-variance (N(0,1)).

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x : ndarray, normalised
    """
    x = x - np.mean(x)
    x = x / np.std(x, ddof=0)
    return x


def sigmoid(y, fac):
    """Non-linear hair-cell transduction function.

    Parameters
    ----------
    y : ndarray
        Input signal.
    fac : float
        Non-linear factor:
        - fac > 0  : sigmoid (transistor-like) compression
        - fac == 0 : hard limiter  (y > 0)
        - fac == -1: half-wave rectifier  max(y, 0)
        - fac == -2: linear (identity)
        - fac == -3: half-duration regulator (halfregu)

    Returns
    -------
    y : ndarray
    """
    if fac > 0:
        y = np.exp(-y / fac)
        y = 1.0 / (1.0 + y)
    elif fac == 0:
        y = (y > 0).astype(np.float64)
    elif fac == -1:
        y = np.maximum(y, 0.0)
    elif fac == -3:
        y = halfregu(y)
    # fac == -2 or any other value: identity (linear), no operation
    return y


def halfregu(y):
    """Half-duration regulator.

    Regulates a vector such that the positive-going duration is halved.

    Parameters
    ----------
    y : ndarray, 1-D

    Returns
    -------
    y : ndarray, 1-D  (binary 0/1)
    """
    y = (y > 0).astype(np.float64)
    dy = np.diff(y)
    edge_up = np.where(dy == 1)[0]
    edge_dn = np.where(dy == -1)[0]

    L_dn = len(edge_dn)
    L_up = len(edge_up)

    if L_dn * L_up == 0:
        return y

    dy[edge_dn] = 0.0

    L = len(y)
    if edge_dn[0] > edge_up[0]:
        # first point is zero
        edge_up_ext = np.append(edge_up, L)
        edge_dn_new = np.round(
            (edge_up_ext[:L_dn] + edge_up_ext[1:L_dn + 1]) / 2.0
        ).astype(int)
        dy[edge_dn_new] = dy[edge_dn_new] - 1
        y = np.cumsum(np.concatenate(([0.0], dy)))
    else:
        # first point is one
        edge_up_ext = np.concatenate(([0], edge_up, [L]))
        edge_dn_new = np.round(
            (edge_up_ext[:L_dn] + edge_up_ext[1:L_dn + 1]) / 2.0
        ).astype(int)
        dy[edge_dn_new] = dy[edge_dn_new] - 1
        y = np.cumsum(np.concatenate(([1.0], dy)))

    return y


# ---------------------------------------------------------------------------
# wav2aud  --  acoustic waveform  ->  auditory spectrogram
# ---------------------------------------------------------------------------

def wav2aud(x, paras, filt='p', verbose=False):
    """Compute the auditory spectrogram for an acoustic waveform.

    Attempt to be a **faithful** translation of the MATLAB ``wav2aud.m``
    from the NSL toolbox (Powen Ru / Taishih Chi, UMD).

    Parameters
    ----------
    x : ndarray, 1-D
        Acoustic input waveform.
    paras : array-like, length 4
        [frmlen, tc, fac, shft]
        - frmlen : frame length in ms (e.g. 8, 16).
        - tc     : time constant in ms (e.g. 4, 16, 64). 0 = short-term avg.
        - fac    : nonlinear factor (see ``sigmoid``). Typical 0.1 for
                   unit-normalised input; -1 for half-wave rectifier;
                   -2 for linear.
        - shft   : octave shift.  0 for 16 kHz, -1 for 8 kHz, etc.
                   Effective sample rate = 16000 * 2**shft.
    filt : str, optional
        'p' (default) or 'p_o' for the old filter set.
    verbose : bool, optional
        Print progress every octave.

    Returns
    -------
    v5 : ndarray, shape (N_frames, M-1)
        Auditory spectrogram.  Channels ordered from *low* to *high* frequency
        (column 0 = lowest CF).
    """
    if filt == 'k':
        raise ValueError("FIR filtering not supported. Use filt='p'.")

    COCHBA = _load_cochba(filt)
    L, M = COCHBA.shape  # L = max coeff rows, M = number of channels

    x = np.array(x, dtype=np.float64).ravel()
    L_x = len(x)

    # parse parameters
    shft = paras[3]
    fac = paras[2]
    L_frm = round(paras[0] * 2 ** (4 + shft))  # frame length in samples

    if paras[1] != 0:
        alph = math.exp(-1.0 / (paras[1] * 2 ** (4 + shft)))
    else:
        alph = 0.0

    # hair cell membrane time constant
    haircell_tc = 0.5
    beta = math.exp(-1.0 / (haircell_tc * 2 ** (4 + shft)))

    # number of frames & zero-pad
    N = math.ceil(L_x / L_frm)
    x = np.pad(x, (0, N * L_frm - L_x), mode='constant')

    v5 = np.zeros((N, M - 1))

    # ------------------------------------------------------------------
    # last (highest-frequency) channel  -- used as y2_h for LIN
    # ------------------------------------------------------------------
    p = int(np.real(COCHBA[0, M - 1]))
    B = np.real(COCHBA[1:p + 2, M - 1]).copy()
    A = np.imag(COCHBA[1:p + 2, M - 1]).copy()
    y1 = lfilter(B, A, x)
    y2 = sigmoid(y1, fac)
    if fac != -2:
        y2 = lfilter([1.0], [1.0, -beta], y2)
    y2_h = y2.copy()

    # ------------------------------------------------------------------
    # remaining channels  (M-2 down to 0, i.e. high -> low freq)
    # ------------------------------------------------------------------
    for ch in range(M - 2, -1, -1):
        # --- cochlear filterbank ---
        p = int(np.real(COCHBA[0, ch]))
        B = np.real(COCHBA[1:p + 2, ch]).copy()
        A = np.imag(COCHBA[1:p + 2, ch]).copy()
        y1 = lfilter(B, A, x)

        # --- hair cell transduction ---
        y2 = sigmoid(y1, fac)
        if fac != -2:
            y2 = lfilter([1.0], [1.0, -beta], y2)

        # --- lateral inhibitory network ---
        y3 = y2 - y2_h
        y2_h = y2.copy()

        # --- half-wave rectifier ---
        y4 = np.maximum(y3, 0.0)

        # --- temporal integration ---
        if alph:
            y5 = lfilter([1.0], [1.0, -alph], y4)
            v5[:, ch] = y5[(np.arange(N) + 1) * L_frm - 1]
        else:
            if L_frm == 1:
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4[:N * L_frm].reshape(N, L_frm), axis=1)

        if verbose and (ch + 1) % 24 == 0:
            print(f'{(M - 1 - ch) // 24} octave(s) processed')

    return v5


# ---------------------------------------------------------------------------
# gen_cort  --  cortical *temporal* filter (rate)
# ---------------------------------------------------------------------------

def gen_cort(fc, L, STF, PASS=None):
    """Generate a (bandpass) cortical temporal filter transfer function.

    Parameters
    ----------
    fc : float
        Characteristic rate in Hz.
    L : int
        Filter length (power of 2 preferred).
    STF : float
        Temporal sampling rate (frames per second).
    PASS : list of two ints, optional
        [idx, K]. idx=1 -> lowpass; idx=K -> highpass; else bandpass.
        Default [2, 3] (bandpass).

    Returns
    -------
    H : ndarray, complex, length L
        One-sided transfer function.
    """
    if PASS is None:
        PASS = [2, 3]

    t = np.arange(L, dtype=np.float64) / STF * fc
    h = np.sin(2 * np.pi * t) * (t ** 2) * np.exp(-3.5 * t) * fc
    h = h - np.mean(h)

    H0 = np.fft.fft(h, 2 * L)
    A = np.angle(H0[:L])
    H = np.abs(H0[:L])
    maxi = np.argmax(H)
    H = H / H[maxi]

    # passband modification
    if PASS[0] == 1:  # lowpass
        H[:maxi] = 1.0
    elif PASS[0] == PASS[1]:  # highpass
        H[maxi + 1:L] = 1.0

    H = H * np.exp(1j * A)
    return H


# ---------------------------------------------------------------------------
# gen_corf  --  cortical *spectral* filter (scale)
# ---------------------------------------------------------------------------

def gen_corf(fc, L, SRF, KIND=None):
    """Generate a (bandpass) cortical spectral filter transfer function.

    Parameters
    ----------
    fc : float
        Characteristic scale in cycles/octave.
    L : int
        Filter length (power of 2 preferred).
    SRF : float
        Spectral sampling rate (channels per octave), typically 24.
    KIND : int or list, optional
        If scalar: 1 = Gabor, 2 = Gaussian (default).
        If [idx, K]: passband spec (same as PASS in gen_cort).

    Returns
    -------
    H : ndarray, real, length L
        One-sided transfer function.
    """
    if KIND is None:
        KIND_val = 2
        PASS = [2, 3]
    elif np.isscalar(KIND):
        KIND_val = KIND
        PASS = [2, 3]
    else:
        PASS = list(KIND)
        KIND_val = 2

    R1 = np.arange(L, dtype=np.float64) / L * SRF / 2.0 / abs(fc)

    if KIND_val == 1:  # Gabor
        C1 = 1.0 / (2.0 * 0.3 * 0.3)
        H = np.exp(-C1 * (R1 - 1.0) ** 2) + np.exp(-C1 * (R1 + 1.0) ** 2)
    else:  # Gaussian (negative 2nd derivative)
        R1 = R1 ** 2
        H = R1 * np.exp(1.0 - R1)

    # passband
    maxi = np.argmax(H)
    if PASS[0] == 1:  # lowpass
        sumH = np.sum(H)
        H[:maxi] = 1.0
        H = H / np.sum(H) * sumH
    elif PASS[0] == PASS[1]:  # highpass
        sumH = np.sum(H)
        H[maxi + 1:L] = 1.0
        H = H / np.sum(H) * sumH

    return H


# ---------------------------------------------------------------------------
# aud2cor  --  auditory spectrogram  ->  cortical representation
# ---------------------------------------------------------------------------

def aud2cor(y, para1, rv, sv):
    """Compute the cortical rate-scale representation (forward transform).

    Parameters
    ----------
    y : ndarray, shape (N, M)
        Auditory spectrogram (from ``wav2aud``).
        N = number of time frames, M = number of frequency channels.
    para1 : array-like
        Parameters vector.  First 4 entries are the ``paras`` used for
        ``wav2aud``.  Optional entries:
        - para1[4] = FULLT (temporal margin fullness, 0..1, default 0)
        - para1[5] = FULLX (spectral margin fullness, 0..1, default FULLT)
        - para1[6] = BP    (pure bandpass indicator, default 0)
    rv : array-like
        Rate vector in Hz, e.g. ``2**np.arange(1, 5.5, 0.5)``.
    sv : array-like
        Scale vector in cyc/oct, e.g. ``2**np.arange(-2, 3.5, 0.5)``.

    Returns
    -------
    cr : ndarray, complex, shape (K2, K1*2, N+2*dN, M+2*dM)
        Cortical representation.
        - Axis 0: scale channels (len = K2 = len(sv))
        - Axis 1: rate channels (len = K1*2 = len(rv)*2).
          First K1 entries are *downward* (sgn = -1), next K1 are *upward* (sgn = +1).
        - Axis 2: time frames (with optional margin dN)
        - Axis 3: frequency channels (with optional margin dM)
    """
    para1 = np.asarray(para1, dtype=np.float64).ravel()
    rv = np.asarray(rv, dtype=np.float64).ravel()
    sv = np.asarray(sv, dtype=np.float64).ravel()

    FULLT = para1[4] if len(para1) > 4 else 0.0
    FULLX = para1[5] if len(para1) > 5 else FULLT
    BP = int(para1[6]) if len(para1) > 6 else 0

    K1 = len(rv)  # number of rate channels
    K2 = len(sv)  # number of scale channels
    N, M = y.shape

    # zero-pad to powers of 2
    N1 = int(2 ** np.ceil(np.log2(N)))
    N2 = N1 * 2
    M1 = int(2 ** np.ceil(np.log2(M)))
    M2 = M1 * 2

    # 2-D FFT of the auditory spectrogram
    # first FFT along frequency axis
    Y = np.zeros((N2, M1), dtype=np.complex128)
    for n in range(N):
        R1 = np.fft.fft(y[n, :], M2)
        Y[n, :] = R1[:M1]
    # second FFT along time axis
    for m in range(M1):
        R1 = np.fft.fft(Y[:N, m], N2)
        Y[:, m] = R1

    paras = para1[:4]
    STF = 1000.0 / paras[0]  # frames per second
    SRF = 20 if M == 95 else 24  # channels per octave

    # frequency margin indices
    dM = int(math.floor(M / 2.0 * FULLX))
    # build index array matching MATLAB: [(1:dM)+M2-dM  1:M+dM]
    # MATLAB is 1-based; convert to 0-based
    mdx1_part1 = np.arange(dM) + M2 - dM  # indices M2-dM .. M2-1
    mdx1_part2 = np.arange(M + dM)         # indices 0 .. M+dM-1
    mdx1 = np.concatenate([mdx1_part1, mdx1_part2]).astype(int)

    # temporal margin indices
    dN = int(math.floor(N / 2.0 * FULLT))
    ndx1 = np.arange(N + 2 * dN)  # 0-based, length N+2*dN

    cr = np.zeros((K2, K1 * 2, N + 2 * dN, M + 2 * dM), dtype=np.complex128)

    for rdx in range(K1):
        fc_rt = rv[rdx]
        HR = gen_cort(fc_rt, N1, STF, [rdx + 1 + BP, K1 + BP * 2])

        for sgn in [1, -1]:
            if sgn > 0:
                # SSB -> DSB: append N1 zeros
                HR = np.concatenate([HR, np.zeros(N1, dtype=np.complex128)])
            else:
                # conjugate flip for downward
                HR = np.concatenate([HR[:1], np.conj(HR[1:N2][::-1])])
                HR[N1] = np.abs(HR[N1 + 1])

            # -- optimised: first inverse FFT (temporal) pulled out of scale loop --
            z1 = np.zeros((N2, M1), dtype=np.complex128)
            for m in range(M1):
                z1[:, m] = HR * Y[:, m]
            z1 = np.fft.ifft(z1, axis=0)
            z1 = z1[ndx1, :]  # shape (N+2*dN, M1)

            for sdx in range(K2):
                fc_sc = sv[sdx]
                HS = gen_corf(fc_sc, M1, SRF, [sdx + 1 + BP, K2 + BP * 2])

                # second inverse FFT (spectral)
                z = np.zeros((N + 2 * dN, M + 2 * dM), dtype=np.complex128)
                for n_idx in range(len(ndx1)):
                    R1 = np.fft.ifft(z1[n_idx, :] * HS, M2)
                    z[n_idx, :] = R1[mdx1]

                # store: upward -> columns K1..2*K1-1, downward -> 0..K1-1
                col_idx = rdx + (1 if sgn == 1 else 0) * K1
                cr[sdx, col_idx, :, :] = z

    return cr


# ---------------------------------------------------------------------------
# Convenience: characteristic frequencies
# ---------------------------------------------------------------------------

def get_cf(shft=0):
    """Return the characteristic frequencies for the 128-channel filterbank.

    Parameters
    ----------
    shft : float
        Octave shift (same as paras[3]).

    Returns
    -------
    cf : ndarray, shape (128,)
        Characteristic frequencies in Hz for channels 1..128
        (matching columns 0..127 of the auditory spectrogram output).
    """
    # MATLAB: CF = 440 * 2 .^ ((-31:97)/24 + shft)
    # Channels 1..128 in MATLAB correspond to COCHBA columns 1..128
    # (column 0 = channel 129 = highest freq, used only as y2_h reference)
    # The auditory spectrogram has M-1 = 128 columns
    return 440.0 * 2.0 ** ((np.arange(-31, 97) / 24.0) + shft)
