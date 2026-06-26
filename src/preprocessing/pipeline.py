"""
Preprocessing Pipeline
-----------------------
Normalization → Windowing → Ready for feature extraction

Steps:
  1. Normalize: zero-mean, unit-variance per signal
  2. Window: split long signal into overlapping segments
     (Hann window to reduce spectral leakage)
"""

import numpy as np
from typing import Iterator


# ── Normalization ────────────────────────────────────────────────────

def normalize(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a vibration signal.

    Parameters
    ----------
    signal : np.ndarray   Raw signal
    method : str          'zscore' | 'minmax' | 'rms'
    """
    if method == "zscore":
        mu, sigma = signal.mean(), signal.std()
        return (signal - mu) / (sigma + 1e-10)

    elif method == "minmax":
        lo, hi = signal.min(), signal.max()
        return (signal - lo) / (hi - lo + 1e-10)

    elif method == "rms":
        rms = np.sqrt(np.mean(signal ** 2))
        return signal / (rms + 1e-10)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ── Windowing ────────────────────────────────────────────────────────

def window_signal(signal: np.ndarray,
                  window_size: int = 1024,
                  hop_size: int = 512,
                  apply_hann: bool = True) -> np.ndarray:
    """
    Slice a signal into overlapping windows.

    Parameters
    ----------
    signal      : np.ndarray   Input signal (1D)
    window_size : int          Samples per window
    hop_size    : int          Step between windows (< window_size = overlap)
    apply_hann  : bool         Apply Hann window to reduce spectral leakage

    Returns
    -------
    windows : np.ndarray  Shape (N_windows, window_size)
    """
    hann = np.hanning(window_size) if apply_hann else np.ones(window_size)
    starts = range(0, len(signal) - window_size + 1, hop_size)
    windows = np.array([signal[s:s + window_size] * hann for s in starts])
    return windows


def window_generator(signal: np.ndarray,
                     window_size: int = 1024,
                     hop_size: int = 512) -> Iterator[np.ndarray]:
    """Memory-efficient windowing generator for streaming use."""
    for start in range(0, len(signal) - window_size + 1, hop_size):
        yield signal[start:start + window_size]


# ── Full preprocessing pipeline ──────────────────────────────────────

class PreprocessingPipeline:
    def __init__(self,
                 norm_method:  str  = "zscore",
                 window_size:  int  = 1024,
                 hop_size:     int  = 512,
                 apply_hann:   bool = True):
        self.norm_method = norm_method
        self.window_size = window_size
        self.hop_size    = hop_size
        self.apply_hann  = apply_hann

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize then window a raw signal.

        Returns
        -------
        windows : np.ndarray  Shape (N_windows, window_size)
        """
        normalized = normalize(signal, self.norm_method)
        windows    = window_signal(normalized, self.window_size,
                                   self.hop_size, self.apply_hann)
        return windows

    def process_batch(self, signals: list) -> list:
        """Process a list of signals. Returns list of window arrays."""
        return [self.process(s) for s in signals]
