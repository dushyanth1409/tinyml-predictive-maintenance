"""
Time Domain Features
---------------------
Statistical features computed directly from the windowed signal.
These capture amplitude characteristics, impulsiveness, and distribution shape.

Features:
  RMS         — overall vibration level (most common in condition monitoring)
  Mean        — DC offset / bias
  Variance    — signal spread
  Std Dev     — square root of variance
  Kurtosis    — impulsiveness indicator (key for bearing faults, >3 = impulsive)
  Skewness    — asymmetry of distribution
  Peak        — maximum absolute value
  Peak-Peak   — full amplitude range
  Crest Factor— peak / RMS (>3 suggests impulses)
  Shape Factor— RMS / mean absolute (waveform shape)
  Impulse Factor — peak / mean absolute
"""

import numpy as np
from scipy.stats import kurtosis, skew


def rms(signal: np.ndarray) -> float:
    """Root Mean Square — primary vibration severity indicator."""
    return float(np.sqrt(np.mean(signal ** 2)))


def mean(signal: np.ndarray) -> float:
    return float(np.mean(signal))


def variance(signal: np.ndarray) -> float:
    return float(np.var(signal))


def std_dev(signal: np.ndarray) -> float:
    return float(np.std(signal))


def kurtosis_val(signal: np.ndarray) -> float:
    """
    Kurtosis — 4th statistical moment.
    Healthy bearing: ~3. Faulty bearing: >4 (impulse-like).
    """
    return float(kurtosis(signal, fisher=False))  # Pearson kurtosis


def skewness(signal: np.ndarray) -> float:
    """Skewness — 3rd statistical moment. Asymmetry of waveform."""
    return float(skew(signal))


def peak(signal: np.ndarray) -> float:
    """Maximum absolute amplitude."""
    return float(np.max(np.abs(signal)))


def peak_to_peak(signal: np.ndarray) -> float:
    """Full amplitude range."""
    return float(np.max(signal) - np.min(signal))


def crest_factor(signal: np.ndarray) -> float:
    """
    Peak / RMS — indicates presence of impulses.
    CF > 3–4 suggests bearing faults.
    """
    r = rms(signal)
    return float(peak(signal) / r) if r > 1e-10 else 0.0


def shape_factor(signal: np.ndarray) -> float:
    """RMS / Mean Absolute Value."""
    mean_abs = float(np.mean(np.abs(signal)))
    r = rms(signal)
    return r / mean_abs if mean_abs > 1e-10 else 0.0


def impulse_factor(signal: np.ndarray) -> float:
    """Peak / Mean Absolute Value."""
    mean_abs = float(np.mean(np.abs(signal)))
    return peak(signal) / mean_abs if mean_abs > 1e-10 else 0.0


def extract_time_features(signal: np.ndarray) -> dict:
    """
    Extract all time domain features from a signal window.

    Returns
    -------
    dict of feature_name → float
    """
    return {
        "rms":            rms(signal),
        "mean":           mean(signal),
        "variance":       variance(signal),
        "std_dev":        std_dev(signal),
        "kurtosis":       kurtosis_val(signal),
        "skewness":       skewness(signal),
        "peak":           peak(signal),
        "peak_to_peak":   peak_to_peak(signal),
        "crest_factor":   crest_factor(signal),
        "shape_factor":   shape_factor(signal),
        "impulse_factor": impulse_factor(signal),
    }
