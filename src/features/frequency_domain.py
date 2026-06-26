"""
Frequency Domain Features
--------------------------
Features derived from the FFT magnitude spectrum.
These reveal periodic components linked to bearing fault frequencies.

Features:
  dominant_freq     — frequency of highest magnitude peak
  spectral_centroid — weighted mean frequency (centre of mass of spectrum)
  spectral_spread   — weighted std dev of frequencies
  band_energy_*     — energy in sub-bands (0-1kHz, 1-3kHz, 3-6kHz)
  fft_peak_1/2/3    — top 3 spectral peak magnitudes
  spectral_entropy  — randomness of spectrum (low = tonal faults)
  spectral_flatness — ratio of geometric to arithmetic mean (tonal vs noise)
"""

import numpy as np


def compute_fft(signal: np.ndarray, sample_rate: float = 12_000.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided FFT magnitude spectrum.

    Returns
    -------
    freqs : np.ndarray  Frequency bins in Hz
    mag   : np.ndarray  Magnitude (normalized)
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    mag      = (2.0 / N) * np.abs(fft_vals)
    freqs    = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    return freqs, mag


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Frequency of the highest magnitude component."""
    return float(freqs[np.argmax(mag)])


def spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Weighted mean frequency — centre of spectral mass."""
    total = np.sum(mag)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * mag) / total)


def spectral_spread(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Weighted std dev of frequency distribution."""
    centroid = spectral_centroid(freqs, mag)
    total = np.sum(mag)
    if total < 1e-10:
        return 0.0
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total))


def band_energy(freqs: np.ndarray, mag: np.ndarray,
                f_low: float, f_high: float) -> float:
    """Total energy in a frequency sub-band [f_low, f_high] Hz."""
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(mag[mask] ** 2))


def top_fft_peaks(mag: np.ndarray, n: int = 3) -> list[float]:
    """Magnitudes of the top N spectral peaks."""
    sorted_mags = np.sort(mag)[::-1]
    return [float(v) for v in sorted_mags[:n]]


def spectral_entropy(mag: np.ndarray) -> float:
    """
    Spectral entropy — low value = tonal signal (strong fault frequency).
    Normalized to [0, 1].
    """
    ps = mag ** 2
    total = ps.sum()
    if total < 1e-10:
        return 0.0
    p = ps / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy / np.log2(len(mag)))  # normalize


def spectral_flatness(mag: np.ndarray) -> float:
    """
    Ratio of geometric to arithmetic mean.
    ~1 = flat noise; ~0 = tonal signal (strong fault peak).
    """
    mag_pos = mag[mag > 0]
    if len(mag_pos) == 0:
        return 0.0
    geo_mean  = float(np.exp(np.mean(np.log(mag_pos))))
    arith_mean = float(np.mean(mag_pos))
    return geo_mean / arith_mean if arith_mean > 1e-10 else 0.0


def extract_frequency_features(signal: np.ndarray,
                                sample_rate: float = 12_000.0) -> dict:
    """
    Extract all frequency domain features from a signal window.

    Returns
    -------
    dict of feature_name → float
    """
    freqs, mag = compute_fft(signal, sample_rate)
    peaks = top_fft_peaks(mag, n=3)

    return {
        "dominant_freq":      dominant_frequency(freqs, mag),
        "spectral_centroid":  spectral_centroid(freqs, mag),
        "spectral_spread":    spectral_spread(freqs, mag),
        "band_energy_0_1k":   band_energy(freqs, mag, 0, 1_000),
        "band_energy_1_3k":   band_energy(freqs, mag, 1_000, 3_000),
        "band_energy_3_6k":   band_energy(freqs, mag, 3_000, 6_000),
        "fft_peak_1":         peaks[0],
        "fft_peak_2":         peaks[1] if len(peaks) > 1 else 0.0,
        "fft_peak_3":         peaks[2] if len(peaks) > 2 else 0.0,
        "spectral_entropy":   spectral_entropy(mag),
        "spectral_flatness":  spectral_flatness(mag),
    }
