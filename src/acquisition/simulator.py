"""
Bearing Fault Signal Simulator
--------------------------------
Generates synthetic vibration signals for four fault conditions using
real bearing geometry and fault frequency equations (CWRU dataset parameters).

Fault frequencies based on Case Western Reserve University bearing data:
  Bearing: SKF 6205-2RS JEM deep groove ball bearing
  Speed:   1797 RPM
  BPFO ≈  107.4 Hz  (Ball Pass Frequency Outer Race)
  BPFI ≈  162.2 Hz  (Ball Pass Frequency Inner Race)
  BSF  ≈   70.1 Hz  (Ball Spin Frequency)
  FTF  ≈    14.0 Hz  (Fundamental Train Frequency)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BearingParams:
    """Bearing geometry parameters (CWRU SKF 6205)."""
    n_balls:         int   = 9         # number of rolling elements
    ball_diameter:   float = 0.3126    # inches
    pitch_diameter:  float = 1.748     # inches
    contact_angle:   float = 0.0       # degrees


@dataclass
class SignalParams:
    sample_rate:    float = 12_000.0   # Hz
    duration:       float = 1.0        # seconds
    shaft_rpm:      float = 1797.0     # RPM
    noise_std:      float = 0.05
    impulse_amp:    float = 1.0


def compute_fault_frequencies(bearing: BearingParams, rpm: float) -> dict:
    """Compute characteristic fault frequencies from bearing geometry."""
    fr = rpm / 60.0
    ratio = (bearing.ball_diameter / bearing.pitch_diameter) * np.cos(np.radians(bearing.contact_angle))

    bpfo = (bearing.n_balls / 2) * fr * (1 - ratio)
    bpfi = (bearing.n_balls / 2) * fr * (1 + ratio)
    bsf  = (bearing.pitch_diameter / (2 * bearing.ball_diameter)) * fr * (1 - ratio**2)
    ftf  = (fr / 2) * (1 - ratio)

    return {"shaft": fr, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def _impulse_train(t: np.ndarray, freq: float, amplitude: float,
                   decay: float = 0.002, rng: np.random.Generator = None) -> np.ndarray:
    """Generate a series of decaying impulses at a given frequency."""
    signal = np.zeros_like(t)
    period = 1.0 / freq
    impulse_times = np.arange(0, t[-1], period)

    for ti in impulse_times:
        mask = t >= ti
        dt = t[mask] - ti
        signal[mask] += amplitude * np.exp(-dt / decay) * np.sin(2 * np.pi * 3000 * dt)

    return signal


def generate_signal(fault_type: str,
                    params: Optional[SignalParams] = None,
                    bearing: Optional[BearingParams] = None,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a vibration signal for a given fault condition.

    Parameters
    ----------
    fault_type : str   'healthy' | 'inner_race' | 'outer_race' | 'ball'
    params     : SignalParams
    bearing    : BearingParams
    seed       : int

    Returns
    -------
    t      : np.ndarray   time vector (seconds)
    signal : np.ndarray   vibration acceleration (g)
    """
    if params is None:  params = SignalParams()
    if bearing is None: bearing = BearingParams()

    rng = np.random.default_rng(seed)
    dt  = 1.0 / params.sample_rate
    t   = np.arange(0, params.duration, dt)
    freqs = compute_fault_frequencies(bearing, params.shaft_rpm)

    # Base: shaft harmonics + background noise
    signal  = 0.3 * np.sin(2 * np.pi * freqs["shaft"] * t)
    signal += 0.1 * np.sin(2 * np.pi * 2 * freqs["shaft"] * t)
    signal += rng.normal(0, params.noise_std, len(t))

    if fault_type == "healthy":
        # Slight random modulation only
        signal += 0.05 * np.sin(2 * np.pi * 5 * t) * rng.normal(0, 0.02, len(t))

    elif fault_type == "outer_race":
        signal += _impulse_train(t, freqs["BPFO"], params.impulse_amp)
        signal += _impulse_train(t, 2 * freqs["BPFO"], 0.3 * params.impulse_amp)

    elif fault_type == "inner_race":
        # Inner race fault: amplitude modulated by shaft frequency
        impulses = _impulse_train(t, freqs["BPFI"], params.impulse_amp)
        mod      = 0.5 * (1 + np.sin(2 * np.pi * freqs["shaft"] * t))
        signal  += impulses * mod

    elif fault_type == "ball":
        signal += _impulse_train(t, freqs["BSF"], params.impulse_amp * 0.7)
        signal += _impulse_train(t, 2 * freqs["BSF"], 0.4 * params.impulse_amp)
        signal += _impulse_train(t, freqs["FTF"], 0.2 * params.impulse_amp)

    else:
        raise ValueError(f"Unknown fault type: {fault_type}. "
                         f"Use: 'healthy', 'inner_race', 'outer_race', 'ball'")

    return t, signal.astype(np.float32)


FAULT_LABELS = {
    "healthy":     0,
    "inner_race":  1,
    "outer_race":  2,
    "ball":        3,
}

LABEL_NAMES = {v: k for k, v in FAULT_LABELS.items()}


def generate_dataset(n_samples_per_class: int = 200,
                     params: Optional[SignalParams] = None,
                     seed: int = 0) -> tuple[list, list]:
    """
    Generate a labelled dataset of raw vibration signals.

    Returns
    -------
    signals : list of np.ndarray
    labels  : list of int
    """
    signals, labels = [], []
    for fault, label in FAULT_LABELS.items():
        for i in range(n_samples_per_class):
            _, sig = generate_signal(fault, params=params, seed=seed + label * 1000 + i)
            signals.append(sig)
            labels.append(label)
    return signals, labels
