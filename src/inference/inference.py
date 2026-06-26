"""
Edge Inference
---------------
Lightweight inference script for Raspberry Pi / embedded Linux.

Loads a saved model and runs real-time fault prediction on incoming
vibration data from an IMU (MPU-6050 via I2C or serial stream).

Usage:
    # Simulate (no hardware):
    python src/inference/inference.py --simulate

    # Real MPU-6050 via serial:
    python src/inference/inference.py --port /dev/ttyUSB0 --baud 115200
"""

import numpy as np
import joblib
import json
import os
import argparse
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.features.extractor import FeatureExtractor
from src.acquisition.simulator import generate_signal, LABEL_NAMES

FAULT_EMOJIS = {
    "Healthy":     "✅",
    "Inner Race":  "⚠️ ",
    "Outer Race":  "🔴",
    "Ball":        "🟡",
}


class BearingInference:
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model    = joblib.load(model_path)
        self.extractor = FeatureExtractor()
        self.classes  = ["Healthy", "Inner Race", "Outer Race", "Ball"]

        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.meta = json.load(f)
            self.classes = self.meta.get("classes", self.classes)

        print(f"Loaded model: {os.path.basename(model_path)}")

    def predict(self, signal: np.ndarray) -> tuple[str, float]:
        """
        Predict fault type from a raw vibration signal.

        Returns
        -------
        fault_class : str    Fault label
        confidence  : float  Prediction confidence [0, 1]
        """
        X, _ = self.extractor.build_dataset([signal], [0])
        pred = self.model.predict(X)[0]

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))

        return self.classes[pred], confidence

    def run_realtime(self, signal_source, sample_rate: float = 12_000.0,
                     window_sec: float = 1.0, interval_sec: float = 0.5):
        """
        Continuously predict from a signal source.

        Parameters
        ----------
        signal_source : callable  Returns np.ndarray of vibration samples
        sample_rate   : float     Hz
        window_sec    : float     Window duration for one prediction
        interval_sec  : float     Pause between predictions
        """
        n_samples = int(sample_rate * window_sec)
        print(f"\nRunning real-time inference (window={window_sec}s, interval={interval_sec}s)")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                signal = signal_source(n_samples)
                fault, conf = self.predict(signal)
                emoji = FAULT_EMOJIS.get(fault, "?")
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}]  {emoji}  {fault:<14}  conf={conf:.2f}")
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\nStopped.")


# ── Simulated signal sources ─────────────────────────────────────────

def make_simulator(fault_type: str, sample_rate: float = 12_000.0):
    """Return a callable that generates synthetic fault signals."""
    _counter = [0]
    def source(n_samples: int) -> np.ndarray:
        _, sig = generate_signal(fault_type, seed=_counter[0])
        _counter[0] += 1
        return sig[:n_samples]
    return source


def simulate_sequence(engine: BearingInference,
                      fault_sequence: list = None,
                      steps_each: int = 5):
    """Run a simulated fault sequence to demonstrate the pipeline."""
    if fault_sequence is None:
        fault_sequence = ["healthy", "healthy", "outer_race", "inner_race", "ball"]

    print("\n── Simulated Fault Sequence ─────────────────────────────────")
    print(f"{'Step':<6} {'True Fault':<16} {'Predicted':<16} {'Confidence':>10} {'Correct':>8}")
    print("─" * 60)

    correct = 0
    total   = 0

    for fault in fault_sequence:
        for step in range(steps_each):
            _, sig = generate_signal(fault, seed=step * 100)
            pred, conf = engine.predict(sig)
            true_label = fault.replace("_", " ").title()
            is_correct = pred.lower().replace(" ", "_") == fault
            correct += int(is_correct)
            total   += 1
            mark = "✓" if is_correct else "✗"
            print(f"{total:<6} {true_label:<16} {pred:<16} {conf:>10.3f} {mark:>8}")

    print(f"\nAccuracy on simulation: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bearing Fault Inference")
    p.add_argument("--model",    default="models/saved/RandomForest.pkl")
    p.add_argument("--meta",     default="models/saved/metadata.json")
    p.add_argument("--simulate", action="store_true", help="Run simulation demo")
    p.add_argument("--fault",    default="outer_race",
                   choices=["healthy","inner_race","outer_race","ball"],
                   help="Fault type for live simulation")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run 'python run.py' first to train models.")
        sys.exit(1)

    engine = BearingInference(args.model, args.meta)

    if args.simulate:
        simulate_sequence(engine)
    else:
        source = make_simulator(args.fault)
        engine.run_realtime(source, interval_sec=1.0)
