"""
Terminal Dashboard
-------------------
Real-time terminal display for bearing fault monitoring.
Shows fault status, signal stats, and rolling history.

Usage:
    python src/dashboard/monitor.py --model models/saved/RandomForest.pkl
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


FAULT_COLORS = {
    "Healthy":    "\033[92m",   # green
    "Inner Race": "\033[93m",   # yellow
    "Outer Race": "\033[91m",   # red
    "Ball":       "\033[33m",   # orange/dark yellow
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
CLEAR  = "\033[2J\033[H"


def bar(value: float, max_val: float = 1.0, width: int = 20, char: str = "█") -> str:
    filled = int((value / max_val) * width)
    filled = max(0, min(filled, width))
    return char * filled + "░" * (width - filled)


def render_dashboard(history: deque, current_fault: str, current_conf: float,
                     rms_val: float, kurtosis_val: float, step: int):
    print(CLEAR, end="")

    # ── Header ────────────────────────────────────────────────────────
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  🔧  Bearing Fault Monitor  │  Step {step:04d}{RESET}")
    print(f"{'═'*60}")

    # ── Current Status ────────────────────────────────────────────────
    color = FAULT_COLORS.get(current_fault, "")
    print(f"\n  Status:   {BOLD}{color}{current_fault:<16}{RESET}  "
          f"Confidence: {current_conf*100:5.1f}%")
    print(f"  Conf bar: [{bar(current_conf, 1.0, 30)}]")

    # ── Signal Metrics ────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Signal Metrics")
    print(f"  {'─'*56}")
    rms_norm = min(rms_val / 2.0, 1.0)
    kurt_norm = min(kurtosis_val / 10.0, 1.0)
    print(f"  RMS      {rms_val:6.3f}  [{bar(rms_norm, 1.0, 25)}]")
    print(f"  Kurtosis {kurtosis_val:6.3f}  [{bar(kurt_norm, 1.0, 25)}]  "
          f"{'⚠️  IMPULSIVE' if kurtosis_val > 4 else 'nominal'}")

    # ── History ───────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Recent Predictions (last {len(history)})")
    print(f"  {'─'*56}")
    for i, (fault, conf) in enumerate(list(history)[-10:]):
        color = FAULT_COLORS.get(fault, "")
        marker = "▶" if i == len(list(history)[-10:]) - 1 else " "
        print(f"  {marker} {color}{fault:<14}{RESET}  {conf*100:5.1f}%  "
              f"[{bar(conf, 1.0, 15)}]")

    # ── Fault count summary ───────────────────────────────────────────
    all_faults = [f for f, _ in history]
    if all_faults:
        print(f"\n  {'─'*56}")
        print(f"  Fault Distribution ({len(all_faults)} samples)")
        print(f"  {'─'*56}")
        for fault in ["Healthy", "Inner Race", "Outer Race", "Ball"]:
            count = all_faults.count(fault)
            pct   = count / len(all_faults)
            color = FAULT_COLORS.get(fault, "")
            print(f"  {color}{fault:<14}{RESET} {count:3d}  [{bar(pct, 1.0, 20)}]  {pct*100:5.1f}%")

    print(f"\n  {'─'*56}")
    print(f"  {BOLD}Press Ctrl+C to stop{RESET}")


def run_dashboard(model_path: str, fault_sequence: list = None,
                  interval: float = 1.0):
    from src.inference.inference import BearingInference, make_simulator
    from src.features.time_domain import rms as compute_rms, kurtosis_val as compute_kurtosis

    if fault_sequence is None:
        import itertools
        fault_sequence = itertools.cycle(
            ["healthy"] * 5 + ["outer_race"] * 3 + ["healthy"] * 2 +
            ["inner_race"] * 3 + ["healthy"] * 2 + ["ball"] * 3
        )

    engine  = BearingInference(model_path)
    history = deque(maxlen=50)
    step    = 0

    try:
        for fault in fault_sequence:
            from src.acquisition.simulator import generate_signal
            _, sig = generate_signal(fault, seed=step)
            pred, conf = engine.predict(sig)
            rms_val  = float(np.sqrt(np.mean(sig ** 2)))
            from scipy.stats import kurtosis
            kurt_val = float(kurtosis(sig, fisher=False))

            history.append((pred, conf))
            render_dashboard(history, pred, conf, rms_val, kurt_val, step)
            step += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bearing Fault Monitor Dashboard")
    p.add_argument("--model",    default="models/saved/RandomForest.pkl")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run 'python run.py' first.")
        sys.exit(1)

    run_dashboard(args.model, interval=args.interval)
