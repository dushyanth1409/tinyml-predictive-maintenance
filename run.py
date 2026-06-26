"""
Bearing Fault Detection — Main Runner
---------------------------------------
One command trains, evaluates, and demonstrates the full pipeline.

Usage:
    python run.py                    # full pipeline
    python run.py --skip-train       # skip training, use saved models
    python run.py --simulate         # run inference demo after training
    python run.py --samples 300      # samples per class
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.acquisition.simulator import generate_dataset, SignalParams
from src.features.extractor import FeatureExtractor
from src.models.train import train_and_evaluate
from src.models.evaluate import compare_all

MODELS_DIR = "models/saved"


def main(args):
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Bearing Fault Detection — Full Pipeline          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Generate dataset ──────────────────────────────────────────
    if not args.skip_train:
        print(f"[1/4] Generating synthetic dataset ({args.samples} samples/class)...")
        params  = SignalParams(duration=1.0, noise_std=0.05)
        signals, labels = generate_dataset(
            n_samples_per_class=args.samples,
            params=params,
            seed=42
        )
        print(f"      {len(signals)} total signals  |  4 classes  |  "
              f"{params.sample_rate:.0f} Hz  |  {params.duration}s each")

        # ── 2. Extract features ──────────────────────────────────────
        print("\n[2/4] Extracting features (time + frequency domain)...")
        extractor = FeatureExtractor(sample_rate=params.sample_rate)
        X, y = extractor.build_dataset(signals, labels)
        print(f"      Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"      Features: {list(X.columns)}")

        # Save for later
        os.makedirs(MODELS_DIR, exist_ok=True)
        X.to_csv(os.path.join(MODELS_DIR, "features.csv"), index=False)
        np.save(os.path.join(MODELS_DIR, "labels.npy"), y)

        # ── 3. Train and compare models ──────────────────────────────
        print(f"\n[3/4] Training models (5-fold CV)...")
        results = train_and_evaluate(X, y, n_folds=5, output_dir=MODELS_DIR)

        # ── 4. Detailed evaluation ───────────────────────────────────
        print(f"\n[4/4] Detailed evaluation on best model...")
        import pandas as pd
        best_name = results.iloc[0]["model"]
        best_path = os.path.join(MODELS_DIR, f"{best_name}.pkl")

        from src.models.evaluate import evaluate_model
        evaluate_model(best_path, X, y, plot=not args.no_plot)

    else:
        print("[1/4] Skipping training (--skip-train).")
        import pandas as pd
        X = pd.read_csv(os.path.join(MODELS_DIR, "features.csv"))
        y = np.load(os.path.join(MODELS_DIR, "labels.npy"))
        compare_all(MODELS_DIR, X, y)

    # ── 5. Inference demo ────────────────────────────────────────────
    if args.simulate:
        print("\n[5/5] Running inference simulation...")
        best_pkl = os.path.join(MODELS_DIR, "RandomForest.pkl")
        if not os.path.exists(best_pkl):
            import glob
            pkls = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
            if pkls:
                best_pkl = pkls[0]

        from src.inference.inference import BearingInference, simulate_sequence
        engine = BearingInference(
            best_pkl,
            os.path.join(MODELS_DIR, "metadata.json")
        )
        simulate_sequence(engine, steps_each=3)

    print("\n✅  Pipeline complete.")
    print(f"    Models saved to: {MODELS_DIR}/")
    print(f"    Run dashboard:   python src/dashboard/monitor.py")
    print(f"    Run inference:   python src/inference/inference.py --simulate")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bearing Fault Detection Pipeline")
    p.add_argument("--samples",     type=int,  default=200)
    p.add_argument("--skip-train",  action="store_true")
    p.add_argument("--simulate",    action="store_true")
    p.add_argument("--no-plot",     action="store_true")
    main(p.parse_args())
