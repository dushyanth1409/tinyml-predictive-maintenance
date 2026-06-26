"""
Neural Network + TensorFlow Lite Micro path for bearing fault detection.
----------------------------------------------------------------------
Trains a small dense network on the SAME 22-feature vectors the rest of
the pipeline uses, then converts it to a fully int8-quantized TFLite model
ready for TensorFlow Lite for Microcontrollers (TFLM).

Design note — scaling lives OUTSIDE the model:
    The 22 features have wildly different ranges (RMS ~1, dominant freq in Hz,
    spectral entropy ~0..1). If we fed raw features to an int8 input tensor,
    quantization would crush the small-range features. So we StandardScale on
    the host, train on scaled features, and bake the scaler mean/scale into
    the firmware (see export_tflm.py -> model_settings.h). The device computes
        scaled[i] = (raw[i] - mean[i]) / scale[i]
    and feeds the scaled vector to the model. This keeps the int8 input tensor
    well-conditioned and preserves accuracy.

Usage:
    python src/models/train_nn.py                 # uses models/saved/features.csv if present
    python src/models/train_nn.py --regenerate     # rebuild dataset from the simulator
    python src/models/train_nn.py --samples 300 --epochs 200

Outputs (into models/saved/):
    nn_float.keras        full-precision Keras model
    nn_int8.tflite        int8-quantized model for TFLM
    nn_metadata.json      class names, feature order, scaler params, accuracies
"""

import argparse
import json
import os
import sys

import numpy as np

# Make "src.*" importable when run from the repo root or from src/models/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODELS_DIR = os.path.join("models", "saved")
DEFAULT_CLASS_ORDER = ["Healthy", "Inner Race", "Outer Race", "Ball"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_or_build_dataset(samples_per_class, regenerate):
    """Return (X_df, y_raw, feature_names).

    Prefers the artifacts already written by run.py (models/saved/features.csv
    and labels.npy). Falls back to regenerating from the simulator + extractor
    so this script also works standalone.
    """
    import pandas as pd

    feat_csv = os.path.join(MODELS_DIR, "features.csv")
    labels_npy = os.path.join(MODELS_DIR, "labels.npy")

    if not regenerate and os.path.exists(feat_csv) and os.path.exists(labels_npy):
        print(f"[data] Loading cached features from {feat_csv}")
        X = pd.read_csv(feat_csv)
        y = np.load(labels_npy, allow_pickle=True)
        return X, y, list(X.columns)

    print("[data] Regenerating dataset from simulator + feature extractor...")
    from src.acquisition.simulator import generate_dataset, SignalParams
    from src.features.extractor import FeatureExtractor

    params = SignalParams(duration=1.0, noise_std=0.05)
    signals, labels = generate_dataset(
        n_samples_per_class=samples_per_class, params=params, seed=42
    )
    extractor = FeatureExtractor(sample_rate=params.sample_rate)
    X, y = extractor.build_dataset(signals, labels)

    os.makedirs(MODELS_DIR, exist_ok=True)
    X.to_csv(feat_csv, index=False)
    np.save(labels_npy, y)
    return X, y, list(X.columns)


def resolve_class_order(y_raw):
    """Return the ordered list of class names and an int-encoded y.

    If models/saved/metadata.json exists (written by the existing tree-model
    trainer) we reuse its class order so the NN and the RF agree on label
    indices. Otherwise we use a sensible default if the labels match, else
    fall back to sorted unique labels.
    """
    from sklearn.preprocessing import LabelEncoder

    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    preferred = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            preferred = meta.get("class_names") or meta.get("classes")
        except (json.JSONDecodeError, OSError):
            preferred = None

    uniques = list(np.unique(y_raw))

    # If labels are already integers 0..K-1, map them through a name list.
    if np.issubdtype(np.asarray(y_raw).dtype, np.integer):
        k = int(max(uniques)) + 1
        if preferred and len(preferred) == k:
            class_names = list(preferred)
        elif k == len(DEFAULT_CLASS_ORDER):
            class_names = list(DEFAULT_CLASS_ORDER)
        else:
            class_names = [f"class_{i}" for i in range(k)]
        y_int = np.asarray(y_raw, dtype=np.int64)
        return class_names, y_int

    # String labels -> encode. Prefer the existing metadata order if it covers
    # every observed label.
    if preferred and set(uniques).issubset(set(preferred)):
        class_names = [c for c in preferred if c in set(uniques)]
    elif set(uniques).issubset(set(DEFAULT_CLASS_ORDER)):
        class_names = [c for c in DEFAULT_CLASS_ORDER if c in set(uniques)]
    else:
        class_names = sorted(str(u) for u in uniques)

    index = {name: i for i, name in enumerate(class_names)}
    y_int = np.array([index[str(v)] for v in y_raw], dtype=np.int64)
    # Sanity check via LabelEncoder (catches stray labels).
    LabelEncoder().fit(y_raw)
    return class_names, y_int


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(n_features, n_classes, seed=42):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,), name="features"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax", name="fault"),
        ],
        name="bearing_fault_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# int8 conversion + evaluation
# --------------------------------------------------------------------------- #
def convert_int8(model, x_scaled_repr):
    """Full-integer (int8 in / int8 out) TFLite conversion."""
    import tensorflow as tf

    def representative_dataset():
        # A few hundred real (scaled) feature vectors calibrate the ranges.
        n = min(300, x_scaled_repr.shape[0])
        for i in range(n):
            yield [x_scaled_repr[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_tflite_int8(tflite_model, x_scaled, y_true):
    """Run the int8 model through the TFLite interpreter and return accuracy."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]

    preds = np.empty(len(y_true), dtype=np.int64)
    for i, x in enumerate(x_scaled):
        q = np.round(x / in_scale + in_zp)
        q = np.clip(q, -128, 127).astype(np.int8).reshape(in_d["shape"])
        interpreter.set_tensor(in_d["index"], q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_d["index"])[0]
        preds[i] = int(np.argmax(out))

    acc = float(np.mean(preds == y_true))
    quant = {
        "input": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output": {
            "scale": float(out_d["quantization"][0]),
            "zero_point": int(out_d["quantization"][1]),
        },
    }
    return acc, quant


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print(" Neural Network -> TensorFlow Lite Micro  (bearing faults)")
    print("=" * 60)

    X_df, y_raw, feature_names = load_or_build_dataset(args.samples, args.regenerate)
    class_names, y = resolve_class_order(y_raw)
    X = X_df.to_numpy(dtype=np.float32)

    print(f"[data] {X.shape[0]} samples x {X.shape[1]} features")
    print(f"[data] classes: {class_names}")

    # Stratified split: train / val / test (60 / 20 / 20).
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # Scaler is fit on TRAIN ONLY, then frozen and exported to firmware.
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    import tensorflow as tf

    model = build_model(X.shape[1], len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
    ]
    model.fit(
        X_tr_s,
        y_tr,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    float_loss, float_acc = model.evaluate(X_te_s, y_te, verbose=0)
    print(f"\n[float] test accuracy: {float_acc:.4f}")

    # ---- int8 conversion ------------------------------------------------- #
    tflite_model = convert_int8(model, X_tr_s)
    int8_acc, quant = evaluate_tflite_int8(tflite_model, X_te_s, y_te)
    print(f"[int8 ] test accuracy: {int8_acc:.4f}  "
          f"(delta {int8_acc - float_acc:+.4f})")
    print(f"[int8 ] model size: {len(tflite_model)} bytes "
          f"({len(tflite_model) / 1024:.1f} KB)")

    # ---- save ------------------------------------------------------------ #
    keras_path = os.path.join(MODELS_DIR, "nn_float.keras")
    tflite_path = os.path.join(MODELS_DIR, "nn_int8.tflite")
    meta_path = os.path.join(MODELS_DIR, "nn_metadata.json")

    model.save(keras_path)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    metadata = {
        "model": "nn_int8",
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "quant": quant,
        "float_test_accuracy": float(float_acc),
        "int8_test_accuracy": float(int8_acc),
        "tflite_bytes": int(len(tflite_model)),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(f"  {keras_path}")
    print(f"  {tflite_path}")
    print(f"  {meta_path}")
    print("\nNext: python src/models/export_tflm.py   "
          "# generate the C arrays for firmware")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train NN + export int8 TFLite")
    p.add_argument("--samples", type=int, default=200,
                   help="samples per class when regenerating the dataset")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--regenerate", action="store_true",
                   help="rebuild the dataset from the simulator instead of cache")
    main(p.parse_args())
