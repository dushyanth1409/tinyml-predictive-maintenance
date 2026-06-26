# TFLM Neural Network Firmware

This folder runs the int8 bearing-fault neural network on **TensorFlow Lite for
Microcontrollers (TFLM)**. It is the neural-network counterpart to the
RandomForest-via-emlearn path in `../inference_stub.c`.

## Files

| File | Source | Purpose |
|------|--------|---------|
| `bearing_inference.h/.cc` | committed | Portable TFLM setup + `bearing_nn_classify()` |
| `bearing_nn.ino` | committed | Arduino demo sketch |
| `model_data.h/.cc` | generated | The model bytes as a C array |
| `model_settings.h` | generated | Feature count, labels, StandardScaler params |

Generate the three `model_*` files from the repo root:

```bash
python src/models/train_nn.py      # Keras -> int8 .tflite  (+ nn_metadata.json)
python src/models/export_tflm.py   # -> model_data.{h,cc}, model_settings.h
```

## How a prediction is made on-device

1. The feature extractor (`../inference_stub.c`) produces 22 raw features from a
   window of accelerometer samples.
2. `bearing_nn_classify()` applies the frozen StandardScaler
   (`scaled = (raw - mean) / scale`) using constants baked into
   `model_settings.h`. Scaling is done on-device, *outside* the model, so the
   int8 input tensor stays well-conditioned despite the very different feature
   ranges.
3. The scaled vector is quantized to int8 using the model's own input
   scale/zero-point and fed to the interpreter.
4. The int8 output is read back; `argmax` gives the class, and the values are
   dequantized into probabilities.

## Arduino

Put all six files in one sketch folder, install a TFLM library
(`Arduino_TensorFlowLite` or the maintained `Chirale_TensorFlowLite`) via the
Library Manager, select your board, and upload. The serial monitor (115200
baud) prints the prediction and the arena usage — use the latter to shrink
`kTensorArenaSize` in `model_settings.h`.

## STM32 / bare-metal CMake

TFLM is plain C++. Add the TFLM sources (clone `tflite-micro` or use the STM32
X-CUBE-AI / CMSIS-NN bundle) plus this folder's `.cc` files to your build.
CMSIS-NN kernels give the largest speedup on Cortex-M; enable them in the TFLM
build. Call `bearing_nn_init()` once, then `bearing_nn_classify()` per window.

## Tuning

- **Arena size:** start at 8 KB; the firmware reports `arena_used_bytes()`.
  This MLP typically needs only a few KB.
- **Ops:** the resolver registers FullyConnected, Softmax, Relu, Reshape. For
  int8 the activation is fused into FullyConnected, so the real ops are usually
  just FullyConnected + Softmax — the extras are harmless and tiny.
- **Footprint:** the int8 `.tflite` is a few KB; total flash is dominated by
  the TFLM kernels, not the model.
