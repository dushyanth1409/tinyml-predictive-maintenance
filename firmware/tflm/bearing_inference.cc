// Portable TFLM inference implementation for bearing fault detection.
//
// Works on any TFLM target (STM32 + CMSIS-NN, ESP32, Arduino, etc.). The only
// platform-specific include is <TensorFlowLite.h>, pulled in automatically on
// Arduino. Build the matching .cc/.h files alongside this one:
//     model_data.cc / model_data.h      (auto-generated)
//     model_settings.h                  (auto-generated)
//
// API drift note: TFLM's MicroInterpreter exposes input(i)/output(i)
// (lower-case) across all recent versions; we use those for portability.

#if defined(ARDUINO)
#include <TensorFlowLite.h>
#endif

#include "bearing_inference.h"

#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "model_settings.h"

namespace {

const tflite::Model* g_tfl_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// One arena, statically allocated. 16-byte aligned for the allocator.
alignas(16) uint8_t g_arena[kTensorArenaSize];

inline int8_t clamp_int8(int32_t v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return static_cast<int8_t>(v);
}

}  // namespace

int bearing_nn_init(void) {
  g_tfl_model = tflite::GetModel(g_model);
  if (g_tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d",
                g_tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Ops used by the dense int8 graph. ReLU is fused into FULLY_CONNECTED for
  // int8, so in practice only FullyConnected + Softmax are needed; Relu and
  // Reshape are registered defensively in case the converter emits them.
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddReshape();

  static tflite::MicroInterpreter interpreter(
      g_tfl_model, resolver, g_arena, kTensorArenaSize);
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return 2;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8) {
    MicroPrintf("Expected int8 input/output tensors");
    return 3;
  }
  if (g_input->dims->data[g_input->dims->size - 1] != kFeatureCount) {
    MicroPrintf("Input width %d != kFeatureCount %d",
                g_input->dims->data[g_input->dims->size - 1], kFeatureCount);
    return 4;
  }
  return 0;
}

int bearing_nn_classify(const float* features, float* probs) {
  if (g_interpreter == nullptr || features == nullptr) return -1;

  // Quantization params come straight from the model's tensors.
  const float in_scale = g_input->params.scale;
  const int in_zp = g_input->params.zero_point;

  // 1) StandardScale on the host-trained mean/scale, then 2) quantize to int8.
  for (int i = 0; i < kFeatureCount; ++i) {
    const float scaled = (features[i] - kFeatureMean[i]) / kFeatureScale[i];
    const int32_t q = static_cast<int32_t>(lroundf(scaled / in_scale)) + in_zp;
    g_input->data.int8[i] = clamp_int8(q);
  }

  if (g_interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    return -1;
  }

  const float out_scale = g_output->params.scale;
  const int out_zp = g_output->params.zero_point;

  int best = 0;
  int8_t best_raw = g_output->data.int8[0];
  for (int c = 0; c < kCategoryCount; ++c) {
    const int8_t raw = g_output->data.int8[c];
    if (raw > best_raw) {
      best_raw = raw;
      best = c;
    }
    if (probs != nullptr) {
      probs[c] = (static_cast<int>(raw) - out_zp) * out_scale;
    }
  }
  return best;
}

const char* bearing_nn_label(int class_index) {
  if (class_index < 0 || class_index >= kCategoryCount) return "?";
  return kCategoryLabels[class_index];
}

int bearing_nn_arena_used(void) {
  if (g_interpreter == nullptr) return -1;
  return static_cast<int>(g_interpreter->arena_used_bytes());
}
