// Portable TFLM inference interface for bearing fault detection.
//
// Pairs with the existing firmware/inference_stub.c feature extractor: that
// code produces the 22-element float feature vector; this code classifies it
// with the int8 neural network running on TensorFlow Lite for Microcontrollers.
#ifndef TFLM_BEARING_INFERENCE_H_
#define TFLM_BEARING_INFERENCE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Initialise the interpreter once at startup. Returns 0 on success, non-zero
// on failure (model schema mismatch or tensor allocation failure).
int bearing_nn_init(void);

// Classify one feature vector.
//   features : kFeatureCount (22) raw, UNSCALED features, in the order listed
//              in model_settings.h. Scaling is applied internally.
//   probs    : optional output buffer of kCategoryCount floats for the
//              per-class probabilities; pass NULL if not needed.
// Returns the predicted class index [0, kCategoryCount), or -1 on error.
int bearing_nn_classify(const float* features, float* probs);

// Human-readable label for a class index (e.g. "Inner Race"), or "?".
const char* bearing_nn_label(int class_index);

// Bytes of the tensor arena actually used after allocation (for tuning
// kTensorArenaSize). Valid only after bearing_nn_init().
int bearing_nn_arena_used(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TFLM_BEARING_INFERENCE_H_
