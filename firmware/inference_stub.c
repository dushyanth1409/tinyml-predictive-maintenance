/**
 * @file    inference_stub.c
 * @brief   Bearing fault inference on bare-metal MCU (STM32/Arduino)
 *
 * This stub shows how to deploy the trained model on an embedded target.
 * Feature computation is done in C; the decision tree is hardcoded
 * (export with sklearn-porter or emlearn from the trained RandomForest).
 *
 * Workflow:
 *   1. Train RandomForest in Python (run.py)
 *   2. Export to C with: pip install emlearn && python export_model.py
 *   3. Replace the stub tree below with the generated code
 *   4. Compile and flash to STM32/Arduino
 *
 * Hardware: MPU-6050 on I2C, 12 kHz sampling
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Config ─────────────────────────────────────────────────────────── */
#define SAMPLE_RATE     12000u
#define WINDOW_SIZE     1024u
#define N_FEATURES      22u
#define N_CLASSES       4u

/* ── Fault labels ────────────────────────────────────────────────────── */
typedef enum {
    FAULT_HEALTHY     = 0,
    FAULT_INNER_RACE  = 1,
    FAULT_OUTER_RACE  = 2,
    FAULT_BALL        = 3,
} FaultClass;

static const char *FAULT_NAMES[N_CLASSES] = {
    "Healthy", "Inner Race", "Outer Race", "Ball"
};

/* ── Feature vector ──────────────────────────────────────────────────── */
typedef struct {
    /* Time domain */
    float rms;
    float mean;
    float variance;
    float std_dev;
    float kurtosis;
    float skewness;
    float peak;
    float peak_to_peak;
    float crest_factor;
    float shape_factor;
    float impulse_factor;
    /* Frequency domain */
    float dominant_freq;
    float spectral_centroid;
    float spectral_spread;
    float band_energy_0_1k;
    float band_energy_1_3k;
    float band_energy_3_6k;
    float fft_peak_1;
    float fft_peak_2;
    float fft_peak_3;
    float spectral_entropy;
    float spectral_flatness;
} FeatureVector;

/* ── Time domain feature extraction ─────────────────────────────────── */
static void extract_time_features(const float *window, uint32_t n, FeatureVector *fv)
{
    float sum = 0.0f, sum_sq = 0.0f, sum_abs = 0.0f;
    float max_val = -1e10f, min_val = 1e10f;
    uint32_t i;

    for (i = 0; i < n; i++) {
        sum    += window[i];
        sum_sq += window[i] * window[i];
        sum_abs += fabsf(window[i]);
        if (window[i] > max_val) max_val = window[i];
        if (window[i] < min_val) min_val = window[i];
    }

    float mu    = sum / n;
    float var   = (sum_sq / n) - (mu * mu);
    float sigma = sqrtf(var > 0 ? var : 0);
    float rms   = sqrtf(sum_sq / n);
    float peak  = fabsf(max_val) > fabsf(min_val) ? fabsf(max_val) : fabsf(min_val);
    float mean_abs = sum_abs / n;

    /* Higher-order moments (4th = kurtosis, 3rd = skewness) */
    float m3 = 0.0f, m4 = 0.0f;
    for (i = 0; i < n; i++) {
        float d = window[i] - mu;
        float d2 = d * d;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m3 /= n; m4 /= n;
    float kurt = (sigma > 1e-10f) ? (m4 / (sigma*sigma*sigma*sigma)) : 0.0f;
    float skew = (sigma > 1e-10f) ? (m3 / (sigma*sigma*sigma)) : 0.0f;

    fv->rms           = rms;
    fv->mean          = mu;
    fv->variance      = var;
    fv->std_dev       = sigma;
    fv->kurtosis      = kurt;
    fv->skewness      = skew;
    fv->peak          = peak;
    fv->peak_to_peak  = max_val - min_val;
    fv->crest_factor  = (rms > 1e-10f) ? peak / rms : 0.0f;
    fv->shape_factor  = (mean_abs > 1e-10f) ? rms / mean_abs : 0.0f;
    fv->impulse_factor= (mean_abs > 1e-10f) ? peak / mean_abs : 0.0f;
}

/* ── Stub: Decision tree inference ───────────────────────────────────── */
/**
 * Replace this stub with the output of:
 *   python -c "
 *   import joblib, emlearn
 *   model = joblib.load('models/saved/RandomForest.pkl')
 *   clf = model.named_steps['clf']
 *   cmodel = emlearn.convert(clf, method='inline')
 *   cmodel.save(file='firmware/model_generated.h')
 *   "
 */
static FaultClass run_inference_stub(const FeatureVector *fv)
{
    /* Simple threshold-based demo — replace with exported model */
    if (fv->kurtosis < 3.5f && fv->crest_factor < 4.0f) {
        return FAULT_HEALTHY;
    }
    if (fv->band_energy_1_3k > fv->band_energy_0_1k) {
        return FAULT_INNER_RACE;
    }
    if (fv->dominant_freq > 90.0f && fv->dominant_freq < 130.0f) {
        return FAULT_OUTER_RACE;
    }
    return FAULT_BALL;
}

/* ── Public API ──────────────────────────────────────────────────────── */
FaultClass bearing_predict(const float *window, uint32_t n)
{
    FeatureVector fv;
    memset(&fv, 0, sizeof(fv));
    extract_time_features(window, n, &fv);
    /* TODO: add FFT-based frequency features */
    return run_inference_stub(&fv);
}

const char *bearing_fault_name(FaultClass cls)
{
    if (cls < N_CLASSES) return FAULT_NAMES[cls];
    return "Unknown";
}

/* ── Arduino-style main loop example ─────────────────────────────────── */
/*
void loop(void) {
    float window[WINDOW_SIZE];
    collect_imu_samples(window, WINDOW_SIZE);    // your IMU read function
    FaultClass fault = bearing_predict(window, WINDOW_SIZE);
    Serial.println(bearing_fault_name(fault));
    delay(500);
}
*/
