/*
 * arduino_inference.cpp -- Auto-generated on 2026-03-08 15:17:40 UTC
 *
 * INT8 fixed-point inference for MEGStrokeNet.
 * Architecture: Conv1d->Conv1d->Conv1d->GAP->Dense->Dense->Sigmoid
 * Input: 6 channels x 100 timepoints
 * Output: 3 valve control values [0.0, 1.0]
 *
 * Uses Q7.8 arithmetic, minimal dynamic allocation.
 */

#include "model_weights.h"
#include <stdint.h>
#include <string.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
  #define READ_WEIGHT(addr)  ((int8_t)pgm_read_byte(addr))
#else
  #define READ_WEIGHT(addr)  (*(addr))
#endif

/* ------------------------------------------------------------------ */
/* Static activation buffers                                          */
/* ------------------------------------------------------------------ */
static int16_t buf_a[800];   /* reusable buffer A */
static int16_t buf_b[800];   /* reusable buffer B */
static int16_t pool_buf[16];
static int16_t fc1_buf[8];
static int16_t fc2_buf[3];

/* ------------------------------------------------------------------ */
/* Fixed-point helpers (Q7.8)                                          */
/* ------------------------------------------------------------------ */

static inline int16_t fp_relu(int16_t x) {
    return (x > 0) ? x : 0;
}

static inline int16_t fp_sigmoid(int16_t x) {
    const int16_t NEG4 = -4 * (1 << FIXED_FRAC_BITS);
    const int16_t POS4 =  4 * (1 << FIXED_FRAC_BITS);
    const int16_t ONE  =  1 * (1 << FIXED_FRAC_BITS);
    const int16_t HALF =      (1 << (FIXED_FRAC_BITS - 1));
    if (x <= NEG4) return 0;
    if (x >= POS4) return ONE;
    return (int16_t)(HALF + (x >> 3));
}

/* ------------------------------------------------------------------ */
/* Conv1D with stride                                                 */
/* ------------------------------------------------------------------ */
static void conv1d_stride(
    const int16_t *in_data, int16_t in_ch, int16_t in_len,
    const int8_t *weights, const int8_t *bias,
    int16_t out_ch, int16_t kernel, int16_t stride, int16_t pad,
    int16_t scale_q8,
    int16_t *out_data, int16_t out_len)
{
    for (int16_t f = 0; f < out_ch; f++) {
        int16_t b = (int16_t)READ_WEIGHT(&bias[f]) << FIXED_FRAC_BITS;
        for (int16_t t = 0; t < out_len; t++) {
            int32_t accum = (int32_t)b;
            int16_t t_in_start = t * stride - pad;
            for (int16_t c = 0; c < in_ch; c++) {
                for (int16_t k = 0; k < kernel; k++) {
                    int16_t tt = t_in_start + k;
                    if (tt < 0 || tt >= in_len) continue;
                    int16_t w_idx = (f * in_ch + c) * kernel + k;
                    int16_t w = (int16_t)READ_WEIGHT(&weights[w_idx]) << FIXED_FRAC_BITS;
                    int16_t x = in_data[c * in_len + tt];
                    accum += ((int32_t)w * (int32_t)x) >> FIXED_FRAC_BITS;
                }
            }
            int32_t scaled = ((int32_t)(int16_t)(accum) * (int32_t)scale_q8) >> FIXED_FRAC_BITS;
            if (scaled >  32767) scaled =  32767;
            if (scaled < -32768) scaled = -32768;
            out_data[f * out_len + t] = (int16_t)scaled;
        }
    }
}

static void relu_inplace(int16_t *data, int16_t n) {
    for (int16_t i = 0; i < n; i++) {
        data[i] = fp_relu(data[i]);
    }
}

static void global_avg_pool(
    const int16_t *in_data, int16_t channels, int16_t seq_len,
    int16_t *out_data)
{
    for (int16_t c = 0; c < channels; c++) {
        int32_t sum = 0;
        for (int16_t t = 0; t < seq_len; t++) {
            sum += in_data[c * seq_len + t];
        }
        out_data[c] = (int16_t)(sum / seq_len);
    }
}

static void dense(
    const int16_t *in_data, int16_t in_features,
    const int8_t *weights, const int8_t *bias,
    int16_t out_features, int16_t scale_q8,
    int16_t *out_data)
{
    for (int16_t o = 0; o < out_features; o++) {
        int32_t accum = (int32_t)((int16_t)READ_WEIGHT(&bias[o]) << FIXED_FRAC_BITS);
        for (int16_t i = 0; i < in_features; i++) {
            int16_t w = (int16_t)READ_WEIGHT(&weights[o * in_features + i]) << FIXED_FRAC_BITS;
            accum += ((int32_t)w * (int32_t)in_data[i]) >> FIXED_FRAC_BITS;
        }
        int32_t scaled = ((int32_t)(int16_t)(accum) * (int32_t)scale_q8) >> FIXED_FRAC_BITS;
        if (scaled >  32767) scaled =  32767;
        if (scaled < -32768) scaled = -32768;
        out_data[o] = (int16_t)scaled;
    }
}

/* ------------------------------------------------------------------ */
/* Main inference entry point                                          */
/* ------------------------------------------------------------------ */
void meg_inference(const int8_t *input, float *output) {
    /* Convert input to Q7.8 */
    for (int16_t i = 0; i < NUM_MEG_CHANNELS * NUM_TIMEPOINTS; i++) {
        buf_a[i] = (int16_t)input[i] << FIXED_FRAC_BITS;
    }

    /* Conv1: (6, 100) -> (16, 50), k=5, s=2, p=2 */
    conv1d_stride(buf_a, NUM_MEG_CHANNELS, NUM_TIMEPOINTS,
                  conv1_weight, conv1_bias,
                  CONV1_OUT_CH, 5, 2, 2,
                  conv1_weight_scale,
                  buf_b, CONV1_OUT_LEN);
    relu_inplace(buf_b, CONV1_OUT_CH * CONV1_OUT_LEN);

    /* Conv2: (16, 50) -> (32, 25), k=3, s=2, p=1 */
    conv1d_stride(buf_b, CONV1_OUT_CH, CONV1_OUT_LEN,
                  conv2_weight, conv2_bias,
                  CONV2_OUT_CH, 3, 2, 1,
                  conv2_weight_scale,
                  buf_a, CONV2_OUT_LEN);
    relu_inplace(buf_a, CONV2_OUT_CH * CONV2_OUT_LEN);

    /* Conv3: (32, 25) -> (16, 13), k=3, s=2, p=1 */
    conv1d_stride(buf_a, CONV2_OUT_CH, CONV2_OUT_LEN,
                  conv3_weight, conv3_bias,
                  CONV3_OUT_CH, 3, 2, 1,
                  conv3_weight_scale,
                  buf_b, CONV3_OUT_LEN);
    relu_inplace(buf_b, CONV3_OUT_CH * CONV3_OUT_LEN);

    /* Global Average Pooling: (16, 13) -> (16,) */
    global_avg_pool(buf_b, CONV3_OUT_CH, CONV3_OUT_LEN, pool_buf);

    /* Dense 1: 16 -> 8 + ReLU */
    dense(pool_buf, CONV3_OUT_CH,
          fc1_weight, fc1_bias, 8,
          fc1_weight_scale,
          fc1_buf);
    relu_inplace(fc1_buf, 8);

    /* Dense 2: 8 -> 3 + Sigmoid */
    dense(fc1_buf, 8,
          fc2_weight, fc2_bias, NUM_OUTPUTS,
          fc2_weight_scale,
          fc2_buf);

    for (int16_t i = 0; i < NUM_OUTPUTS; i++) {
        int16_t s = fp_sigmoid(fc2_buf[i]);
        output[i] = (float)s / (float)(1 << FIXED_FRAC_BITS);
    }
}
