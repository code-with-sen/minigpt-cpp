// The handful of math kernels GPT-2 needs, for one token at a time.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "thread_pool.hpp"

namespace minigpt {

// int8 numbers are grouped in blocks of 32 that share one scale (the "Q8_0" layout used by
// llama.cpp). Small blocks keep the error low even when a few values are much larger than
// the rest, which happens in GPT-2's activations.
constexpr int kBlock = 32;

inline int n_blocks(int n) { return (n + kBlock - 1) / kBlock; }

// Quantizes n floats into int8 blocks: x[i] ~= q[i] * scale[i / 32], with q in [-127, 127].
inline void quantize_blocks(const float* x, int n, int8_t* q, float* scale) {
    for (int b = 0, start = 0; start < n; ++b, start += kBlock) {
        const int len = std::min(kBlock, n - start);
        float maxabs = 0.f;
        for (int i = 0; i < len; ++i) maxabs = std::max(maxabs, std::fabs(x[start + i]));
        const float s = maxabs > 0.f ? maxabs / 127.f : 1.f;
        scale[b] = s;
        const float inv = 1.f / s;
        for (int i = 0; i < len; ++i) q[start + i] = static_cast<int8_t>(std::lrint(x[start + i] * inv));
    }
}

// A weight matrix stored as [rows][cols] = [out_features][in_features], so each output
// value is one dot product over a contiguous row. Either fp32 or int8 is populated.
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<float> f32;      // rows * cols, when not quantized
    std::vector<int8_t> q8;      // rows * cols, when quantized
    std::vector<float> scale;    // rows * n_blocks(cols), when quantized

    bool quantized() const { return !q8.empty(); }
    size_t bytes() const {
        return f32.size() * sizeof(float) + q8.size() * sizeof(int8_t) + scale.size() * sizeof(float);
    }
    // Reconstructs one row as floats (used for the int8 embedding lookup).
    void dequantize_row(int r, float* out) const {
        const int nb = n_blocks(cols);
        const int8_t* q = &q8[static_cast<size_t>(r) * cols];
        const float* s = &scale[static_cast<size_t>(r) * nb];
        for (int c = 0; c < cols; ++c) out[c] = q[c] * s[c / kBlock];
    }
};

inline void quantize_int8(Matrix& m) {
    const int nb = n_blocks(m.cols);
    m.q8.resize(m.f32.size());
    m.scale.resize(static_cast<size_t>(m.rows) * nb);
    for (int r = 0; r < m.rows; ++r)
        quantize_blocks(&m.f32[static_cast<size_t>(r) * m.cols], m.cols, &m.q8[static_cast<size_t>(r) * m.cols],
                        &m.scale[static_cast<size_t>(r) * nb]);
    std::vector<float>().swap(m.f32);  // free the fp32 copy
}

// Dot products written with independent accumulators so the compiler can vectorize them
// without -ffast-math (which would change results in ways that are hard to test).
inline float dot_f32(const float* a, const float* b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 += a[i] * b[i];         s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
    }
    float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// int8 x int8 dot product over 32-value blocks: integer multiply-adds inside a block, then one
// float multiply-add per block for the two scales. Hand-written SIMD paths for x86 (AVX2) and
// ARM (NEON dot product, e.g. Apple Silicon), with a portable fallback for everything else.
inline float dot_q8(const int8_t* w, const float* ws, const int8_t* x, const float* xs, int n) {
    int b = 0, start = 0;
    float sum = 0.f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (; start + kBlock <= n; ++b, start += kBlock) {
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + start));
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + start));
        // maddubs multiplies unsigned by signed bytes, so move w's sign onto x: |w| * (x * sign(w)).
        const __m256i pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        const __m256 block = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));  // 8 int32 partial sums
        acc = _mm256_add_ps(acc, _mm256_mul_ps(block, _mm256_set1_ps(ws[b] * xs[b])));
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    sum = _mm_cvtss_f32(lo);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t acc = vdupq_n_f32(0.f);
    for (; start + kBlock <= n; ++b, start += kBlock) {
        int32x4_t p = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + start), vld1q_s8(x + start));
        p = vdotq_s32(p, vld1q_s8(w + start + 16), vld1q_s8(x + start + 16));
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ws[b] * xs[b]);
    }
    sum = vaddvq_f32(acc);
#else
    for (; start + kBlock <= n; ++b, start += kBlock) {
        int32_t acc = 0;
        for (int i = 0; i < kBlock; ++i) acc += static_cast<int32_t>(w[start + i]) * x[start + i];
        sum += static_cast<float>(acc) * ws[b] * xs[b];
    }
#endif
    if (start < n) {
        int32_t acc = 0;
        for (int i = start; i < n; ++i) acc += static_cast<int32_t>(w[i]) * x[i];
        sum += static_cast<float>(acc) * ws[b] * xs[b];
    }
    return sum;
}

// Scratch space for the quantized input vector of an int8 matvec.
struct QuantBuffer {
    std::vector<int8_t> q;
    std::vector<float> scale;
};

// out[r] = dot(W[r], x) + bias[r] for every row r, split across threads by row.
inline void matvec(ThreadPool& pool, const Matrix& W, const float* bias, const float* x, float* out,
                   QuantBuffer& qbuf) {
    const int cols = W.cols;
    if (W.quantized()) {
        const int nb = n_blocks(cols);
        qbuf.q.resize(cols);
        qbuf.scale.resize(nb);
        quantize_blocks(x, cols, qbuf.q.data(), qbuf.scale.data());
        const int8_t* xq = qbuf.q.data();
        const float* xs = qbuf.scale.data();
        pool.parallel_for(W.rows, [&](size_t begin, size_t end) {
            for (size_t r = begin; r < end; ++r) {
                float v = dot_q8(&W.q8[r * cols], &W.scale[r * nb], xq, xs, cols);
                out[r] = bias ? v + bias[r] : v;
            }
        });
    } else {
        pool.parallel_for(W.rows, [&](size_t begin, size_t end) {
            for (size_t r = begin; r < end; ++r) {
                float v = dot_f32(x, &W.f32[r * cols], cols);
                out[r] = bias ? v + bias[r] : v;
            }
        });
    }
}

inline void layernorm(const float* x, const float* gamma, const float* beta, float* out, int n,
                      float eps = 1e-5f) {
    double mean = 0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int i = 0; i < n; ++i) out[i] = (x[i] - static_cast<float>(mean)) * inv * gamma[i] + beta[i];
}

// GPT-2 uses the tanh approximation of GELU ("gelu_new" in HuggingFace).
inline float gelu(float x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

inline void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    float inv = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace minigpt
