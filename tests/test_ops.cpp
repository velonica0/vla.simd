#include "hal/arch.h"
#include "hal/common/layout.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include "ops/quant_ops.h"
#include "nn/attention.h"
#include "nn/conv.h"
#include "nn/encoder.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace tcpu;

static int failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static uint64_t state = 0x243F6A8885A308D3ull;
static float frand(float lo, float hi) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return lo + (hi - lo)*(float)((state >> 40) & 0xFFFFFF)/16777216.0f;
}
static std::vector<float> rv(size_t n, float lo = -1.f, float hi = 1.f) {
    std::vector<float> v(n);
    for (float& x : v) x = frand(lo, hi);
    return v;
}
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void naive_linear(std::vector<double>& y, std::vector<double>& mag, const float* x,
                         const float* W, const float* b, int seq, int N, int K) {
    y.assign((size_t)seq*N, 0.0); mag.assign((size_t)seq*N, 0.0);
    for (int t = 0; t < seq; t++)
        for (int n = 0; n < N; n++) {
            double s = b ? b[n] : 0.0, m = b ? std::fabs(b[n]) : 0.0;
            for (int k = 0; k < K; k++) {
                const double p = (double)x[(size_t)t*K+k]*W[(size_t)n*K+k];
                s += p; m += std::fabs(p);
            }
            y[(size_t)t*N+n] = s; mag[(size_t)t*N+n] = m;
        }
}

static double scaled_error(const std::vector<float>& got, const std::vector<double>& ref,
                           const std::vector<double>& mag) {
    if (got.size() != ref.size() || got.size() != mag.size()) return INFINITY;
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) ||
            !std::isfinite(mag[i]) || mag[i] < 0.0) return INFINITY;
        const double e = std::fabs(got[i] - ref[i])/(mag[i] + 1e-30);
        if (e > worst) worst = e;
    }
    return worst;
}

static void expect_close(const char* what, const std::vector<float>& got, const std::vector<double>& ref,
                         const std::vector<double>& mag, double rel) {
    const double worst = scaled_error(got, ref, mag);
    CHECK(worst <= rel, "%s: worst scaled error %.3e > %.1e", what, worst, rel);
}

static void test_linear() {
    {
        const int width = std::numeric_limits<int>::max() / (255*127) + 1;
        std::vector<float> weights((size_t)16*width, 1), input(width, 1), out(16);
        nn::Linear linear;
        linear.init(weights.data(), nullptr, 16, width, nn::Linear::Role::Generic);
        CHECK(!linear.init_int8(), "unsafe INT8 linear reduction accepted");
        linear.forward(out.data(), input.data(), 1);
        for (float value : out) CHECK(value == width, "wide linear fallback changed output");
        nn::Conv2d conv;
        conv.init(weights.data(), nullptr, 16, 1, width);
        CHECK(!conv.init_int8(), "unsafe INT8 convolution reduction accepted");
        conv.forward(out.data(), input.data(), 1, 1, 1, 0);
        for (float value : out) CHECK(value == width, "wide convolution fallback changed output");
    }
    for (float magnitude : {0.0f, std::numeric_limits<float>::denorm_min(), 1e-37f, 1.0f}) {
        for (int width : {3, 16, 33}) {
            std::vector<float> x(width);
            for (int i=0; i<width; i++) x[i] = (i % 2 ? -1 : 1)*magnitude;
            std::vector<int8_t> q(i8_kpad(width));
            float scale = 0;
            quantize_act_i8(x.data(), q.data(), &scale, 1, width);
            CHECK(std::isfinite(scale) && scale > 0, "invalid quantization scale");
            for (int i=0; i<width; i++) {
                CHECK(q[i] >= -127, "quantization escaped symmetric range");
                CHECK(std::fabs((double)q[i]*scale-x[i]) <= (double)scale*.51,
                      "quantization error exceeds half a bin");
            }
            for (int i=width; i<i8_kpad(width); i++) CHECK(q[i] == 0, "nonzero quantization padding");
            std::vector<float> weights((size_t)16*width, magnitude), scales(16);
            std::vector<int32_t> packed(packed_i8_words(16, width));
            pack_weights_i8(weights.data(), (int8_t*)packed.data(), scales.data(), 16, width);
            for (float s : scales) CHECK(std::isfinite(s) && s > 0, "invalid weight scale");
        }
    }
    const int shapes[][3] = {{1, 16, 3}, {7, 48, 64}, {50, 768, 1027}, {13, 64, 17}, {257, 96, 48}};
    for (auto& s : shapes) {
        const int seq = s[0], N = s[1], K = s[2];
        auto x = rv((size_t)seq*K), W = rv((size_t)N*K, -.2f, .2f), b = rv(N);
        std::vector<double> ref, mag;
        naive_linear(ref, mag, x.data(), W.data(), b.data(), seq, N, K);
        std::vector<float> y((size_t)seq*N);
        dense_linear(y.data(), x.data(), W.data(), b.data(), seq, N, K);
        expect_close("dense_linear", y, ref, mag, 1e-5);

        if (N % 16) continue;
        std::vector<float> Wp((size_t)N*K), yp((size_t)seq*N);
        pack_weights16(W.data(), Wp.data(), N, K);
        dense_linear_packed(yp.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        expect_close("dense_linear_packed", yp, ref, mag, 1e-5);

        // bf16 panels: reference on the bf16-rounded weights. The RVV Zvfbfwma
        // kernel also rounds x to bf16 (unit roundoff 2^-8); every other kernel
        // keeps x in fp32.
        {
            std::vector<uint16_t> Wpb((size_t)N*K);
            std::vector<float> Wr((size_t)N*K), yb((size_t)seq*N);
            for (size_t i = 0; i < Wr.size(); i++) {
                uint32_t u;
                std::memcpy(&u, &W[i], 4);
                const uint16_t h = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
                Wr[i] = bf16_f32(h);
            }
            for (int bl = 0; bl < N/16; bl++)
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < 16; j++) {
                        uint32_t u;
                        std::memcpy(&u, &Wr[(size_t)(bl*16+j)*K+k], 4);
                        Wpb[((size_t)bl*K+k)*16+j] = (uint16_t)(u >> 16);
                    }
            std::vector<double> rb, mb;
            naive_linear(rb, mb, x.data(), Wr.data(), b.data(), seq, N, K);
            dense_linear_packed_bf16(yb.data(), x.data(), Wpb.data(), b.data(), seq, N, K);
#if TCPU_HAL_RVV && defined(__riscv_zvfbfwma)
            expect_close("dense_linear_packed_bf16", yb, rb, mb, 4e-3);
#else
            expect_close("dense_linear_packed_bf16", yb, rb, mb, 1e-5);
#endif
        }

        // IME2 fp16 tiles (TCPU_IME=1 on the K3's A100 cores only): x and W are
        // rounded to fp16 (unit roundoff 2^-11 each), accumulation is fp32.
        if (ime_available()) {
            std::vector<uint16_t> Wh(ime_packed_size(N, K));
            std::vector<float> yh((size_t)seq*N);
            CHECK(pack_weights_ime(W.data(), Wh.data(), N, K), "pack_weights_ime refused fp16-range weights");
            dense_linear_ime(yh.data(), x.data(), Wh.data(), b.data(), seq, N, K);
            expect_close("dense_linear_ime", yh, ref, mag, 2e-3);
        }

        const int ldo = hal::kt_stride(seq);
        std::vector<float> kt((size_t)N*ldo, 0.f), ktt((size_t)seq*N);
        dense_linear_packed_kt(kt.data(), x.data(), Wp.data(), b.data(), seq, N, K, ldo);
        for (int n = 0; n < N; n++)
            for (int t = 0; t < seq; t++) ktt[(size_t)t*N+n] = kt[(size_t)n*ldo+t];
        CHECK(same_bits(ktt, yp), "dense_linear_packed_kt != dense_linear_packed (%d,%d,%d)", seq, N, K);

#if TCPU_HAL_X86
        std::vector<float> g((size_t)seq*N), gg(yp);
        dense_linear_packed_gelu(g.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        gelu_tanh(gg.data(), seq*N);
        CHECK(same_bits(g, gg), "dense_linear_packed_gelu != gelu_tanh(packed) (%d,%d,%d)", seq, N, K);

        auto acc = rv((size_t)seq*N), sum(acc);
        dense_linear_packed_add(acc.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        for (size_t i = 0; i < sum.size(); i++) sum[i] += yp[i];
        CHECK(same_bits(acc, sum), "dense_linear_packed_add != out + packed (%d,%d,%d)", seq, N, K);
#endif

        std::vector<int8_t> Wq(packed_i8_words(N, K)*4), xq((size_t)seq*i8_kpad(K));
        std::vector<float> ws(N), as(seq), yq((size_t)seq*N), yr((size_t)seq*N);
        pack_weights_i8(W.data(), Wq.data(), ws.data(), N, K);
        quantize_act_i8(x.data(), xq.data(), as.data(), seq, K);
        dense_linear_i8_ref(yr.data(), xq.data(), as.data(), Wq.data(), ws.data(), b.data(), seq, N, K);
        expect_close("dense_linear_i8_ref", yr, ref, mag, 2e-2);
        if (int8_gemm_available()) {
            dense_linear_i8_pre(yq.data(), xq.data(), as.data(), Wq.data(), ws.data(), b.data(), seq, N, K);
            CHECK(same_bits(yq, yr), "dense_linear_i8_pre != dense_linear_i8_ref (%d,%d,%d)", seq, N, K);
        }
    }
}

static void naive_attention(std::vector<double>& out, const float* Q, const float* K, const float* V,
                            const float* mask, int sq, int sk, int nq, int nkv, int hd, float scale) {
    out.assign((size_t)sq*nq*hd, 0.0);
    std::vector<double> s(sk);
    for (int h = 0; h < nq; h++) {
        const int g = h/(nq/nkv);
        for (int i = 0; i < sq; i++) {
            double mx = -INFINITY;
            for (int j = 0; j < sk; j++) {
                double d = 0.0;
                for (int c = 0; c < hd; c++)
                    d += (double)Q[((size_t)i*nq+h)*hd+c]*K[((size_t)j*nkv+g)*hd+c];
                s[j] = d*scale + (mask ? mask[(size_t)i*sk+j] : 0.0);
                if (s[j] > mx) mx = s[j];
            }
            double z = 0.0;
            for (int j = 0; j < sk; j++) { s[j] = std::isinf(s[j]) ? 0.0 : std::exp(s[j]-mx); z += s[j]; }
            for (int c = 0; c < hd; c++) {
                double a = 0.0;
                for (int j = 0; j < sk; j++) a += s[j]*V[((size_t)j*nkv+g)*hd+c];
                out[((size_t)i*nq+h)*hd+c] = a/z;
            }
        }
    }
}

static void test_attention() {
    if (int8_gemm_available()) {
        {
            nn::Linear linear;
            std::vector<float> weights(16*16, 1), input(16, 1), out(16);
            std::vector<uint16_t> zero(16*16);
            linear.init(weights.data(), nullptr, 16, 16, nn::Linear::Role::Gemm);
            CHECK(linear.init_int8(), "int8 init failed");
            linear.init_bf16(zero.data(), nullptr, 16, 16, nn::Linear::Role::Gemm);
            linear.forward(out.data(), input.data(), 1);
            CHECK(!linear.is_int8(), "BF16 reinit retained INT8 state");
            for (float value : out) CHECK(value == 0, "BF16 reinit used old weights");
        }
        const int dim = 32, seq = 7;
        auto weights = rv(dim*dim), bias = rv(dim);
        nn::MhaQKV layer;
        layer.set_shape(4, dim/4);
        for (nn::Linear* linear : {&layer.wq, &layer.wk, &layer.wv, &layer.wo}) {
            linear->init(weights.data(), bias.data(), dim, dim, nn::Linear::Role::Gemm);
            CHECK(linear->init_int8(), "int8 attention init failed");
        }
        auto source = rv(seq*dim), qcopy = source, kcopy = source, vcopy = source;
        nn::Scratch shared, separate;
        for (int rows : {seq, 1}) {
            for (int sharing = 0; sharing < 4; sharing++) {
                std::vector<float> actual(rows*dim), expected(rows*dim);
                const float* q = sharing & 1 ? source.data() : qcopy.data();
                const float* v = sharing & 2 ? source.data() : vcopy.data();
                layer.forward(actual.data(), q, source.data(), v, rows, seq, shared);
                layer.forward(expected.data(), qcopy.data(), kcopy.data(), vcopy.data(), rows, seq, separate);
                CHECK(same_bits(actual, expected), "shared quantization changed attention");
            }
        }
    }
    const int cfgs[][5] = {{1, 50, 8, 2, 64}, {50, 50, 8, 2, 64}, {50, 113, 15, 5, 64},
                           {17, 33, 4, 4, 72}, {9, 20, 2, 1, 256}, {256, 256, 12, 12, 64},
                           {12, 512, 4, 2, 64}};
    const float NINF = -std::numeric_limits<float>::infinity();
    for (auto& c : cfgs) {
        const int sq = c[0], sk = c[1], nq = c[2], nkv = c[3], hd = c[4];
        auto Q = rv((size_t)sq*nq*hd), K = rv((size_t)sk*nkv*hd), V = rv((size_t)sk*nkv*hd);
        const float scale = 1.0f/std::sqrt((float)hd);
        // IME attention rounds Q, K, V and the softmax weights to fp16.
        const double tol = ime_attention_enabled() && hd % 16 == 0 ? 3e-3 : 2e-6;
        std::vector<float> dense((size_t)sq*nq*hd), masked(dense.size()), zero((size_t)sq*sk, 0.f);
        gqa_attention_dense(dense.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale);
        gqa_attention_masked(masked.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale, zero.data());
        CHECK(same_bits(dense, masked), "dense != masked(zero) (%d,%d,%d,%d,%d)", sq, sk, nq, nkv, hd);

        std::vector<double> ref;
        naive_attention(ref, Q.data(), K.data(), V.data(), nullptr, sq, sk, nq, nkv, hd, scale);
        std::vector<double> one(ref.size(), 1.0);
        expect_close("gqa_attention_dense", dense, ref, one, tol);

        std::vector<float> causal((size_t)sq*sk);
        for (int i = 0; i < sq; i++)
            for (int j = 0; j < sk; j++) causal[(size_t)i*sk+j] = j > i + (sk - sq) ? NINF : 0.f;
        gqa_attention_masked(masked.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale, causal.data());
        naive_attention(ref, Q.data(), K.data(), V.data(), causal.data(), sq, sk, nq, nkv, hd, scale);
        expect_close("gqa_attention_masked causal", masked, ref, one, tol);

        const int ldk = hal::kt_stride(sk);
        std::vector<float> KT((size_t)nkv*hd*ldk, 0.f), pre(masked.size());
        for (int h = 0; h < nkv; h++)
            for (int d = 0; d < hd; d++)
                for (int j = 0; j < sk; j++) KT[((size_t)h*hd+d)*ldk+j] = K[((size_t)j*nkv+h)*hd+d];
        gqa_attention_masked(pre.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale,
                             causal.data(), KT.data());
        expect_close("gqa_attention_masked causal K_pre", pre, ref, one, tol);
        CHECK(TCPU_HAL_APPLE || same_bits(pre, masked),
              "masked with K_pre != masked (%d,%d,%d,%d,%d)", sq, sk, nq, nkv, hd);
        for (float block : {NINF, std::numeric_limits<float>::lowest()}) {
            std::fill(causal.begin(), causal.end(), block);
            gqa_attention_masked(masked.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale, causal.data());
            for (int t=0; t<sq; t++)
                for (int h=0; h<nq; h++)
                    for (int d=0; d<hd; d++) {
                        double mean = 0;
                        for (int j=0; j<sk; j++) mean += V[((size_t)j*nkv+h/(nq/nkv))*hd+d]/(double)sk;
                        const float value = masked[((size_t)t*nq+h)*hd+d];
                        CHECK(std::isfinite(value) && std::fabs(value-mean) < tol,
                              "fully masked attention differs from uniform fallback");
                    }
        }
    }
}

// Values past fp16's 65504 must never reach the IME tiles as inf: weights are
// refused at pack time (the layer keeps fp32), activation rows are scaled by a
// power of two, and attention falls back to the fp32 kernel. The checks on
// Linear and attention results run on every backend.
static void test_fp16_range() {
    const int seq = 13, N = 32, K = 40;
    auto x = rv((size_t)seq*K), W = rv((size_t)N*K), b = rv(N);
    for (int k=0; k<K; k++) x[k] *= 1e6f;            // whole row past fp16
    x[3*K + 7] = 3e5f;                                // one outlier
    for (int k=0; k<K; k++) x[5*K + k] *= 1e30f;      // far past it
    x[8*K + 2] = 65504.f;                             // exactly the limit
    std::vector<double> ref, mag;
    naive_linear(ref, mag, x.data(), W.data(), b.data(), seq, N, K);
    if (ime_available()) {
        std::vector<uint16_t> Wh(ime_packed_size(N, K));
        std::vector<float> y((size_t)seq*N);
        CHECK(pack_weights_ime(W.data(), Wh.data(), N, K), "pack_weights_ime refused fp16-range weights");
        dense_linear_ime(y.data(), x.data(), Wh.data(), b.data(), seq, N, K);
        expect_close("dense_linear_ime large activations", y, ref, mag, 2e-3);
    }
    // A weight past fp16: IME refuses it and Linear stays fp32 (error ~1e-7).
    W[5*K + 11] = 1e5f;
    naive_linear(ref, mag, x.data(), W.data(), b.data(), seq, N, K);
    if (ime_available()) {
        std::vector<uint16_t> Wh(ime_packed_size(N, K));
        CHECK(!pack_weights_ime(W.data(), Wh.data(), N, K), "pack_weights_ime accepted a 1e5 weight");
    }
    {
        nn::Linear linear;
        linear.init(W.data(), b.data(), N, K, nn::Linear::Role::Gemm);
        std::vector<float> y((size_t)seq*N);
        linear.forward(y.data(), x.data(), seq);
        expect_close("Linear with a weight past fp16", y, ref, mag, 1e-5);
    }

    // Attention with one large value in Q, K or V.
    const int sq = 50, sk = 113, nq = 8, nkv = 2, hd = 64;
    const float scale = 1.0f/std::sqrt((float)hd);
    for (int which=0; which<4; which++) {
        auto Q = rv((size_t)sq*nq*hd), Kt = rv((size_t)sk*nkv*hd), V = rv((size_t)sk*nkv*hd);
        double vmag = 1.0, tol = 2e-6;
        if (which == 0) Q[(size_t)(7*nq + 3)*hd + 5] = 1e6f;       // scale*Q = 1.25e5
        if (which == 1) Kt[(size_t)(40*nkv + 1)*hd + 9] = 1e5f;
        if (which == 2) { V[(size_t)(90*nkv + 0)*hd + 33] = 1e5f; vmag = 1e5; }
        // Scores of ~1e4 carry fp32 rounding of ~1e-3 into the weights.
        if (which < 2) tol = 5e-3;
        // which == 3: everything in range; IME (when on) must accept it.
        if (ime_attention_enabled()) {
            std::vector<float> o((size_t)sq*nq*hd);
            const bool took = ime_attention(o.data(), Q.data(), Kt.data(), V.data(),
                                            sq, sk, nq, nkv, hd, scale, nullptr);
            CHECK(took == (which == 3), "ime_attention %s case %d", took ? "accepted" : "refused", which);
            if (which == 3) tol = 3e-3;
        }
        std::vector<float> got((size_t)sq*nq*hd), zero((size_t)sq*sk, 0.f);
        gqa_attention_masked(got.data(), Q.data(), Kt.data(), V.data(), sq, sk, nq, nkv, hd, scale, zero.data());
        std::vector<double> aref, amag(got.size(), vmag);
        naive_attention(aref, Q.data(), Kt.data(), V.data(), nullptr, sq, sk, nq, nkv, hd, scale);
        char what[64];
        std::snprintf(what, sizeof what, "attention past fp16, case %d", which);
        expect_close(what, got, aref, amag, tol);
    }
}

static void test_eltwise() {
    const int n = 20011;
    auto x = rv(n, -30.f, 30.f);
    auto run = [&](void (*f)(float*, int), double (*r)(double), double tol, const char* name) {
        std::vector<float> y(x);
        f(y.data(), n);
        for (float v : y) CHECK(std::isfinite(v), "%s: nonfinite output", name);
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            const double e = std::fabs(y[i] - r(x[i]))/std::max(std::fabs(r(x[i])), 1.0);
            if (e > worst) worst = e;
        }
        CHECK(worst <= tol, "%s: worst error %.3e > %.1e", name, worst, tol);
        float nan = std::numeric_limits<float>::quiet_NaN();
        f(&nan, 1);
        CHECK(std::isnan(nan), "%s(NaN) is not NaN", name);
    };
    run(silu, [](double v) { return v/(1.0 + std::exp(-v)); }, 2e-6, "silu");
    run(gelu_tanh, [](double v) { return 0.5*v*(1.0 + std::tanh(0.7978845608028654*(v + 0.044715*v*v*v))); },
        2e-6, "gelu_tanh");
    run(gelu_erf, [](double v) { return 0.5*v*(1.0 + std::erf(v/std::sqrt(2.0))); }, 2e-6, "gelu_erf");
    run(mish, [](double v) { return v > 20.0 ? v : v*std::tanh(std::log1p(std::exp(v))); }, 2e-6, "mish");
}

static void test_norms() {
    const int shapes[][2] = {{1, 768}, {50, 960}, {7, 6}};
    for (auto& s : shapes) {
        const int seq = s[0], H = s[1];
        auto x = rv((size_t)seq*H, -3.f, 5.f), w = rv(H), b = rv(H);
        std::vector<float> y((size_t)seq*H), z((size_t)seq*H);
        rmsnorm(y.data(), x.data(), w.data(), seq, H, 1e-6f);
        layernorm(z.data(), x.data(), w.data(), b.data(), seq, H, 1e-5f);
        for (float v : y) CHECK(std::isfinite(v), "rmsnorm: nonfinite output");
        for (float v : z) CHECK(std::isfinite(v), "layernorm: nonfinite output");
        double wr = 0.0, wl = 0.0;
        for (int t = 0; t < seq; t++) {
            double ss = 0.0, mu = 0.0, var = 0.0;
            for (int i = 0; i < H; i++) { const double v = x[(size_t)t*H+i]; ss += v*v; mu += v; }
            mu /= H;
            for (int i = 0; i < H; i++) { const double d = x[(size_t)t*H+i] - mu; var += d*d; }
            const double r = 1.0/std::sqrt(ss/H + 1e-6), l = 1.0/std::sqrt(var/H + 1e-5);
            for (int i = 0; i < H; i++) {
                const size_t k = (size_t)t*H+i;
                wr = std::max(wr, std::fabs(y[k] - x[k]*r*w[i]));
                wl = std::max(wl, std::fabs(z[k] - ((x[k] - mu)*l*w[i] + b[i])));
            }
        }
        CHECK(wr <= 1e-5 && wl <= 1e-5, "norms (%d,%d): rms %.3e ln %.3e", seq, H, wr, wl);
    }
}

static void test_conv() {
    const int cfgs[][7] = {{30, 30, 3, 64, 7, 2, 3}, {28, 28, 64, 64, 3, 1, 1}, {15, 15, 64, 128, 3, 2, 1}};
    for (auto& c : cfgs) {
        const int H = c[0], Wd = c[1], Cin = c[2], Cout = c[3], k = c[4], st = c[5], pd = c[6];
        const int Ho = (H + 2*pd - k)/st + 1, Wo = (Wd + 2*pd - k)/st + 1, K = k*k*Cin;
        auto x = rv((size_t)H*Wd*Cin), W = rv((size_t)Cout*K, -.1f, .1f), b = rv(Cout);
        std::vector<float> y((size_t)Ho*Wo*Cout), yp(y.size()), Wp((size_t)Cout*K);
        conv2d(y.data(), x.data(), W.data(), b.data(), H, Wd, Cin, Cout, k, st, pd);
        pack_weights16(W.data(), Wp.data(), Cout, K);
        conv2d_packed(yp.data(), x.data(), Wp.data(), b.data(), H, Wd, Cin, Cout, k, st, pd);
        for (float v : y) CHECK(std::isfinite(v), "conv2d: nonfinite output");
        for (float v : yp) CHECK(std::isfinite(v), "conv2d_packed: nonfinite output");
        double worst = 0.0;
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++)
                for (int co = 0; co < Cout; co++) {
                    double s = b[co], m = std::fabs(b[co]);
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++) {
                            const int iy = oy*st - pd + ky, ix = ox*st - pd + kx;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= Wd) continue;
                            for (int ci = 0; ci < Cin; ci++) {
                                const double p = (double)x[((size_t)iy*Wd+ix)*Cin+ci]*
                                                 W[(size_t)co*K + ((size_t)ky*k+kx)*Cin + ci];
                                s += p; m += std::fabs(p);
                            }
                        }
                    const size_t o = ((size_t)oy*Wo+ox)*Cout+co;
                    worst = std::max(worst, std::max(std::fabs(y[o] - s), std::fabs(yp[o] - s))/(m + 1e-30));
                }
        CHECK(worst <= 1e-5, "conv2d (%d,%d,%d,%d): scaled error %.3e", H, Cin, Cout, k, worst);
    }

    const int T = 8, Cin = 64, Cout = 32, k = 4, st = 2, pd = 1, To = (T - 1)*st - 2*pd + k;
    auto x = rv((size_t)T*Cin), W = rv((size_t)Cin*k*Cout, -.1f, .1f), b = rv(Cout);
    std::vector<float> y((size_t)To*Cout);
    conv_transpose1d(y.data(), x.data(), W.data(), b.data(), T, Cin, Cout, k, st, pd);
    for (float v : y) CHECK(std::isfinite(v), "conv_transpose1d: nonfinite output");
    double worst = 0.0;
    for (int o = 0; o < To; o++)
        for (int co = 0; co < Cout; co++) {
            double s = b[co], m = std::fabs(b[co]);
            for (int t = 0; t < T; t++) {
                const int kt = o + pd - t*st;
                if (kt < 0 || kt >= k) continue;
                for (int ci = 0; ci < Cin; ci++) {
                    const double p = (double)x[(size_t)t*Cin+ci]*W[((size_t)ci*k+kt)*Cout+co];
                    s += p; m += std::fabs(p);
                }
            }
            worst = std::max(worst, std::fabs(y[(size_t)o*Cout+co] - s)/(m + 1e-30));
        }
    CHECK(worst <= 1e-5, "conv_transpose1d: scaled error %.3e", worst);

    const int P = 64, C = 256, G = 8;
    auto gx = rv((size_t)P*C, -2.f, 3.f), gs = rv(C), gb = rv(C);
    std::vector<float> g1((size_t)P*C), g2((size_t)P*C);
    groupnorm(g1.data(), gx.data(), gs.data(), gb.data(), P, C, G, 1e-5f, false);
    groupnorm(g2.data(), gx.data(), gs.data(), gb.data(), P, C, G, 1e-5f, false);
    for (float v : g1) CHECK(std::isfinite(v), "groupnorm: nonfinite output");
    CHECK(same_bits(g1, g2), "groupnorm is not deterministic");
    double gw = 0.0;
    const int gc = C/G;
    for (int g = 0; g < G; g++) {
        double mu = 0.0, var = 0.0;
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++) mu += gx[(size_t)p*C+c];
        mu /= (double)P*gc;
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++) {
            const double d = gx[(size_t)p*C+c] - mu; var += d*d;
        }
        const double inv = 1.0/std::sqrt(var/((double)P*gc) + 1e-5);
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++)
            gw = std::max(gw, std::fabs(g1[(size_t)p*C+c] - ((gx[(size_t)p*C+c] - mu)*inv*gs[c] + gb[c])));
    }
    CHECK(gw <= 1e-5, "groupnorm error %.3e", gw);
}

// IME2 fp16 tile tensors: every op against its fp32 reference on the same
// (fp16-representable) inputs. Runs only where the tile path is enabled.
static void test_f16_tiles() {
    if (!ime_f16_enabled()) return;
    auto round16 = [](std::vector<float> x, int rows, int K) {
        std::vector<uint16_t> t(ime_f16_size(rows, K));
        ime_f16_pack(t.data(), x.data(), rows, K);
        CHECK(ime_f16_unpack(x.data(), t.data(), rows, K), "ime_f16_unpack reported non-finite values");
        return x;
    };
    for (int rows : {1, 31, 50, 256, 261}) {
        const int K = 96, N = 160;
        const auto x0 = rv((size_t)rows*K, -2.f, 2.f);
        const auto x = round16(x0, rows, K);
        for (size_t i = 0; i < x.size(); i++)
            if (std::fabs(x[i] - x0[i]) > 5e-4f*std::fabs(x0[i]) + 1e-7f) {
                CHECK(false, "ime_f16 pack/unpack moved %g to %g (rows %d)", x0[i], x[i], rows);
                break;
            }
        std::vector<uint16_t> xt(ime_f16_size(rows, K)), yt(ime_f16_size(rows, N)), zt(xt.size());
        ime_f16_pack(xt.data(), x.data(), rows, K);
        std::vector<double> one((size_t)rows*K, 1.0);

        // Linear, two jobs over one input.
        const auto W = rv((size_t)N*K, -0.2f, 0.2f), W2 = rv((size_t)K*K, -0.2f, 0.2f), b = rv(N), b2 = rv(K);
        std::vector<uint16_t> Wh(ime_packed_size(N, K)), Wh2(ime_packed_size(K, K));
        CHECK(pack_weights_ime(W.data(), Wh.data(), N, K) && pack_weights_ime(W2.data(), Wh2.data(), K, K),
              "pack_weights_ime refused fp16-range weights");
        std::vector<float> bt((size_t)N*8), bt2((size_t)K*8), y((size_t)rows*N), z((size_t)rows*K);
        ime_f16_bias_tiles(bt.data(), b.data(), N);
        ime_f16_bias_tiles(bt2.data(), b2.data(), K);
        const ImeF16Job jobs[2] = {{yt.data(), Wh.data(), bt.data(), N}, {zt.data(), Wh2.data(), bt2.data(), K}};
        ime_f16_linear(jobs, 2, xt.data(), rows, K);
        CHECK(ime_f16_unpack(y.data(), yt.data(), rows, N) && ime_f16_unpack(z.data(), zt.data(), rows, K),
              "ime_f16_linear produced non-finite values");
        std::vector<double> ref, mag;
        naive_linear(ref, mag, x.data(), W.data(), b.data(), rows, N, K);
        expect_close("ime_f16_linear", y, ref, mag, 2e-3);
        naive_linear(ref, mag, x.data(), W2.data(), b2.data(), rows, K, K);
        expect_close("ime_f16_linear job 2", z, ref, mag, 2e-3);

        // LayerNorm.
        const auto g = rv(K, 0.5f, 1.5f), be = rv(K);
        std::vector<float> gt((size_t)K*8), bet((size_t)K*8), lnref((size_t)rows*K), got((size_t)rows*K);
        ime_f16_bias_tiles(gt.data(), g.data(), K);
        ime_f16_bias_tiles(bet.data(), be.data(), K);
        ime_f16_layernorm(zt.data(), xt.data(), gt.data(), bet.data(), rows, K, 1e-6f);
        CHECK(ime_f16_unpack(got.data(), zt.data(), rows, K), "ime_f16_layernorm produced non-finite values");
        layernorm(lnref.data(), x.data(), g.data(), be.data(), rows, K, 1e-6f);
        expect_close("ime_f16_layernorm", got, std::vector<double>(lnref.begin(), lnref.end()), one, 3e-3);

        // GELU and the residual add.
        auto gref = x;
        gelu_tanh(gref.data(), rows*K);
        std::vector<uint16_t> gtile = xt;
        ime_f16_gelu_tanh(gtile.data(), rows, K);
        CHECK(ime_f16_unpack(got.data(), gtile.data(), rows, K), "ime_f16_gelu_tanh produced non-finite values");
        expect_close("ime_f16_gelu_tanh", got, std::vector<double>(gref.begin(), gref.end()), one, 2e-3);
        ime_f16_add(gtile.data(), xt.data(), rows, K);
        CHECK(ime_f16_unpack(got.data(), gtile.data(), rows, K), "ime_f16_add produced non-finite values");
        for (size_t i = 0; i < gref.size(); i++) gref[i] += x[i];
        expect_close("ime_f16_add", got, std::vector<double>(gref.begin(), gref.end()), one, 4e-3);

        gref = x;
        gelu_erf(gref.data(), rows*K);
        gtile = xt;
        ime_f16_gelu_erf(gtile.data(), rows, K);
        CHECK(ime_f16_unpack(got.data(), gtile.data(), rows, K), "ime_f16_gelu_erf produced non-finite values");
        expect_close("ime_f16_gelu_erf", got, std::vector<double>(gref.begin(), gref.end()), one, 2e-3);

        // Rotary embedding on rows [first, rows), 3 heads of 32.
        {
            const int hd = 32, half = hd/2, first = rows > 5 ? 5 : 0, nr = rows - first;
            std::vector<float> ang = rv((size_t)nr*hd, -3.f, 3.f), cs(ang.size()), sn(ang.size());
            for (size_t i = 0; i < ang.size(); i++) { cs[i] = std::cos(ang[i]); sn[i] = std::sin(ang[i]); }
            std::vector<float> ct(ime_f16_rope_size(rows, hd)), st(ct.size());
            ime_f16_rope_tables(ct.data(), st.data(), cs.data(), sn.data(), rows, first, hd);
            gtile = xt;
            ime_f16_rope(gtile.data(), ct.data(), st.data(), rows, 3, hd);
            CHECK(ime_f16_unpack(got.data(), gtile.data(), rows, K), "ime_f16_rope produced non-finite values");
            std::vector<double> rref(x.begin(), x.end());
            for (int r = first; r < rows; r++)
                for (int h = 0; h < 3; h++)
                    for (int d = 0; d < half; d++) {
                        const size_t o = (size_t)r*K + h*hd + d, t = (size_t)(r - first)*hd + d;
                        const double xa = x[o], xb = x[o + half];
                        rref[o] = xa*cs[t] - xb*sn[t];
                        rref[o + half] = xb*cs[t + half] + xa*sn[t + half];
                    }
            expect_close("ime_f16_rope", got, rref, one, 2e-3);
        }

        // Self-attention: 3 heads of 32 over the K = 96 columns.
        const int heads = 3, hd = 32;
        const auto kx = round16(rv((size_t)rows*K), rows, K), vx = round16(rv((size_t)rows*K), rows, K);
        std::vector<uint16_t> kt(xt.size()), vt(xt.size());
        ime_f16_pack(kt.data(), kx.data(), rows, K);
        ime_f16_pack(vt.data(), vx.data(), rows, K);
        const float scale = 1.0f/std::sqrt((float)hd);
        CHECK(ime_f16_attention(zt.data(), xt.data(), kt.data(), vt.data(), rows, heads, hd, scale),
              "ime_f16_attention refused rows %d", rows);
        CHECK(ime_f16_unpack(got.data(), zt.data(), rows, K), "ime_f16_attention produced non-finite values");
        naive_attention(ref, x.data(), kx.data(), vx.data(), nullptr, rows, rows, heads, heads, hd, scale);
        expect_close("ime_f16_attention", got, ref, one, 3e-3);
    }
    {   // A value past fp16's range must be reported, not returned.
        std::vector<float> x(64*32, 1.0f);
        x[100] = 1e6f;
        std::vector<uint16_t> t(ime_f16_size(64, 32));
        ime_f16_pack(t.data(), x.data(), 64, 32);
        CHECK(!ime_f16_unpack(x.data(), t.data(), 64, 32), "ime_f16_unpack accepted an overflowed value");
    }
    {   // A whole encoder layer: tile path against the fp32-tensor path.
        const int d = 64, mlp = 128, total = 70;
        const auto wq = rv(d*d, -0.2f, 0.2f), wk = rv(d*d, -0.2f, 0.2f), wv = rv(d*d, -0.2f, 0.2f),
                   wo = rv(d*d, -0.2f, 0.2f), w1 = rv(mlp*d, -0.2f, 0.2f), w2 = rv(d*mlp, -0.2f, 0.2f);
        const auto bq = rv(d), bk = rv(d), bv = rv(d), bo = rv(d), b1 = rv(mlp), b2 = rv(d);
        const auto g1 = rv(d, 0.5f, 1.5f), e1 = rv(d), g2 = rv(d, 0.5f, 1.5f), e2 = rv(d);
        nn::EncoderLayer layer;
        using Role = nn::Linear::Role;
        layer.attn.wq.init(wq.data(), bq.data(), d, d, Role::Gemm);
        layer.attn.wk.init(wk.data(), bk.data(), d, d, Role::Gemm);
        layer.attn.wv.init(wv.data(), bv.data(), d, d, Role::Gemm);
        layer.attn.wo.init(wo.data(), bo.data(), d, d, Role::Gemm);
        layer.attn.set_shape(4, 16);
        layer.w1.init(w1.data(), b1.data(), mlp, d, Role::Mlp);
        layer.w2.init(w2.data(), b2.data(), d, mlp, Role::Mlp);
        layer.ln1_s = g1.data(); layer.ln1_b = e1.data(); layer.ln2_s = g2.data(); layer.ln2_b = e2.data();
        layer.d = d; layer.mlp = mlp;
        CHECK(layer.prepare_f16(), "EncoderLayer::prepare_f16 refused an IME layer");
        auto x = rv((size_t)total*d);
        std::vector<uint16_t> xt(ime_f16_size(total, d));
        ime_f16_pack(xt.data(), x.data(), total, d);
        nn::Scratch sc;
        layer.forward_f16(xt.data(), total, sc, nullptr);
        std::vector<float> got((size_t)total*d);
        CHECK(ime_f16_unpack(got.data(), xt.data(), total, d), "forward_f16 produced non-finite values");
        layer.forward(x.data(), total, nullptr, sc, nullptr);
        std::vector<double> one(x.size(), 1.0);
        expect_close("EncoderLayer::forward_f16", got, std::vector<double>(x.begin(), x.end()), one, 2e-2);
    }
}

int main() {
    CHECK(std::isinf(scaled_error({NAN}, {0}, {1})), "comparison accepts NaN");
    CHECK(std::isinf(scaled_error({INFINITY}, {0}, {1})), "comparison accepts infinity");
    CHECK(std::isinf(scaled_error({0}, {}, {1})), "comparison accepts shape mismatch");
    test_linear();
    test_attention();
    test_fp16_range();
    test_f16_tiles();
    test_eltwise();
    test_norms();
    test_conv();
    if (failures) std::fprintf(stderr, "%d check(s) failed on %s\n", failures, hal::backend_name());
    else std::printf("ok (%s)\n", hal::backend_name());
    return failures ? 1 : 0;
}
