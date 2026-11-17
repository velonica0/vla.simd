/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// SpacemiT IME2 fp16 GEMM and attention for the K3's A100 cores (VLEN 1024).
//
// vfwmadot vd, vs1, vs2, fp16 (vtype e16/m1): vd, an 8x8 fp32 tile held row-major
// in a register pair, accumulates vs1 * vs2^T, where vs1 is an 8x8 fp16 tile of
// activations (8 rows x 8 k) and vs2 an 8x8 fp16 tile of weights (8 output
// columns x 8 k): 512 multiply-adds per instruction. Only the A100 cores have
// it; the X100 cores raise SIGILL. This is the only file built with
// _xsmtvdotii, so nothing else can emit the instructions.
//
// Weights are packed once at load into fp16 tiles [N/16][K8][2][8][8] (K
// zero-padded to a multiple of 8); activations are rounded to fp16 and packed
// per call into [rows/(8*MT)][K8][MT][8][8]. Accumulation is fp32. Rounding to
// fp16 (11-bit significand, range 65504) makes this a lossy path, so it is
// opt-in (TCPU_IME=1), like the bf16 panels, and it needs the process started
// on the A100 cores (see detect()). Values past the fp16 range never become
// inf: a weight matrix that does not fit is refused at pack time (the layer
// keeps its fp32/bf16 path), an activation row that does not fit is packed
// scaled by a power of two and rescaled in fp32, and attention whose Q, K or V
// does not fit falls back to the RVV kernel. ime_attention, below, runs attention on the
// same micro-kernel (TCPU_IME_ATTN=0 keeps attention on the RVV kernel).

#include "../arch.h"
#if TCPU_HAL_RVV && TCPU_RVV_IME

#include "../common/env.h"
#include "../simd.h"
#include "../../ops/lm_ops.h"
#include <riscv_vector.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
using std::size_t;

namespace tcpu {

namespace {

#ifndef TCPU_IME_MT
#define TCPU_IME_MT 4   // activation tiles per micro-kernel call (4 or 6)
#endif
constexpr int MT = TCPU_IME_MT;
static_assert(MT == 4 || MT == 6, "the IME micro-kernel exists for 4 or 6 activation tiles");

// MT activation tiles (8*MT rows) x 2 weight tiles (16 columns) over k8 steps.
// a: [k8][MT][64] fp16, b: [k8][2][64] fp16. Output tile (t, h), 64 fp32, is
// stored at c + (h ? c1off : 0) + t*cstep floats: ime_tile's default [MT][2][64]
// for the GEMM, or [2][MT][64] (c1off = MT*64, cstep = 64), the layout the next
// GEMM reads as activations, for attention scores.
void ime_tile(float* c, const _Float16* a, const _Float16* b, size_t k8,
              size_t c1off = 64, size_t cstep = 128) {
    float* c1 = c + c1off;
    const size_t st = cstep*sizeof(float);
    const size_t vl = 64;
    if constexpr (MT == 4) {
        // Two k8 steps per iteration: the 8 activation tiles come from one
        // e16/m8 load (v0-v7) and the 4 weight tiles from one e16/m4 load
        // (v8-v11). The loop is bound by its loads, not by vfwmadot (0.7 ns
        // each with register operands), and six 128-byte loads per step cost
        // about twice what these two wide loads per two steps do.
        const size_t vl2 = 128, vl4 = 256, vl8 = 512, odd = k8 & 1;
        size_t n2 = k8/2;
        __asm__ volatile(
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vmv.v.i  v16, 0 \n\t" "vmv.v.i  v18, 0 \n\t" "vmv.v.i  v20, 0 \n\t" "vmv.v.i  v22, 0 \n\t"
            "vmv.v.i  v24, 0 \n\t" "vmv.v.i  v26, 0 \n\t" "vmv.v.i  v28, 0 \n\t" "vmv.v.i  v30, 0 \n\t"
            "beqz %[n2], 2f \n\t"
            "1: \n\t"
            "vsetvli  zero, %[vl8], e16, m8, ta, ma \n\t"
            "vle16.v  v0, (%[a]) \n\t" "addi %[a], %[a], 1024 \n\t"
            "vsetvli  zero, %[vl4], e16, m4, ta, ma \n\t"
            "vle16.v  v8, (%[b]) \n\t" "addi %[b], %[b], 512 \n\t"
            "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
            "vfwmadot v16, v0, v8, fp16 \n\t" "vfwmadot v18, v0, v9, fp16 \n\t"
            "vfwmadot v20, v1, v8, fp16 \n\t" "vfwmadot v22, v1, v9, fp16 \n\t"
            "vfwmadot v24, v2, v8, fp16 \n\t" "vfwmadot v26, v2, v9, fp16 \n\t"
            "vfwmadot v28, v3, v8, fp16 \n\t" "vfwmadot v30, v3, v9, fp16 \n\t"
            "vfwmadot v16, v4, v10, fp16 \n\t" "vfwmadot v18, v4, v11, fp16 \n\t"
            "vfwmadot v20, v5, v10, fp16 \n\t" "vfwmadot v22, v5, v11, fp16 \n\t"
            "vfwmadot v24, v6, v10, fp16 \n\t" "vfwmadot v26, v6, v11, fp16 \n\t"
            "vfwmadot v28, v7, v10, fp16 \n\t" "vfwmadot v30, v7, v11, fp16 \n\t"
            "addi %[n2], %[n2], -1 \n\t"
            "bnez %[n2], 1b \n\t"
            "2: \n\t"
            "beqz %[odd], 3f \n\t"
            "vsetvli  zero, %[vl4], e16, m4, ta, ma \n\t"
            "vle16.v  v0, (%[a]) \n\t"
            "vsetvli  zero, %[vl2], e16, m2, ta, ma \n\t"
            "vle16.v  v8, (%[b]) \n\t"
            "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
            "vfwmadot v16, v0, v8, fp16 \n\t" "vfwmadot v18, v0, v9, fp16 \n\t"
            "vfwmadot v20, v1, v8, fp16 \n\t" "vfwmadot v22, v1, v9, fp16 \n\t"
            "vfwmadot v24, v2, v8, fp16 \n\t" "vfwmadot v26, v2, v9, fp16 \n\t"
            "vfwmadot v28, v3, v8, fp16 \n\t" "vfwmadot v30, v3, v9, fp16 \n\t"
            "3: \n\t"
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vse32.v  v16, (%[c]) \n\t" "vse32.v  v18, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v20, (%[c]) \n\t" "vse32.v  v22, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v24, (%[c]) \n\t" "vse32.v  v26, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v28, (%[c]) \n\t" "vse32.v  v30, (%[c1]) \n\t"
            : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [c1] "+r"(c1), [n2] "+r"(n2)
            : [vl] "r"(vl), [vl2] "r"(vl2), [vl4] "r"(vl4), [vl8] "r"(vl8), [odd] "r"(odd), [st] "r"(st)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v16", "v17", "v18", "v19", "v20",
              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    } else {
        __asm__ volatile(
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vmv.v.i  v8, 0 \n\t"  "vmv.v.i  v10, 0 \n\t" "vmv.v.i  v12, 0 \n\t" "vmv.v.i  v14, 0 \n\t"
            "vmv.v.i  v16, 0 \n\t" "vmv.v.i  v18, 0 \n\t" "vmv.v.i  v20, 0 \n\t" "vmv.v.i  v22, 0 \n\t"
            "vmv.v.i  v24, 0 \n\t" "vmv.v.i  v26, 0 \n\t" "vmv.v.i  v28, 0 \n\t" "vmv.v.i  v30, 0 \n\t"
            "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
            "1: \n\t"
            "vle16.v  v6, (%[b]) \n\t" "addi %[b], %[b], 128 \n\t"
            "vle16.v  v7, (%[b]) \n\t" "addi %[b], %[b], 128 \n\t"
            "vle16.v  v0, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vle16.v  v1, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v8,  v0, v6, fp16 \n\t"
            "vfwmadot v10, v0, v7, fp16 \n\t"
            "vle16.v  v2, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v12, v1, v6, fp16 \n\t"
            "vfwmadot v14, v1, v7, fp16 \n\t"
            "vle16.v  v3, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v16, v2, v6, fp16 \n\t"
            "vfwmadot v18, v2, v7, fp16 \n\t"
            "vle16.v  v4, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v20, v3, v6, fp16 \n\t"
            "vfwmadot v22, v3, v7, fp16 \n\t"
            "vle16.v  v5, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v24, v4, v6, fp16 \n\t"
            "vfwmadot v26, v4, v7, fp16 \n\t"
            "vfwmadot v28, v5, v6, fp16 \n\t"
            "vfwmadot v30, v5, v7, fp16 \n\t"
            "addi %[n], %[n], -1 \n\t"
            "bnez %[n], 1b \n\t"
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vse32.v  v8, (%[c]) \n\t" "vse32.v  v10, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v12, (%[c]) \n\t" "vse32.v  v14, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v16, (%[c]) \n\t" "vse32.v  v18, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v20, (%[c]) \n\t" "vse32.v  v22, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v24, (%[c]) \n\t" "vse32.v  v26, (%[c1]) \n\t"
            "add %[c], %[c], %[st] \n\t" "add %[c1], %[c1], %[st] \n\t"
            "vse32.v  v28, (%[c]) \n\t" "vse32.v  v30, (%[c1]) \n\t"
            : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [c1] "+r"(c1), [n] "+r"(k8)
            : [vl] "r"(vl), [st] "r"(st)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
}

// One vfwmadot on zero data: the availability probe (SIGILL on a core without IME2).
void ime_probe_insn() {
    alignas(64) static _Float16 z[64];
    alignas(64) float c[64];
    const size_t vl = 64;
    __asm__ volatile(
        "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
        "vmv.v.i  v16, 0 \n\t"
        "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
        "vle16.v  v0, (%[z]) \n\t"
        "vfwmadot v16, v0, v0, fp16 \n\t"
        "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
        "vse32.v  v16, (%[c]) \n\t"
        :
        : [z] "r"(z), [c] "r"(c), [vl] "r"(vl)
        : "memory", "v0", "v16", "v17");
}

bool on_vlen1024() { return __riscv_vlenb()*8 == 1024; }

sigjmp_buf probe_jmp;
void on_sigill(int) { siglongjmp(probe_jmp, 1); }

bool detect() {
    if (!hal::env::ime()) return false;
    // Every thread that can run a GEMM must already be on an A100 core with
    // VLEN 1024. The process cannot move itself there reliably: a write of "0"
    // to /proc/set_ai_thread migrates the calling thread at once, but a thread
    // of an already multithreaded process (Python has NumPy's pool before this
    // library loads) keeps VLEN 256 on the A100 core. A process started from a
    // shell that ran `echo $$ > /proc/set_ai_thread` gets VLEN 1024 everywhere.
    bool all_on = on_vlen1024();
#if defined(_OPENMP)
    #pragma omp parallel reduction(&& : all_on)
    all_on = all_on && on_vlen1024();
#endif
    if (!all_on) {
        std::fprintf(stderr, "vla_simd: TCPU_IME=1 needs the process on the K3's A100 cores "
                             "(start it from a shell after `echo $$ > /proc/set_ai_thread`); IME off\n");
        return false;
    }
    struct sigaction sa = {}, old = {};
    sa.sa_handler = on_sigill;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, &old);
    bool ok = false;
    if (sigsetjmp(probe_jmp, 1) == 0) {
        ime_probe_insn();
        ok = true;
    }
    sigaction(SIGILL, &old, nullptr);
    std::fprintf(stderr, "vla_simd: IME2 fp16 GEMM %s (A100 cores, VLEN 1024)\n",
                 ok ? "on" : "unavailable (vfwmadot raised SIGILL), IME off");
    return ok;
}

inline _Float16 bf16_to_f16(uint16_t h) { return (_Float16)bf16_f32(h); }

constexpr float F16_MAX = 65504.0f;

// max |x[i]| with full-length vector ops (NaN is ignored, as vfmax does).
float abs_max(const float* x, size_t n) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        acc = __riscv_vfmax_vv_f32m4_tu(acc, acc, __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(x + i, vl), vl), vl);
        i += vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax));
}

// True when every |x[i]| <= F16_MAX (NaN passes: it stays NaN either way).
bool fits_f16(size_t n, const float* x) {
    for (size_t i=0; i<n; i++) if (std::fabs(x[i]) > F16_MAX) return false;
    return true;
}

template <class Get>
void pack_w(uint16_t* Wh, int N, int K, Get get) {
    const int K8 = (K+7)/8;
    _Float16* d = reinterpret_cast<_Float16*>(Wh);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int nb=0; nb<N/16; nb++)
        for (int k8=0; k8<K8; k8++)
            for (int h=0; h<2; h++)
                for (int r=0; r<8; r++)
                    for (int c=0; c<8; c++) {
                        const int n = nb*16 + h*8 + r, k = k8*8 + c;
                        d[(((size_t)nb*K8 + k8)*2 + h)*64 + r*8 + c] =
                            k < K ? get((size_t)n*K + k) : (_Float16)0.0f;
                    }
}

} // namespace

bool ime_available() {
    static const bool v = detect();
    return v;
}

size_t ime_packed_size(int N, int K) { return (size_t)(N/16)*((K+7)/8)*128; }

bool pack_weights_ime(const float* W, uint16_t* Wh, int N, int K) {
    if (!fits_f16((size_t)N*K, W)) return false;
    pack_w(Wh, N, K, [W](size_t i) { return (_Float16)W[i]; });
    return true;
}

bool pack_weights_ime_bf16(const uint16_t* Wb16, uint16_t* Wh, int N, int K) {
    for (size_t i=0, n=(size_t)N*K; i<n; i++)
        if (std::fabs(bf16_f32(Wb16[i])) > F16_MAX) return false;
    pack_w(Wh, N, K, [Wb16](size_t i) { return bf16_to_f16(Wb16[i]); });
    return true;
}

void dense_linear_ime(float* out, const float* x, const uint16_t* Wh, const float* bias,
                      int seq, int N, int K) {
    const int K8 = (K+7)/8, MB = (seq + 8*MT - 1)/(8*MT), NB = N/16;
    const _Float16* wp = reinterpret_cast<const _Float16*>(Wh);

    // Activations -> fp16 tiles [MB][K8][MT][8][8], zero rows/columns as padding.
    // thread_local: the view fan-out runs several threads through one Linear.
    // A row whose largest |x| exceeds the fp16 range is packed as x * 2^-e and
    // its output rescaled by rs[row] = 2^e in fp32 (exact powers of two; other
    // rows keep rs = 1, so their results are unchanged).
    static thread_local std::vector<_Float16> xp_buf;
    static thread_local std::vector<float> rs_buf;
    const size_t nx = (size_t)MB*K8*MT*64;
    if (xp_buf.size() < nx) xp_buf.resize(nx);
    if (rs_buf.size() < (size_t)seq) rs_buf.resize(seq);
    _Float16* xp = xp_buf.data();
    float* rs = rs_buf.data();
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int mb=0; mb<MB; mb++)
        for (int t=0; t<MT; t++)
            for (int r=0; r<8; r++) {
                const int row = mb*8*MT + t*8 + r;
                _Float16* d = xp + ((size_t)mb*K8*MT + t)*64 + r*8;
                if (row >= seq) {
                    for (int k8=0; k8<K8; k8++)
                        std::memset(d + (size_t)k8*MT*64, 0, 16);
                    continue;
                }
                const float* xr = x + (size_t)row*K;
                const float amax = abs_max(xr, (size_t)K);
                float s = 1.0f;
                rs[row] = 1.0f;
                if (amax > F16_MAX && amax <= std::numeric_limits<float>::max()) {
                    int e;
                    std::frexp(amax, &e);            // amax < 2^e
                    s = std::ldexp(1.0f, 15 - e);    // amax*s < 2^15
                    rs[row] = std::ldexp(1.0f, e - 15);
                }
                for (int k8=0; k8<K8; k8++) {
                    _Float16* dk = d + (size_t)k8*MT*64;
                    const int k0 = k8*8;
                    if (k0 + 8 <= K) {
                        const vfloat32m1_t v = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(xr + k0, 8), s, 8);
                        __riscv_vse16_v_f16mf2(dk, __riscv_vfncvt_f_f_w_f16mf2(v, 8), 8);
                    } else {
                        for (int c=0; c<8; c++)
                            dk[c] = k0 + c < K ? (_Float16)(xr[k0 + c]*s) : (_Float16)0.0f;
                    }
                }
            }

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    alignas(64) float c[MT*2*64];
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 1) collapse(2)
#endif
    for (int nb=0; nb<NB; nb++) {
        for (int mb=0; mb<MB; mb++) {
            ime_tile(c, xp + (size_t)mb*K8*MT*64, wp + (size_t)nb*K8*128, (size_t)K8);
            const vfloat32m1_t b0 = bias ? __riscv_vle32_v_f32m1(bias + nb*16, 8)
                                         : __riscv_vfmv_v_f_f32m1(0.0f, 8);
            const vfloat32m1_t b1 = bias ? __riscv_vle32_v_f32m1(bias + nb*16 + 8, 8)
                                         : __riscv_vfmv_v_f_f32m1(0.0f, 8);
            for (int t=0; t<MT; t++)
                for (int r=0; r<8; r++) {
                    const int row = mb*8*MT + t*8 + r;
                    if (row >= seq) break;
                    float* o = out + (size_t)row*N + nb*16;
                    const float* ct = c + (t*2)*64 + r*8;
                    const float sr = rs[row];        // c*1 + b == c + b exactly
                    __riscv_vse32_v_f32m1(o, __riscv_vfmadd_vf_f32m1(
                        __riscv_vle32_v_f32m1(ct, 8), sr, b0, 8), 8);
                    __riscv_vse32_v_f32m1(o + 8, __riscv_vfmadd_vf_f32m1(
                        __riscv_vle32_v_f32m1(ct + 64, 8), sr, b1, 8), 8);
                }
        }
    }
  }
}

// ---- attention ----
//
// Per (query head, block of 8*MT queries): S = (scale*Q) K^T and O = P V both run
// on ime_tile, with an online (flash) softmax in fp32 between them over chunks
// of KC keys. Q, K, V and the softmax numerators P (in [0, 1]) are rounded to
// fp16; the row sums and the output accumulation are fp32.
//
// Everything between the two GEMMs stays in the activation layout of the second
// one, [keys/8][MT][8 rows][8 keys]: QK^T stores its tiles there directly, the
// softmax runs on whole 64*MT-float slabs (per-row max and sum are folded out of
// elementwise slab accumulators), and P is converted to fp16 in order. On the
// A100 a dependent chain of 8-lane vector operations costs ~32 cycles per 8
// elements and a scalar fp16 convert ~7 cycles, while full-length FP32 vector
// arithmetic runs at ~13 elements per cycle, so the kernel keeps to full-length
// operations and to chunks small enough for L1 (a 1024-key fp32 score block is
// 128 KB; streaming it through L2 tripled the softmax's cost per element).
// ViT shape (1024 x 1024, 12 heads, 8 threads): 13 ms vs 64 ms for the RVV kernel.
//
// K is packed per kv head as the GEMM's weight operand [SKP/16][hd/8][2][64]
// (keys padded to 16 with zeros), V transposed as [hd/16][SKP/8][2][64], so the
// group's query heads share them. The additive mask, with BLOCK -> -inf and the
// padded keys at -inf, is packed once per query block in the score layout and
// shared by every head. Masking and the all-blocked-row fallback match
// hal/rvv/attn.cpp.

// exp(x) for the softmax numerators: x <= 0 (score minus row max) and the
// result is rounded to fp16 next, so a degree-4 2^f (relative error 6.4e-6
// on [-17.5, 0], 40x under fp16's half-ulp) replaces exp_ps's Cephes
// polynomial, at about half the A100 vector instructions. x is clamped at
// -17.5: exp(-17.5) = 2.5e-8 rounds to an fp16 0, so blocked (-inf) and padded
// keys still get exactly zero weight; their 2.5e-8 in the fp32 sum is
// negligible. c1..c4 are the polynomial's coefficients, broadcast by the caller.
static inline vfloat32m4_t exp_softmax(vfloat32m4_t x, vfloat32m4_t c1, vfloat32m4_t c2,
                                       vfloat32m4_t c3, vfloat32m4_t c4, size_t vl) {
    const vfloat32m4_t t = __riscv_vfmul_vf_f32m4(__riscv_vfmax_vf_f32m4(x, -17.5f, vl),
                                                  1.44269504088896341f, vl);
    const vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4(t, vl);   // round to nearest
    const vfloat32m4_t f = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(n, vl), vl);
    vfloat32m4_t y = __riscv_vfmadd_vv_f32m4(c4, f, c3, vl);
    y = __riscv_vfmadd_vv_f32m4(y, f, c2, vl);
    y = __riscv_vfmadd_vv_f32m4(y, f, c1, vl);
    y = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), y, f, vl);   // 1 + f*y
    const vfloat32m4_t pow2n = __riscv_vreinterpret_v_i32m4_f32m4(
        __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl));
    return __riscv_vfmul_vv_f32m4(y, pow2n, vl);
}

bool ime_attention_enabled() {
    static const bool v = ime_available() && hal::env::ime_attn();
    return v;
}

bool ime_attention(float* out, const float* Q, const float* K, const float* V,
                   int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                   float scale, const float* mask) {
    if (head_dim % 16 != 0 || head_dim > 256 || seq_q <= 0 || seq_k <= 0 || !ime_attention_enabled()) return false;
    const int    group = n_q/n_kv;
    const int    QB    = 8*MT;
    const size_t SLAB  = (size_t)MT*64;                // one 8-key group of a block, fp32
    const size_t SKP   = ((size_t)seq_k + 15)/16*16;   // keys padded to a weight tile
    const size_t K8q   = (size_t)head_dim/8;           // QK reduction steps
    const size_t K8p   = SKP/8;                        // key groups
    const size_t NBD   = (size_t)head_dim/16;          // output tiles per row block
    const size_t kp_n  = SKP/16*K8q*128;               // per kv head
    const size_t vt_n  = NBD*K8p*128;                  // per kv head
    const size_t ldkv  = (size_t)n_kv*head_dim;        // K/V row stride
    const float  NINF  = -std::numeric_limits<float>::infinity();
    const float  BLOCK = std::numeric_limits<float>::lowest();
    const int    qblocks = (seq_q + QB - 1)/QB;
    const bool   use_mp  = mask || SKP != (size_t)seq_k;

    static thread_local std::vector<_Float16> kp_buf, vt_buf;
    static thread_local std::vector<float> mp_buf, vm_buf;
    if (kp_buf.size() < (size_t)n_kv*kp_n) kp_buf.resize((size_t)n_kv*kp_n);
    if (vt_buf.size() < (size_t)n_kv*vt_n) vt_buf.resize((size_t)n_kv*vt_n);
    if (use_mp && mp_buf.size() < (size_t)qblocks*K8p*SLAB) mp_buf.resize((size_t)qblocks*K8p*SLAB);
    _Float16* kp = kp_buf.data();
    _Float16* vt = vt_buf.data();
    float*    mp = use_mp ? mp_buf.data() : nullptr;
    // Mean of V per kv head, for rows whose every key is blocked (only a mask
    // can block a row). SmolVLA's padded language tokens are such rows in every
    // prefill layer, so this is computed once per call, not per row.
    if (mask && vm_buf.size() < ldkv) vm_buf.resize(ldkv);
    float* vm = mask ? vm_buf.data() : nullptr;
    // fp16 range: the largest |K|, |V| and |scale*Q|, gathered while packing.
    // Past 65504 the whole call returns false and the caller's fp32 kernel runs.
    float kv_amax = 0.0f, q_amax = 0.0f;

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    const size_t vlmax4 = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vamax = __riscv_vfmv_v_f_f32m4(0.0f, vlmax4);
    // K and V^T are converted a whole row at a time (one full-length vector
    // convert into a small fp16 staging row) and then moved into the tiles
    // with 8-byte / 2-byte scalar copies: per-element scalar converts and
    // 8-lane vector ops each cost tens of cycles on the A100.
    //
    // K tile (nb, k8) half h, row r = key nb*16+h*8+r, columns d = k8*8 .. +7.
#if defined(_OPENMP)
    #pragma omp for schedule(static) collapse(2)
#endif
    for (int kv=0; kv<n_kv; kv++)
        for (size_t nb=0; nb<SKP/16; nb++) {
            alignas(16) _Float16 row[256];
            _Float16* tile = kp + kv*kp_n + nb*K8q*128;
            for (int r=0; r<16; r++) {
                const size_t key = nb*16 + r;
                if (key < (size_t)seq_k) {
                    const float* src = K + key*ldkv + (size_t)kv*head_dim;
                    for (size_t i = 0; i < (size_t)head_dim;) {
                        const size_t vl = __riscv_vsetvl_e32m4(head_dim - i);
                        const vfloat32m4_t v = __riscv_vle32_v_f32m4(src + i, vl);
                        vamax = __riscv_vfmax_vv_f32m4_tu(vamax, vamax, __riscv_vfabs_v_f32m4(v, vl), vl);
                        __riscv_vse16_v_f16m2(row + i, __riscv_vfncvt_f_f_w_f16m2(v, vl), vl);
                        i += vl;
                    }
                } else {
                    std::memset(row, 0, head_dim*sizeof(_Float16));
                }
                for (size_t k8=0; k8<K8q; k8++) std::memcpy(tile + k8*128 + r*8, row + k8*8, 16);
            }
        }
    // V^T tile (nb, k8) half h, row r = dim nb*16+h*8+r, columns key = k8*8 .. +7:
    // stage the 8 keys' rows, then transpose with 2-byte moves.
#if defined(_OPENMP)
    #pragma omp for schedule(static) collapse(2)
#endif
    for (int kv=0; kv<n_kv; kv++)
        for (size_t k8=0; k8<K8p; k8++) {
            alignas(16) _Float16 rows[8][256];
            for (int c=0; c<8; c++) {
                const size_t key = k8*8 + c;
                if (key < (size_t)seq_k) {
                    const float* src = V + key*ldkv + (size_t)kv*head_dim;
                    for (size_t i = 0; i < (size_t)head_dim;) {
                        const size_t vl = __riscv_vsetvl_e32m4(head_dim - i);
                        const vfloat32m4_t v = __riscv_vle32_v_f32m4(src + i, vl);
                        vamax = __riscv_vfmax_vv_f32m4_tu(vamax, vamax, __riscv_vfabs_v_f32m4(v, vl), vl);
                        __riscv_vse16_v_f16m2(rows[c] + i, __riscv_vfncvt_f_f_w_f16m2(v, vl), vl);
                        i += vl;
                    }
                } else {
                    std::memset(rows[c], 0, head_dim*sizeof(_Float16));
                }
            }
            for (size_t nb=0; nb<NBD; nb++) {
                _Float16* d = vt + kv*vt_n + (nb*K8p + k8)*128;
                for (int r=0; r<16; r++, d += 8)
                    for (int c=0; c<8; c++) d[c] = rows[c][nb*16 + r];
            }
        }
    if (vm) {   // contiguous row sums, full-length vector ops; one thread
#if defined(_OPENMP)
        #pragma omp single nowait
#endif
        {
            std::memset(vm, 0, ldkv*sizeof(float));
            for (int t=0; t<seq_k; t++)
                for (size_t i = 0; i < ldkv;) {
                    const size_t vl = __riscv_vsetvl_e32m4(ldkv - i);
                    __riscv_vse32_v_f32m4(vm + i, __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(vm + i, vl),
                        __riscv_vle32_v_f32m4(V + (size_t)t*ldkv + i, vl), vl), vl);
                    i += vl;
                }
            for (size_t i=0; i<ldkv; i++) vm[i] /= (float)seq_k;
        }
    }
    // Mask in the score layout, per query block: BLOCK -> -inf, padded keys -inf.
    if (use_mp) {
#if defined(_OPENMP)
        #pragma omp for schedule(static) collapse(2)
#endif
        for (int blk=0; blk<qblocks; blk++)
            for (size_t k8=0; k8<K8p; k8++) {
                float* d = mp + ((size_t)blk*K8p + k8)*SLAB;
                for (int r=0; r<QB; r++) {
                    const int q = blk*QB + r;
                    for (int c=0; c<8; c++) {
                        const size_t key = k8*8 + c;
                        float m = 0.0f;
                        if (key >= (size_t)seq_k) m = NINF;
                        else if (mask && q < seq_q) {
                            m = mask[(size_t)q*seq_k + key];
                            if (m == BLOCK) m = NINF;
                        }
                        d[(size_t)r*8 + c] = m;
                    }
                }
            }
    }
    {
        const size_t ldq = (size_t)n_q*head_dim;
        float qa = 0.0f;
#if defined(_OPENMP)
        #pragma omp for schedule(static) nowait
#endif
        for (int t=0; t<seq_q; t++) qa = std::max(qa, abs_max(Q + (size_t)t*ldq, ldq));
        const float ka = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(
            vamax, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax4));
#if defined(_OPENMP)
        #pragma omp critical(tcpu_ime_amax)
#endif
        { kv_amax = std::max(kv_amax, ka); q_amax = std::max(q_amax, qa); }
    }
#if defined(_OPENMP)
    #pragma omp barrier
#endif
    // Same value in every thread after the barrier, so all skip the loop together.
    const bool fits = kv_amax <= F16_MAX && q_amax*std::fabs(scale) <= F16_MAX;

    // Flash-style: keys in chunks of KC so the chunk's scores, P and the output
    // accumulators stay in L1 (a 1024-key score block is 128 KB; streaming it
    // through L2 made the softmax 3x slower per element on the A100). Running
    // row max mrow and sum lrow; the output accumulator is rescaled by
    // alpha = exp(m_old - m_new) per chunk.
    constexpr size_t KC = 128;                      // keys per chunk (multiple of 16)
    const size_t KC8 = KC/8;
    std::vector<_Float16> qp(K8q*SLAB), pp(KC8*SLAB);
    std::vector<float> sp(KC8*SLAB), oacc(NBD*2*SLAB);
    alignas(64) float acc[64*MT], rowv[64*MT], alphav[2*64*MT], c[MT*2*64];
    float mrow[8*MT], lrow[8*MT], alpha[8*MT];
    if (fits) {
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 1) collapse(2)
#endif
    for (int h=0; h<n_q; h++)
        for (int blk=0; blk<qblocks; blk++) {
            const int kv  = h/group;
            const int b0  = blk*QB;
            const int nbq = seq_q - b0 < QB ? seq_q - b0 : QB;

            // scale*Q -> [K8q][MT][64], written in order; rows past seq_q are zero.
            {
                _Float16* d = qp.data();
                for (size_t k8=0; k8<K8q; k8++)
                    for (int r=0; r<QB; r++, d += 8) {
                        if (r >= nbq) { std::memset(d, 0, 16); continue; }
                        const float* src = Q + ((size_t)(b0 + r)*n_q + h)*head_dim + k8*8;
                        for (int cc=0; cc<8; cc++) d[cc] = (_Float16)(src[cc]*scale);
                    }
            }
            for (int r=0; r<QB; r++) { mrow[r] = NINF; lrow[r] = 0.0f; }
            std::memset(oacc.data(), 0, oacc.size()*sizeof(float));

            const _Float16* kph = kp + kv*kp_n;
            const _Float16* vth = vt + kv*vt_n;
            const float* mpb = mp ? mp + (size_t)blk*K8p*SLAB : nullptr;
            for (size_t c0=0; c0<SKP; c0+=KC) {
                const size_t kc  = SKP - c0 < KC ? SKP - c0 : KC;
                const size_t nk8 = kc/8, g0 = c0/8;
                // S chunk = Q K^T, straight into [nk8][MT][64].
                for (size_t nb=0; nb<kc/16; nb++)
                    ime_tile(sp.data() + nb*2*SLAB, qp.data(), kph + (c0/16 + nb)*K8q*128, K8q, SLAB, 64);

                // Mask; slab-wise max -> per-row chunk max -> running max.
                for (size_t i = 0; i < SLAB;) {
                    const size_t vl = __riscv_vsetvl_e32m4(SLAB - i);
                    vfloat32m4_t mx = __riscv_vfmv_v_f_f32m4(NINF, vl);
                    for (size_t k8=0; k8<nk8; k8++) {
                        float* s = sp.data() + k8*SLAB + i;
                        vfloat32m4_t x = __riscv_vle32_v_f32m4(s, vl);
                        if (mpb) {
                            x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(mpb + (g0 + k8)*SLAB + i, vl), vl);
                            __riscv_vse32_v_f32m4(s, x, vl);
                        }
                        mx = __riscv_vfmax_vv_f32m4(mx, x, vl);
                    }
                    __riscv_vse32_v_f32m4(acc + i, mx, vl);
                    i += vl;
                }
                for (int r=0; r<QB; r++) {
                    float m = mrow[r];
                    for (int cc=0; cc<8; cc++) m = acc[r*8 + cc] > m ? acc[r*8 + cc] : m;
                    alpha[r] = mrow[r] == NINF ? 0.0f : std::exp(mrow[r] - m);
                    mrow[r] = m;
                    const float ref = m == NINF ? 0.0f : m;   // all blocked so far: P -> 0
                    for (int cc=0; cc<8; cc++) rowv[r*8 + cc] = ref;
                }

                // P = exp(S - m) -> fp16 [nk8][MT][64]; slab-wise sums -> running sums.
                for (size_t i = 0; i < SLAB;) {
                    const size_t vl = __riscv_vsetvl_e32m4(SLAB - i);
                    const vfloat32m4_t mv = __riscv_vle32_v_f32m4(rowv + i, vl);
                    const vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(0.6931280494f, vl);
                    const vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.2402367741f, vl);
                    const vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.0558702424f, vl);
                    const vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.0095902430f, vl);
                    vfloat32m4_t sum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    for (size_t k8=0; k8<nk8; k8++) {
                        const vfloat32m4_t e = exp_softmax(__riscv_vfsub_vv_f32m4(
                            __riscv_vle32_v_f32m4(sp.data() + k8*SLAB + i, vl), mv, vl), c1, c2, c3, c4, vl);
                        sum = __riscv_vfadd_vv_f32m4(sum, e, vl);
                        __riscv_vse16_v_f16m2(pp.data() + k8*SLAB + i, __riscv_vfncvt_f_f_w_f16m2(e, vl), vl);
                    }
                    __riscv_vse32_v_f32m4(acc + i, sum, vl);
                    i += vl;
                }
                for (int r=0; r<QB; r++) {
                    float z = 0.0f;
                    for (int cc=0; cc<8; cc++) z += acc[r*8 + cc];
                    lrow[r] = lrow[r]*alpha[r] + z;
                }
                // alpha in the [MT][2][64] output layout: element (t, h, r, col) -> row t*8+r.
                for (int t=0; t<MT; t++)
                    for (int hh=0; hh<2; hh++)
                        for (int r=0; r<8; r++)
                            for (int cc=0; cc<8; cc++) alphav[((t*2 + hh)*8 + r)*8 + cc] = alpha[t*8 + r];

                // O = O*alpha + P V_chunk, 16 head dims per call.
                for (size_t nb=0; nb<NBD; nb++) {
                    ime_tile(c, pp.data(), vth + (nb*K8p + g0)*128, nk8);
                    float* o = oacc.data() + nb*2*SLAB;
                    for (size_t i = 0; i < 2*SLAB;) {
                        const size_t vl = __riscv_vsetvl_e32m4(2*SLAB - i);
                        __riscv_vse32_v_f32m4(o + i, __riscv_vfmacc_vv_f32m4(
                            __riscv_vle32_v_f32m4(c + i, vl), __riscv_vle32_v_f32m4(o + i, vl),
                            __riscv_vle32_v_f32m4(alphav + i, vl), vl), vl);
                        i += vl;
                    }
                }
            }

            // out = O / l; a row whose every key was blocked gets the mean of V
            // (uniform weights, as the RVV kernel), in fp32.
            for (int r=0; r<nbq; r++) {
                float* o = out + ((size_t)(b0 + r)*n_q + h)*head_dim;
                if (mrow[r] == NINF) {
                    std::memcpy(o, vm + (size_t)kv*head_dim, head_dim*sizeof(float));
                    continue;
                }
                const float il = 1.0f/lrow[r];
                const int t = r/8, rr = r%8;
                for (size_t nb=0; nb<NBD; nb++)
                    for (int hh=0; hh<2; hh++) {
                        const float* src = oacc.data() + nb*2*SLAB + ((t*2 + hh)*8 + rr)*8;
                        for (int cc=0; cc<8; cc++) o[nb*16 + hh*8 + cc] = src[cc]*il;
                    }
            }
        }
    }
  }
    return kv_amax <= F16_MAX && q_amax*std::fabs(scale) <= F16_MAX;
}

// ---- fp16 tile tensors and TCM ----
//
// A tile tensor holds [rows][K] as fp16 in vfwmadot's activation layout,
// [ceil(rows/32)][K/8][4][8 rows][8 k]. That is also the layout of the GEMM's
// output tiles, so a Linear reads and writes it with no pack and no epilogue,
// and LayerNorm, GELU, the residual add and attention run on it with
// full-length vector operations. A whole encoder block then stays in fp16: only
// sums (GEMM, softmax, LayerNorm statistics) are carried in fp32. Rows past
// `rows` in the last block hold finite junk that nothing reads back.
//
// TCM (TCPU_IME_TCM=1): each A100 core has 384 KB of SRAM behind /dev/tcm that
// is faster than the cache hierarchy and far faster than DRAM. The A100's L2 is
// 1 MB per four cores, so the operand a GEMM re-reads (the activation rows, once
// per 16 output columns) falls out of cache at ViT shapes. Each thread copies
// its chunk of rows into its own TCM block and runs every weight panel over it;
// attention keeps one head's K and V^T there. The driver has no allocator and
// no arbitration: a thread pins itself to a core and uses that core's block, and
// another process using the TCM at the same time would corrupt both.

namespace {

struct tcm_info_t { void* base; size_t block_size; size_t block_num; };
struct tcm_block_info_t { uint32_t block_id, reserved; uint64_t phys, size, cpu_affinity_mask; };
constexpr unsigned long TCM_INFO_GET = _IOR('c', 7, int);
constexpr unsigned long TCM_BLOCK_INFO_GET = _IOWR('c', 9, int);

struct Tcm {
    int fd = -1;
    size_t block = 0;                   // bytes per block
    std::vector<uint64_t> cpus;         // per block: the cores it belongs to
    std::vector<char> taken;            // per block
    uint64_t cpus_taken = 0;
    std::mutex mu;
};

Tcm& tcm() {
    static Tcm t;
    static const bool once = [] {
        if (!hal::env::ime_tcm() || !ime_available()) return false;
        const int fd = open("/dev/tcm", O_RDWR | O_CLOEXEC);
        tcm_info_t info{};
        if (fd < 0 || ioctl(fd, TCM_INFO_GET, &info) != 0 || info.block_num == 0 || info.block_num > 64) {
            std::fprintf(stderr, "vla_simd: TCPU_IME_TCM=1 but /dev/tcm is not usable; TCM off\n");
            if (fd >= 0) close(fd);
            return false;
        }
        // One vla_simd process at a time (advisory; other TCM users do not take it).
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            std::fprintf(stderr, "vla_simd: /dev/tcm is in use by another vla_simd process; TCM off\n");
            close(fd);
            return false;
        }
        t.cpus.resize(info.block_num);
        t.taken.assign(info.block_num, 0);
        for (size_t i=0; i<info.block_num; i++) {
            tcm_block_info_t b{};
            b.block_id = (uint32_t)i;
            if (ioctl(fd, TCM_BLOCK_INFO_GET, &b) == 0 && b.size == info.block_size) t.cpus[i] = b.cpu_affinity_mask;
        }
        t.block = info.block_size;
        t.fd = fd;
        std::fprintf(stderr, "vla_simd: IME2 TCM on (%zu blocks of %zu KB)\n", info.block_num, info.block_size >> 10);
        return true;
    }();
    (void)once;
    return t;
}

// This thread's TCM block, or nullptr. The first call pins the thread to a free
// core that owns a free block and maps that block.
_Float16* tcm_block() {
    static thread_local int state = 0;
    static thread_local _Float16* p = nullptr;
    if (state) return p;
    state = 1;
    Tcm& t = tcm();
    if (t.fd < 0) return nullptr;
    cpu_set_t aff;
    CPU_ZERO(&aff);
    if (sched_getaffinity(0, sizeof aff, &aff) != 0) return nullptr;
    std::lock_guard<std::mutex> lk(t.mu);
    for (size_t b=0; b<t.cpus.size(); b++) {
        if (t.taken[b]) continue;
        for (int c=0; c<64; c++) {
            if (!(t.cpus[b] >> c & 1) || (t.cpus_taken >> c & 1) || !CPU_ISSET(c, &aff)) continue;
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            if (sched_setaffinity(0, sizeof one, &one) != 0) continue;
            void* m = mmap(nullptr, t.block, PROT_READ | PROT_WRITE, MAP_SHARED, t.fd, (off_t)(b*t.block));
            if (m == MAP_FAILED) { sched_setaffinity(0, sizeof aff, &aff); return nullptr; }
            t.taken[b] = 1;
            t.cpus_taken |= 1ull << c;
            p = static_cast<_Float16*>(m);
            return p;
        }
    }
    return nullptr;
}

size_t tcm_words() { return tcm().fd >= 0 ? tcm().block/2 : 0; }   // fp16 per block

#if TCPU_IME_MT == 4
// ime_tile with the accumulators started from two bias tiles (bias: [2][64]
// fp32, every row of a tile the 8 biases) and the results narrowed to fp16:
// tiles (t, 0) go to c0 + t*64 and tiles (t, 1) to c1 + t*64, each run of four
// in one store.
void ime_tile16(_Float16* c0, _Float16* c1, const _Float16* a, const _Float16* b, size_t k8,
                const float* bias) {
    const size_t vl = 64, vl2 = 128, vl4 = 256, vl8 = 512, odd = k8 & 1;
    size_t n2 = k8/2;
    __asm__ volatile(
        "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
        "vle32.v  v16, (%[bi]) \n\t" "addi %[bi], %[bi], 256 \n\t" "vle32.v  v18, (%[bi]) \n\t"
        "vmv.v.v  v20, v16 \n\t" "vmv.v.v  v22, v18 \n\t" "vmv.v.v  v24, v16 \n\t" "vmv.v.v  v26, v18 \n\t"
        "vmv.v.v  v28, v16 \n\t" "vmv.v.v  v30, v18 \n\t"
        "beqz %[n2], 2f \n\t"
        "1: \n\t"
        "vsetvli  zero, %[vl8], e16, m8, ta, ma \n\t"
        "vle16.v  v0, (%[a]) \n\t" "addi %[a], %[a], 1024 \n\t"
        "vsetvli  zero, %[vl4], e16, m4, ta, ma \n\t"
        "vle16.v  v8, (%[b]) \n\t" "addi %[b], %[b], 512 \n\t"
        "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
        "vfwmadot v16, v0, v8, fp16 \n\t" "vfwmadot v18, v0, v9, fp16 \n\t"
        "vfwmadot v20, v1, v8, fp16 \n\t" "vfwmadot v22, v1, v9, fp16 \n\t"
        "vfwmadot v24, v2, v8, fp16 \n\t" "vfwmadot v26, v2, v9, fp16 \n\t"
        "vfwmadot v28, v3, v8, fp16 \n\t" "vfwmadot v30, v3, v9, fp16 \n\t"
        "vfwmadot v16, v4, v10, fp16 \n\t" "vfwmadot v18, v4, v11, fp16 \n\t"
        "vfwmadot v20, v5, v10, fp16 \n\t" "vfwmadot v22, v5, v11, fp16 \n\t"
        "vfwmadot v24, v6, v10, fp16 \n\t" "vfwmadot v26, v6, v11, fp16 \n\t"
        "vfwmadot v28, v7, v10, fp16 \n\t" "vfwmadot v30, v7, v11, fp16 \n\t"
        "addi %[n2], %[n2], -1 \n\t"
        "bnez %[n2], 1b \n\t"
        "2: \n\t"
        "beqz %[odd], 3f \n\t"
        "vsetvli  zero, %[vl4], e16, m4, ta, ma \n\t"
        "vle16.v  v0, (%[a]) \n\t"
        "vsetvli  zero, %[vl2], e16, m2, ta, ma \n\t"
        "vle16.v  v8, (%[b]) \n\t"
        "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
        "vfwmadot v16, v0, v8, fp16 \n\t" "vfwmadot v18, v0, v9, fp16 \n\t"
        "vfwmadot v20, v1, v8, fp16 \n\t" "vfwmadot v22, v1, v9, fp16 \n\t"
        "vfwmadot v24, v2, v8, fp16 \n\t" "vfwmadot v26, v2, v9, fp16 \n\t"
        "vfwmadot v28, v3, v8, fp16 \n\t" "vfwmadot v30, v3, v9, fp16 \n\t"
        "3: \n\t"
        "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
        "vfncvt.f.f.w v0, v16 \n\t" "vfncvt.f.f.w v1, v20 \n\t"
        "vfncvt.f.f.w v2, v24 \n\t" "vfncvt.f.f.w v3, v28 \n\t"
        "vfncvt.f.f.w v4, v18 \n\t" "vfncvt.f.f.w v5, v22 \n\t"
        "vfncvt.f.f.w v6, v26 \n\t" "vfncvt.f.f.w v7, v30 \n\t"
        "vsetvli  zero, %[vl4], e16, m4, ta, ma \n\t"
        "vse16.v  v0, (%[c0]) \n\t" "vse16.v  v4, (%[c1]) \n\t"
        : [a] "+r"(a), [b] "+r"(b), [n2] "+r"(n2), [bi] "+r"(bias)
        : [vl] "r"(vl), [vl2] "r"(vl2), [vl4] "r"(vl4), [vl8] "r"(vl8), [odd] "r"(odd),
          [c0] "r"(c0), [c1] "r"(c1)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
#endif

inline void copy16(_Float16* d, const _Float16* s, size_t n) {
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e16m8(n - i);
        __riscv_vse16_v_f16m8(d + i, __riscv_vle16_v_f16m8(s + i, vl), vl);
        i += vl;
    }
}

// Sum the 8 lanes of each of the 8 rows of a [8][8] fp32 tile.
inline void fold8(float* out8, const float* tile) {
    for (int r=0; r<8; r++) {
        const float* p = tile + r*8;
        out8[r] = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
    }
}

// Broadcast 8 per-row values to a [8][8] tile.
inline void spread8(float* tile, const float* in8) {
    for (int r=0; r<8; r++)
        for (int c=0; c<8; c++) tile[r*8 + c] = in8[r];
}

} // namespace

bool ime_f16_enabled() {
#if TCPU_IME_MT == 4
    static const bool v = ime_available() && hal::env::ime_f16();
    return v;
#else
    return false;
#endif
}

size_t ime_f16_size(int rows, int K) { return (size_t)((rows + 31)/32)*(size_t)(K/8)*256; }

void ime_f16_bias_tiles(float* bt, const float* bias, int N) {
    for (int n8=0; n8<N/8; n8++)
        for (int r=0; r<8; r++)
            for (int c=0; c<8; c++) bt[(size_t)n8*64 + r*8 + c] = bias ? bias[n8*8 + c] : 0.0f;
}

void ime_f16_pack(uint16_t* t16, const float* x, int rows, int K) {
    const int K8 = K/8, MB = (rows + 31)/32;
    _Float16* t = reinterpret_cast<_Float16*>(t16);
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<_Float16> stage(K);
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int row=0; row<MB*32; row++) {
        _Float16* d = t + ((size_t)(row/32)*K8*4 + (row%32)/8)*64 + (row%8)*8;
        if (row >= rows) {
            for (int k8=0; k8<K8; k8++) std::memset(d + (size_t)k8*256, 0, 16);
            continue;
        }
        const float* xr = x + (size_t)row*K;
        for (size_t i = 0; i < (size_t)K;) {
            const size_t vl = __riscv_vsetvl_e32m4(K - i);
            __riscv_vse16_v_f16m2(stage.data() + i, __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(xr + i, vl), vl), vl);
            i += vl;
        }
        for (int k8=0; k8<K8; k8++) std::memcpy(d + (size_t)k8*256, stage.data() + k8*8, 16);
    }
  }
}

bool ime_f16_unpack(float* x, const uint16_t* t16, int rows, int K) {
    const int K8 = K/8;
    const _Float16* t = reinterpret_cast<const _Float16*>(t16);
    bool finite = true;
#if defined(_OPENMP)
    #pragma omp parallel reduction(&& : finite)
#endif
  {
    std::vector<_Float16> stage(K);
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t bad = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);   // inf*0 and nan*0 are nan
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int row=0; row<rows; row++) {
        const _Float16* s = t + ((size_t)(row/32)*K8*4 + (row%32)/8)*64 + (row%8)*8;
        for (int k8=0; k8<K8; k8++) std::memcpy(stage.data() + k8*8, s + (size_t)k8*256, 16);
        float* xr = x + (size_t)row*K;
        for (size_t i = 0; i < (size_t)K;) {
            const size_t vl = __riscv_vsetvl_e32m4(K - i);
            const vfloat32m4_t v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(stage.data() + i, vl), vl);
            bad = __riscv_vfmacc_vf_f32m4_tu(bad, 0.0f, v, vl);
            __riscv_vse32_v_f32m4(xr + i, v, vl);
            i += vl;
        }
    }
    const float b = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(
        bad, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax));
    finite = finite && b == 0.0f;
  }
    return finite;
}

void ime_f16_linear(const ImeF16Job* jobs, int n_jobs, const uint16_t* x16, int rows, int K) {
#if TCPU_IME_MT == 4
    const int K8 = K/8, MB = (rows + 31)/32;
    const size_t blk = (size_t)K8*256;                    // fp16 per 32-row block
    const _Float16* x = reinterpret_cast<const _Float16*>(x16);
#if defined(_OPENMP)
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    // With TCM the rows are the reused operand: a thread copies a chunk of row
    // blocks into its TCM block and runs weight panels over it. The work, row
    // blocks x panels (all jobs' panels in one list), is split evenly and
    // statically: thread i gets a contiguous run of cells in block-major order,
    // so it owns whole row blocks apart from a shared first and last one, and
    // all threads sweep the panels in the same order (the A100's L2 is shared by
    // four cores, so a panel is fetched from DRAM about once per cluster).
    // Without TCM, or with fewer row blocks than threads, the weight panel is
    // the reused operand instead (it fits L1/L2) and the rows are read in place.
    const int cap = (int)(tcm_words()/blk);
    if (cap >= 1 && MB >= nt) {
        int total_nb = 0;
        for (int j=0; j<n_jobs; j++) total_nb += jobs[j].N/16;
        // Blocks [b0, b0+bn) against panels [p0, p1) of the flattened list.
        auto run = [&](int b0, int bn, int p0, int p1) {
            _Float16* buf = tcm_block();
            for (int c0=b0; c0<b0+bn; c0+=cap) {
                const int cn = std::min(cap, b0 + bn - c0);
                const _Float16* src = x + (size_t)c0*blk;
                if (buf) {
                    copy16(buf, src, (size_t)cn*blk);
                    src = buf;
                }
                int base = 0;
                for (int j=0; j<n_jobs; j++) {
                    const ImeF16Job& job = jobs[j];
                    const int N8 = job.N/8, NB = job.N/16;
                    const int nb0 = std::max(p0 - base, 0), nb1 = std::min(p1 - base, NB);
                    base += NB;
                    const _Float16* wp = reinterpret_cast<const _Float16*>(job.Wh);
                    _Float16* out = reinterpret_cast<_Float16*>(job.out);
                    for (int nb=nb0; nb<nb1; nb++)
                        for (int i=0; i<cn; i++) {
                            _Float16* o = out + ((size_t)(c0 + i)*N8 + nb*2)*256;
                            ime_tile16(o, o + 256, src + (size_t)i*blk, wp + (size_t)nb*K8*128, (size_t)K8,
                                       job.bias_tiles + (size_t)nb*128);
                        }
                }
            }
        };
        const long cells = (long)MB*total_nb;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 1) num_threads(nt)
#endif
        for (int t=0; t<nt; t++) {
            const long f0 = cells*t/nt, f1 = cells*(t + 1)/nt;
            int bs = (int)(f0/total_nb), ps = (int)(f0%total_nb);
            const int be = (int)(f1/total_nb), pe = (int)(f1%total_nb);
            if (bs == be) { run(bs, 1, ps, pe); continue; }
            if (ps > 0) { run(bs, 1, ps, total_nb); bs++; }
            if (be > bs) run(bs, be - bs, 0, total_nb);
            if (pe > 0) run(be, 1, 0, pe);
        }
        return;
    }
    for (int j=0; j<n_jobs; j++) {
        const ImeF16Job& job = jobs[j];
        const int N8 = job.N/8, NB = job.N/16;
        const _Float16* wp = reinterpret_cast<const _Float16*>(job.Wh);
        _Float16* out = reinterpret_cast<_Float16*>(job.out);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 1) collapse(2)
#endif
        for (int nb=0; nb<NB; nb++)
            for (int mb=0; mb<MB; mb++) {
                _Float16* o = out + ((size_t)mb*N8 + nb*2)*256;
                ime_tile16(o, o + 256, x + (size_t)mb*blk, wp + (size_t)nb*K8*128, (size_t)K8,
                           job.bias_tiles + (size_t)nb*128);
            }
    }
#else
    (void)jobs; (void)n_jobs; (void)x16; (void)rows; (void)K;
#endif
}

void ime_f16_layernorm(uint16_t* out16, const uint16_t* x16, const float* w_tiles, const float* b_tiles,
                       int rows, int K, float eps) {
    const int K8 = K/8, MB = (rows + 31)/32;
    const _Float16* x = reinterpret_cast<const _Float16*>(x16);
    _Float16* out = reinterpret_cast<_Float16*>(out16);
    const float invK = 1.0f/(float)K;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int mb=0; mb<MB; mb++)
        for (int t=0; t<4; t++) {
            // One group of 8 rows: tiles [8 rows][8 k] at a stride of one slab.
            const _Float16* p = x + ((size_t)mb*K8*4 + t)*64;
            _Float16* q = out + ((size_t)mb*K8*4 + t)*64;
            alignas(64) float tile[64];
            float r8[8];
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, 64);
            for (int k8=0; k8<K8; k8++)
                acc = __riscv_vfwadd_wv_f32m2(acc, __riscv_vle16_v_f16m1(p + (size_t)k8*256, 64), 64);
            __riscv_vse32_v_f32m2(tile, acc, 64);
            fold8(r8, tile);
            for (int r=0; r<8; r++) r8[r] *= invK;
            spread8(tile, r8);
            const vfloat32m2_t mean = __riscv_vle32_v_f32m2(tile, 64);
            acc = __riscv_vfmv_v_f_f32m2(0.0f, 64);
            for (int k8=0; k8<K8; k8++) {
                const vfloat32m2_t d = __riscv_vfsub_vv_f32m2(
                    __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(p + (size_t)k8*256, 64), 64), mean, 64);
                acc = __riscv_vfmacc_vv_f32m2(acc, d, d, 64);
            }
            __riscv_vse32_v_f32m2(tile, acc, 64);
            fold8(r8, tile);
            for (int r=0; r<8; r++) r8[r] = 1.0f/std::sqrt(r8[r]*invK + eps);
            spread8(tile, r8);
            const vfloat32m2_t inv = __riscv_vle32_v_f32m2(tile, 64);
            for (int k8=0; k8<K8; k8++) {
                vfloat32m2_t y = __riscv_vfsub_vv_f32m2(
                    __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(p + (size_t)k8*256, 64), 64), mean, 64);
                y = __riscv_vfmul_vv_f32m2(y, inv, 64);
                y = __riscv_vfmadd_vv_f32m2(y, __riscv_vle32_v_f32m2(w_tiles + (size_t)k8*64, 64),
                                            __riscv_vle32_v_f32m2(b_tiles + (size_t)k8*64, 64), 64);
                __riscv_vse16_v_f16m1(q + (size_t)k8*256, __riscv_vfncvt_f_f_w_f16m1(y, 64), 64);
            }
        }
}

void ime_f16_add(uint16_t* x16, const uint16_t* y16, int rows, int K) {
    const size_t n = ime_f16_size(rows, K);
    _Float16* x = reinterpret_cast<_Float16*>(x16);
    const _Float16* y = reinterpret_cast<const _Float16*>(y16);
    const size_t step = 8192;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t b=0; b<n; b+=step) {
        const size_t e = std::min(n, b + step);
        for (size_t i = b; i < e;) {
            const size_t vl = __riscv_vsetvl_e16m8(e - i);
            __riscv_vse16_v_f16m8(x + i, __riscv_vfadd_vv_f16m8(__riscv_vle16_v_f16m8(x + i, vl),
                                                               __riscv_vle16_v_f16m8(y + i, vl), vl), vl);
            i += vl;
        }
    }
}

// 1/d for finite d >= 1: vfrec7 and two Newton steps (about 28 bits), far
// cheaper than a vector divide on the A100.
static inline vfloat32m4_t recip(vfloat32m4_t d, size_t vl) {
    vfloat32m4_t r = __riscv_vfrec7_v_f32m4(d, vl);
    r = __riscv_vfmul_vv_f32m4(r, __riscv_vfnmsac_vv_f32m4(__riscv_vfmv_v_f_f32m4(2.0f, vl), d, r, vl), vl);
    r = __riscv_vfmul_vv_f32m4(r, __riscv_vfnmsac_vv_f32m4(__riscv_vfmv_v_f_f32m4(2.0f, vl), d, r, vl), vl);
    return r;
}

// gelu_tanh's formula, v * sigmoid(2c(v + 0.044715 v^3)), evaluated in fp32.
void ime_f16_gelu_tanh(uint16_t* x16, int rows, int K) {
    const size_t n = ime_f16_size(rows, K);
    _Float16* x = reinterpret_cast<_Float16*>(x16);
    const float c2 = 2.0f*0.7978845608028654f;
    const size_t step = 4096;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t b=0; b<n; b+=step) {
        const size_t e = std::min(n, b + step);
        for (size_t i = b; i < e;) {
            const size_t vl = __riscv_vsetvl_e32m4(e - i);
            const vfloat32m4_t v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x + i, vl), vl);
            const vfloat32m4_t v3 = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(v, v, vl), v, vl);
            // exp's argument is capped at 60 so 1 + e stays finite for the reciprocal.
            const vfloat32m4_t y = __riscv_vfmin_vf_f32m4(
                __riscv_vfmul_vf_f32m4(__riscv_vfmacc_vf_f32m4(v, 0.044715f, v3, vl), -c2, vl), 60.0f, vl);
            const vfloat32m4_t d = __riscv_vfadd_vf_f32m4(exp_ps(y, vl), 1.0f, vl);
            __riscv_vse16_v_f16m2(x + i, __riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vv_f32m4(v, recip(d, vl), vl), vl), vl);
            i += vl;
        }
    }
}

// gelu_erf's formula (Abramowitz-Stegun 7.1.26 erf), evaluated in fp32.
void ime_f16_gelu_erf(uint16_t* x16, int rows, int K) {
    const size_t n = ime_f16_size(rows, K);
    _Float16* x = reinterpret_cast<_Float16*>(x16);
    const float inv_sqrt2 = 0.70710678118654752f, p = 0.3275911f;
    const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    const size_t step = 4096;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t b=0; b<n; b+=step) {
        const size_t e = std::min(n, b + step);
        for (size_t i = b; i < e;) {
            const size_t vl = __riscv_vsetvl_e32m4(e - i);
            const vfloat32m4_t v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x + i, vl), vl);
            const vfloat32m4_t z = __riscv_vfmul_vf_f32m4(__riscv_vfabs_v_f32m4(v, vl), inv_sqrt2, vl);
            const vfloat32m4_t t = recip(
                __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), p, z, vl), vl);
            vfloat32m4_t poly = __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(a4, vl), a5, t, vl);
            poly = __riscv_vfmadd_vv_f32m4(poly, t, __riscv_vfmv_v_f_f32m4(a3, vl), vl);
            poly = __riscv_vfmadd_vv_f32m4(poly, t, __riscv_vfmv_v_f_f32m4(a2, vl), vl);
            poly = __riscv_vfmadd_vv_f32m4(poly, t, __riscv_vfmv_v_f_f32m4(a1, vl), vl);
            poly = __riscv_vfmul_vv_f32m4(poly, t, vl);
            const vfloat32m4_t ex = exp_ps(__riscv_vfneg_v_f32m4(__riscv_vfmul_vv_f32m4(z, z, vl), vl), vl);
            const vfloat32m4_t erf_abs = __riscv_vfnmsac_vv_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), poly, ex, vl);
            const vfloat32m4_t erf = __riscv_vfsgnj_vv_f32m4(erf_abs, v, vl);
            const vfloat32m4_t y = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(v, 0.5f, vl),
                                                         __riscv_vfadd_vf_f32m4(erf, 1.0f, vl), vl);
            __riscv_vse16_v_f16m2(x + i, __riscv_vfncvt_f_f_w_f16m2(y, vl), vl);
            i += vl;
        }
    }
}

size_t ime_f16_rope_size(int rows, int head_dim) { return (size_t)((rows + 31)/32)*(size_t)(head_dim/8)*256; }

void ime_f16_rope_tables(float* ct, float* st, const float* cos_t, const float* sin_t,
                         int rows, int first, int head_dim) {
    const int MB = (rows + 31)/32, J = head_dim/8;
    for (int mb=0; mb<MB; mb++)
        for (int j=0; j<J; j++)
            for (int r=0; r<32; r++)
                for (int c=0; c<8; c++) {
                    const int row = mb*32 + r;
                    const bool rot = row >= first && row < rows;
                    const size_t o = ((size_t)mb*J + j)*256 + r*8 + c;
                    ct[o] = rot ? cos_t[(size_t)(row - first)*head_dim + j*8 + c] : 1.0f;
                    st[o] = rot ? sin_t[(size_t)(row - first)*head_dim + j*8 + c] : 0.0f;
                }
}

void ime_f16_rope(uint16_t* x16, const float* ct, const float* st, int rows, int heads, int head_dim) {
    const int MB = (rows + 31)/32, J = head_dim/8, JH = J/2;
    const size_t K8 = (size_t)heads*J;
    _Float16* x = reinterpret_cast<_Float16*>(x16);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int mb=0; mb<MB; mb++)
        for (int h=0; h<heads; h++)
            for (int j=0; j<JH; j++) {
                _Float16* pa = x + ((size_t)mb*K8 + (size_t)h*J + j)*256;
                _Float16* pb = pa + (size_t)JH*256;
                const float* c1 = ct + ((size_t)mb*J + j)*256;
                const float* s1 = st + ((size_t)mb*J + j)*256;
                const float* c2 = c1 + (size_t)JH*256;
                const float* s2 = s1 + (size_t)JH*256;
                for (size_t i=0; i<256; i+=128) {
                    const vfloat32m4_t a = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(pa + i, 128), 128);
                    const vfloat32m4_t b = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(pb + i, 128), 128);
                    const vfloat32m4_t ra = __riscv_vfnmsac_vv_f32m4(
                        __riscv_vfmul_vv_f32m4(a, __riscv_vle32_v_f32m4(c1 + i, 128), 128),
                        b, __riscv_vle32_v_f32m4(s1 + i, 128), 128);
                    const vfloat32m4_t rb = __riscv_vfmacc_vv_f32m4(
                        __riscv_vfmul_vv_f32m4(b, __riscv_vle32_v_f32m4(c2 + i, 128), 128),
                        a, __riscv_vle32_v_f32m4(s2 + i, 128), 128);
                    __riscv_vse16_v_f16m2(pa + i, __riscv_vfncvt_f_f_w_f16m2(ra, 128), 128);
                    __riscv_vse16_v_f16m2(pb + i, __riscv_vfncvt_f_f_w_f16m2(rb, 128), 128);
                }
            }
}

bool ime_f16_attention(uint16_t* out16, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                       int rows, int heads, int head_dim, float scale) {
#if TCPU_IME_MT == 4
    if (head_dim % 16 != 0 || head_dim > 256 || rows <= 0 || !ime_f16_enabled()) return false;
    const size_t SLAB = 256;                              // one 8-key group of a query block, fp32
    const size_t SKP  = ((size_t)rows + 15)/16*16;        // keys padded to a weight tile
    const size_t K8q  = (size_t)head_dim/8;
    const size_t K8p  = SKP/8;
    const size_t NBD  = (size_t)head_dim/16;
    const size_t D8   = (size_t)heads*K8q;
    const size_t kp_n = SKP/16*K8q*128;                   // fp16 per head
    const size_t vt_n = NBD*K8p*128;
    const float  NINF = -std::numeric_limits<float>::infinity();
    const int    qblocks = (rows + 31)/32;
    const _Float16* q = reinterpret_cast<const _Float16*>(q16);
    const _Float16* k = reinterpret_cast<const _Float16*>(k16);
    const _Float16* v = reinterpret_cast<const _Float16*>(v16);
    _Float16* out = reinterpret_cast<_Float16*>(out16);

    // Keys past `rows` in the last key group(s) are junk: -inf for them.
    const size_t kfirst = (size_t)rows/8;                 // first key group with a padded key
    alignas(64) float padmask[2*256];
    for (size_t g=kfirst; g<K8p; g++)
        for (size_t i=0; i<SLAB; i++)
            padmask[(g - kfirst)*SLAB + i] = g*8 + i%8 < (size_t)rows ? 0.0f : NINF;

    // A unit is one head and a share of the query blocks; a thread packs the
    // head's K and V^T (into its TCM block when they fit) and runs the share.
#if defined(_OPENMP)
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    int parts = (3*nt + heads - 1)/heads;
    if (parts > qblocks) parts = qblocks;
    if (parts < 1) parts = 1;

    // idx[i]: the 8x8 transpose as a gather.
    alignas(64) uint16_t tidx[64];
    for (int i=0; i<64; i++) tidx[i] = (uint16_t)((i%8)*8 + i/8);

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    constexpr size_t KC = 128;                            // keys per chunk (multiple of 16)
    const size_t KC8 = KC/8;
    std::vector<_Float16> pp(KC8*SLAB), kv_heap;
    std::vector<float> sp(KC8*SLAB), oacc(NBD*2*SLAB);
    alignas(64) float acc[256], rowv[256], alphav[512], c[512], ilv[256];
    float mrow[32], lrow[32], alpha[32];
    const vuint16m1_t vtidx = __riscv_vle16_v_u16m1(tidx, 64);
    _Float16* kp = tcm_block();
    if (!kp || kp_n + vt_n > tcm_words()) {
        kv_heap.resize(kp_n + vt_n);
        kp = kv_heap.data();
    }
    _Float16* vt = kp + kp_n;
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 1) collapse(2)
#endif
    for (int h=0; h<heads; h++)
        for (int part=0; part<parts; part++) {
            // K tile pair (nb, k8): halves are key tiles 2nb and 2nb+1, whole tile copies.
            for (size_t kt=0; kt<SKP/8; kt++)
                for (size_t k8=0; k8<K8q; k8++)
                    __riscv_vse16_v_f16m1(kp + ((kt/2)*K8q + k8)*128 + (kt%2)*64,
                        __riscv_vle16_v_f16m1(k + (((kt/4)*D8 + h*K8q + k8)*4 + kt%4)*64, 64), 64);
            // V^T tile (nb, key group), half hh: the transpose of V's tile (key group, dims).
            for (size_t kg=0; kg<K8p; kg++)
                for (size_t d8=0; d8<K8q; d8++)
                    __riscv_vse16_v_f16m1(vt + ((d8/2)*K8p + kg)*128 + (d8%2)*64,
                        __riscv_vrgather_vv_f16m1(
                            __riscv_vle16_v_f16m1(v + (((kg/4)*D8 + h*K8q + d8)*4 + kg%4)*64, 64), vtidx, 64), 64);

            const int blk0 = (int)((long)qblocks*part/parts), blk1 = (int)((long)qblocks*(part + 1)/parts);
            for (int blk=blk0; blk<blk1; blk++) {
                const _Float16* qp = q + ((size_t)blk*D8 + h*K8q)*256;     // [K8q][4][64]
                for (int r=0; r<32; r++) { mrow[r] = NINF; lrow[r] = 0.0f; }
                std::memset(oacc.data(), 0, oacc.size()*sizeof(float));

                for (size_t c0=0; c0<SKP; c0+=KC) {
                    const size_t kc  = SKP - c0 < KC ? SKP - c0 : KC;
                    const size_t nk8 = kc/8, g0 = c0/8;
                    for (size_t nb=0; nb<kc/16; nb++)
                        ime_tile(sp.data() + nb*2*SLAB, qp, kp + (c0/16 + nb)*K8q*128, K8q, SLAB, 64);

                    // scale (and pad mask); slab-wise max -> per-row chunk max -> running max.
                    for (size_t i = 0; i < SLAB;) {
                        const size_t vl = __riscv_vsetvl_e32m4(SLAB - i);
                        vfloat32m4_t mx = __riscv_vfmv_v_f_f32m4(NINF, vl);
                        for (size_t k8=0; k8<nk8; k8++) {
                            float* s = sp.data() + k8*SLAB + i;
                            vfloat32m4_t x = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(s, vl), scale, vl);
                            if (g0 + k8 >= kfirst)
                                x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(
                                    padmask + (g0 + k8 - kfirst)*SLAB + i, vl), vl);
                            __riscv_vse32_v_f32m4(s, x, vl);
                            mx = __riscv_vfmax_vv_f32m4(mx, x, vl);
                        }
                        __riscv_vse32_v_f32m4(acc + i, mx, vl);
                        i += vl;
                    }
                    for (int r=0; r<32; r++) {
                        float m = mrow[r];
                        for (int cc=0; cc<8; cc++) m = acc[r*8 + cc] > m ? acc[r*8 + cc] : m;
                        alpha[r] = mrow[r] == NINF ? 0.0f : std::exp(mrow[r] - m);
                        mrow[r] = m;
                        for (int cc=0; cc<8; cc++) rowv[r*8 + cc] = m;
                    }

                    // P = exp(S - m) -> fp16; slab-wise sums -> running sums.
                    for (size_t i = 0; i < SLAB;) {
                        const size_t vl = __riscv_vsetvl_e32m4(SLAB - i);
                        const vfloat32m4_t mv = __riscv_vle32_v_f32m4(rowv + i, vl);
                        const vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(0.6931280494f, vl);
                        const vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.2402367741f, vl);
                        const vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.0558702424f, vl);
                        const vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.0095902430f, vl);
                        vfloat32m4_t sum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                        for (size_t k8=0; k8<nk8; k8++) {
                            const vfloat32m4_t e = exp_softmax(__riscv_vfsub_vv_f32m4(
                                __riscv_vle32_v_f32m4(sp.data() + k8*SLAB + i, vl), mv, vl), c1, c2, c3, c4, vl);
                            sum = __riscv_vfadd_vv_f32m4(sum, e, vl);
                            __riscv_vse16_v_f16m2(pp.data() + k8*SLAB + i, __riscv_vfncvt_f_f_w_f16m2(e, vl), vl);
                        }
                        __riscv_vse32_v_f32m4(acc + i, sum, vl);
                        i += vl;
                    }
                    for (int r=0; r<32; r++) {
                        float z = 0.0f;
                        for (int cc=0; cc<8; cc++) z += acc[r*8 + cc];
                        lrow[r] = lrow[r]*alpha[r] + z;
                    }
                    for (int t=0; t<4; t++)
                        for (int hh=0; hh<2; hh++)
                            for (int r=0; r<8; r++)
                                for (int cc=0; cc<8; cc++) alphav[((t*2 + hh)*8 + r)*8 + cc] = alpha[t*8 + r];

                    // O = O*alpha + P V_chunk, 16 head dims per call.
                    for (size_t nb=0; nb<NBD; nb++) {
                        ime_tile(c, pp.data(), vt + (nb*K8p + g0)*128, nk8);
                        float* o = oacc.data() + nb*2*SLAB;
                        for (size_t i = 0; i < 2*SLAB;) {
                            const size_t vl = __riscv_vsetvl_e32m4(2*SLAB - i);
                            __riscv_vse32_v_f32m4(o + i, __riscv_vfmacc_vv_f32m4(
                                __riscv_vle32_v_f32m4(c + i, vl), __riscv_vle32_v_f32m4(o + i, vl),
                                __riscv_vle32_v_f32m4(alphav + i, vl), vl), vl);
                            i += vl;
                        }
                    }
                }

                // out tile (dims nb*16 + hh*8 .., rows t*8 ..) = O / l, narrowed in place.
                for (int r=0; r<32; r++) {
                    const float il = 1.0f/lrow[r];
                    for (int cc=0; cc<8; cc++) ilv[r*8 + cc] = il;
                }
                for (size_t nb=0; nb<NBD; nb++)
                    for (int t=0; t<4; t++)
                        for (int hh=0; hh<2; hh++) {
                            const vfloat32m2_t o = __riscv_vfmul_vv_f32m2(
                                __riscv_vle32_v_f32m2(oacc.data() + nb*2*SLAB + (t*2 + hh)*64, 64),
                                __riscv_vle32_v_f32m2(ilv + t*64, 64), 64);
                            __riscv_vse16_v_f16m1(out + (((size_t)blk*D8 + h*K8q + nb*2 + hh)*4 + t)*64,
                                                  __riscv_vfncvt_f_f_w_f16m1(o, 64), 64);
                        }
            }
        }
  }
    return true;
#else
    (void)out16; (void)q16; (void)k16; (void)v16; (void)rows; (void)heads; (void)head_dim; (void)scale;
    return false;
#endif
}

} // namespace tcpu

#endif // TCPU_HAL_RVV && TCPU_RVV_IME

