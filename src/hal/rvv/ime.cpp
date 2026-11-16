/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// SpacemiT IME2 fp16 GEMM for the K3's A100 cores (VLEN 1024).
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
// on the A100 cores (see detect()).

#include "../arch.h"
#if TCPU_HAL_RVV && TCPU_RVV_IME

#include "../common/env.h"
#include "../../ops/lm_ops.h"
#include <riscv_vector.h>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// a: [k8][MT][64] fp16, b: [k8][2][64] fp16, c: [MT][2][64] fp32 (overwritten).
void ime_tile(float* c, const _Float16* a, const _Float16* b, size_t k8) {
    const size_t vl = 64;
    if constexpr (MT == 4) {
        __asm__ volatile(
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vmv.v.i  v16, 0 \n\t" "vmv.v.i  v18, 0 \n\t" "vmv.v.i  v20, 0 \n\t" "vmv.v.i  v22, 0 \n\t"
            "vmv.v.i  v24, 0 \n\t" "vmv.v.i  v26, 0 \n\t" "vmv.v.i  v28, 0 \n\t" "vmv.v.i  v30, 0 \n\t"
            "vsetvli  zero, %[vl], e16, m1, ta, ma \n\t"
            "1: \n\t"
            "vle16.v  v0, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vle16.v  v4, (%[b]) \n\t" "addi %[b], %[b], 128 \n\t"
            "vle16.v  v5, (%[b]) \n\t" "addi %[b], %[b], 128 \n\t"
            "vle16.v  v1, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v16, v0, v4, fp16 \n\t"
            "vfwmadot v18, v0, v5, fp16 \n\t"
            "vle16.v  v2, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v20, v1, v4, fp16 \n\t"
            "vfwmadot v22, v1, v5, fp16 \n\t"
            "vle16.v  v3, (%[a]) \n\t" "addi %[a], %[a], 128 \n\t"
            "vfwmadot v24, v2, v4, fp16 \n\t"
            "vfwmadot v26, v2, v5, fp16 \n\t"
            "vfwmadot v28, v3, v4, fp16 \n\t"
            "vfwmadot v30, v3, v5, fp16 \n\t"
            "addi %[n], %[n], -1 \n\t"
            "bnez %[n], 1b \n\t"
            "vsetvli  zero, %[vl], e32, m2, ta, ma \n\t"
            "vse32.v  v16, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v18, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v20, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v22, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v24, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v26, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v28, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v30, (%[c]) \n\t"
            : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [n] "+r"(k8)
            : [vl] "r"(vl)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20",
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
            "vse32.v  v8,  (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v10, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v12, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v14, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v16, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v18, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v20, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v22, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v24, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v26, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v28, (%[c]) \n\t" "addi %[c], %[c], 256 \n\t"
            "vse32.v  v30, (%[c]) \n\t"
            : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [n] "+r"(k8)
            : [vl] "r"(vl)
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

void pack_weights_ime(const float* W, uint16_t* Wh, int N, int K) {
    pack_w(Wh, N, K, [W](size_t i) { return (_Float16)W[i]; });
}

void pack_weights_ime_bf16(const uint16_t* Wb16, uint16_t* Wh, int N, int K) {
    pack_w(Wh, N, K, [Wb16](size_t i) { return bf16_to_f16(Wb16[i]); });
}

void dense_linear_ime(float* out, const float* x, const uint16_t* Wh, const float* bias,
                      int seq, int N, int K) {
    const int K8 = (K+7)/8, MB = (seq + 8*MT - 1)/(8*MT), NB = N/16;
    const _Float16* wp = reinterpret_cast<const _Float16*>(Wh);

    // Activations -> fp16 tiles [MB][K8][MT][8][8], zero rows/columns as padding.
    // thread_local: the view fan-out runs several threads through one Linear.
    static thread_local std::vector<_Float16> xp_buf;
    const size_t nx = (size_t)MB*K8*MT*64;
    if (xp_buf.size() < nx) xp_buf.resize(nx);
    _Float16* xp = xp_buf.data();
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
                for (int k8=0; k8<K8; k8++) {
                    _Float16* dk = d + (size_t)k8*MT*64;
                    const int k0 = k8*8;
                    if (k0 + 8 <= K) {
                        const vfloat32m1_t v = __riscv_vle32_v_f32m1(xr + k0, 8);
                        __riscv_vse16_v_f16mf2(dk, __riscv_vfncvt_f_f_w_f16mf2(v, 8), 8);
                    } else {
                        for (int c=0; c<8; c++)
                            dk[c] = k0 + c < K ? (_Float16)xr[k0 + c] : (_Float16)0.0f;
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
                    __riscv_vse32_v_f32m1(o, __riscv_vfadd_vv_f32m1(
                        __riscv_vle32_v_f32m1(ct, 8), b0, 8), 8);
                    __riscv_vse32_v_f32m1(o + 8, __riscv_vfadd_vv_f32m1(
                        __riscv_vle32_v_f32m1(ct + 64, 8), b1, 8), 8);
                }
        }
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_RVV && TCPU_RVV_IME
