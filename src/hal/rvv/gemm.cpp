/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// RISC-V Vector GEMM backend (SpacemiT K3 tuning: X100 cores, VLEN 256).
//
// The packed-panel layout is the shared one from pack_weights16: a 16-wide
// weight column per k, contiguous. At VLEN 256 that column is exactly one
// e32/LMUL=2 register group, so the micro-kernel is MR x 16 with MR accumulator
// groups plus one weight group, and the x operand comes straight from a scalar
// FP register (vfmacc.vf) - RVV has no lane-indexed FMA and needs no broadcast.
// 32 vector registers / LMUL 2 = 16 groups: MR <= 14 fits without spills.
// MR is fixed at build time (-DVLA_RVV_MR); tools/bench_tile measures it.

#include "../arch.h"
#if TCPU_HAL_RVV

#include "../simd.h"
#include "../common/env.h"
#include "../../ops/lm_ops.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
using std::size_t;

#ifndef TCPU_RVV_MR
#define TCPU_RVV_MR 14
#endif

namespace tcpu {
using hal::env::gemm_force_static;

namespace {
// The kernels assume one 16-float panel column fits one LMUL=2 group, which the
// build asserts with _zvl256b. A binary built for VLEN>=256 run on a narrower
// part would compute garbage, so refuse to load instead.
struct VlenCheck {
    VlenCheck() {
        if (__riscv_vsetvlmax_e32m2() < 16) {
            std::fprintf(stderr, "vla_simd: this build needs RVV VLEN >= 256 (found %zu); "
                                 "rebuild with -DVLA_SCALAR=ON for this CPU\n",
                         __riscv_vsetvlmax_e32m1()*32);
            std::abort();
        }
    }
};
const VlenCheck vlen_check;
} // namespace

// Unpacked row GEMM (W is [N][K], K contiguous): 4 output rows per block, one
// LMUL=4 accumulator per row strip-mined along K, reduced at the end.
static inline void mm_block4_rvv(float* out, const float* x,
                                 const float* w0, const float* w1, const float* w2, const float* w3,
                                 int seq, int N, int K, const float* bias, int n0) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*K;
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        vfloat32m4_t a1 = a0, a2 = a0, a3 = a0;
        for (size_t k = 0, kk = (size_t)K; k < kk;) {
            const size_t vl = __riscv_vsetvl_e32m4(kk - k);
            const vfloat32m4_t xv = __riscv_vle32_v_f32m4(xt+k, vl);
            a0 = __riscv_vfmacc_vv_f32m4_tu(a0, xv, __riscv_vle32_v_f32m4(w0+k, vl), vl);
            a1 = __riscv_vfmacc_vv_f32m4_tu(a1, xv, __riscv_vle32_v_f32m4(w1+k, vl), vl);
            a2 = __riscv_vfmacc_vv_f32m4_tu(a2, xv, __riscv_vle32_v_f32m4(w2+k, vl), vl);
            a3 = __riscv_vfmacc_vv_f32m4_tu(a3, xv, __riscv_vle32_v_f32m4(w3+k, vl), vl);
            k += vl;
        }
        float* o = out + (size_t)t*N + n0;
        const float s0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a0, z, vlmax));
        const float s1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a1, z, vlmax));
        const float s2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a2, z, vlmax));
        const float s3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a3, z, vlmax));
        o[0] = bias ? s0+bias[n0+0] : s0;
        o[1] = bias ? s1+bias[n0+1] : s1;
        o[2] = bias ? s2+bias[n0+2] : s2;
        o[3] = bias ? s3+bias[n0+3] : s3;
    }
}

// ROWS x 16 C-resident packed micro-kernel. Per k: one 16-float weight load
// shared by all ROWS rows, ROWS scalar x loads, ROWS vfmacc.vf. Same k order and
// same c += w*x[k] per lane as the scalar reference. RVV types are sizeless and
// cannot form arrays, so the accumulators are named and pruned by if constexpr.
template <int ROWS>
static inline void mm_pack_rows16_rvv(float* out, const float* x, const float* wp,
                                      const float* bias, int N, int K, int n0) {
    constexpr size_t VL = 16;
    const size_t ks = (size_t)K;
    vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c6 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c7 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c8 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c9 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c10 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c11 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c12 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    vfloat32m2_t c13 = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    size_t k = 0;
    for (; k+2<=ks; k+=2) {
        const vfloat32m2_t b0 = __riscv_vle32_v_f32m2(wp+k*16, VL);
        const vfloat32m2_t b1 = __riscv_vle32_v_f32m2(wp+k*16+16, VL);
        if constexpr (ROWS > 0) c0 = __riscv_vfmacc_vf_f32m2(c0, x[0*ks+k], b0, VL);
        if constexpr (ROWS > 1) c1 = __riscv_vfmacc_vf_f32m2(c1, x[1*ks+k], b0, VL);
        if constexpr (ROWS > 2) c2 = __riscv_vfmacc_vf_f32m2(c2, x[2*ks+k], b0, VL);
        if constexpr (ROWS > 3) c3 = __riscv_vfmacc_vf_f32m2(c3, x[3*ks+k], b0, VL);
        if constexpr (ROWS > 4) c4 = __riscv_vfmacc_vf_f32m2(c4, x[4*ks+k], b0, VL);
        if constexpr (ROWS > 5) c5 = __riscv_vfmacc_vf_f32m2(c5, x[5*ks+k], b0, VL);
        if constexpr (ROWS > 6) c6 = __riscv_vfmacc_vf_f32m2(c6, x[6*ks+k], b0, VL);
        if constexpr (ROWS > 7) c7 = __riscv_vfmacc_vf_f32m2(c7, x[7*ks+k], b0, VL);
        if constexpr (ROWS > 8) c8 = __riscv_vfmacc_vf_f32m2(c8, x[8*ks+k], b0, VL);
        if constexpr (ROWS > 9) c9 = __riscv_vfmacc_vf_f32m2(c9, x[9*ks+k], b0, VL);
        if constexpr (ROWS > 10) c10 = __riscv_vfmacc_vf_f32m2(c10, x[10*ks+k], b0, VL);
        if constexpr (ROWS > 11) c11 = __riscv_vfmacc_vf_f32m2(c11, x[11*ks+k], b0, VL);
        if constexpr (ROWS > 12) c12 = __riscv_vfmacc_vf_f32m2(c12, x[12*ks+k], b0, VL);
        if constexpr (ROWS > 13) c13 = __riscv_vfmacc_vf_f32m2(c13, x[13*ks+k], b0, VL);
        if constexpr (ROWS > 0) c0 = __riscv_vfmacc_vf_f32m2(c0, x[0*ks+k+1], b1, VL);
        if constexpr (ROWS > 1) c1 = __riscv_vfmacc_vf_f32m2(c1, x[1*ks+k+1], b1, VL);
        if constexpr (ROWS > 2) c2 = __riscv_vfmacc_vf_f32m2(c2, x[2*ks+k+1], b1, VL);
        if constexpr (ROWS > 3) c3 = __riscv_vfmacc_vf_f32m2(c3, x[3*ks+k+1], b1, VL);
        if constexpr (ROWS > 4) c4 = __riscv_vfmacc_vf_f32m2(c4, x[4*ks+k+1], b1, VL);
        if constexpr (ROWS > 5) c5 = __riscv_vfmacc_vf_f32m2(c5, x[5*ks+k+1], b1, VL);
        if constexpr (ROWS > 6) c6 = __riscv_vfmacc_vf_f32m2(c6, x[6*ks+k+1], b1, VL);
        if constexpr (ROWS > 7) c7 = __riscv_vfmacc_vf_f32m2(c7, x[7*ks+k+1], b1, VL);
        if constexpr (ROWS > 8) c8 = __riscv_vfmacc_vf_f32m2(c8, x[8*ks+k+1], b1, VL);
        if constexpr (ROWS > 9) c9 = __riscv_vfmacc_vf_f32m2(c9, x[9*ks+k+1], b1, VL);
        if constexpr (ROWS > 10) c10 = __riscv_vfmacc_vf_f32m2(c10, x[10*ks+k+1], b1, VL);
        if constexpr (ROWS > 11) c11 = __riscv_vfmacc_vf_f32m2(c11, x[11*ks+k+1], b1, VL);
        if constexpr (ROWS > 12) c12 = __riscv_vfmacc_vf_f32m2(c12, x[12*ks+k+1], b1, VL);
        if constexpr (ROWS > 13) c13 = __riscv_vfmacc_vf_f32m2(c13, x[13*ks+k+1], b1, VL);
    }
    for (; k<ks; k++) {
        const vfloat32m2_t b0 = __riscv_vle32_v_f32m2(wp+k*16, VL);
        if constexpr (ROWS > 0) c0 = __riscv_vfmacc_vf_f32m2(c0, x[0*ks+k], b0, VL);
        if constexpr (ROWS > 1) c1 = __riscv_vfmacc_vf_f32m2(c1, x[1*ks+k], b0, VL);
        if constexpr (ROWS > 2) c2 = __riscv_vfmacc_vf_f32m2(c2, x[2*ks+k], b0, VL);
        if constexpr (ROWS > 3) c3 = __riscv_vfmacc_vf_f32m2(c3, x[3*ks+k], b0, VL);
        if constexpr (ROWS > 4) c4 = __riscv_vfmacc_vf_f32m2(c4, x[4*ks+k], b0, VL);
        if constexpr (ROWS > 5) c5 = __riscv_vfmacc_vf_f32m2(c5, x[5*ks+k], b0, VL);
        if constexpr (ROWS > 6) c6 = __riscv_vfmacc_vf_f32m2(c6, x[6*ks+k], b0, VL);
        if constexpr (ROWS > 7) c7 = __riscv_vfmacc_vf_f32m2(c7, x[7*ks+k], b0, VL);
        if constexpr (ROWS > 8) c8 = __riscv_vfmacc_vf_f32m2(c8, x[8*ks+k], b0, VL);
        if constexpr (ROWS > 9) c9 = __riscv_vfmacc_vf_f32m2(c9, x[9*ks+k], b0, VL);
        if constexpr (ROWS > 10) c10 = __riscv_vfmacc_vf_f32m2(c10, x[10*ks+k], b0, VL);
        if constexpr (ROWS > 11) c11 = __riscv_vfmacc_vf_f32m2(c11, x[11*ks+k], b0, VL);
        if constexpr (ROWS > 12) c12 = __riscv_vfmacc_vf_f32m2(c12, x[12*ks+k], b0, VL);
        if constexpr (ROWS > 13) c13 = __riscv_vfmacc_vf_f32m2(c13, x[13*ks+k], b0, VL);
    }
    const vfloat32m2_t bb = bias ? __riscv_vle32_v_f32m2(bias+n0, VL) : __riscv_vfmv_v_f_f32m2(0.0f, VL);
    if constexpr (ROWS > 0) __riscv_vse32_v_f32m2(out+(size_t)0*N+n0, __riscv_vfadd_vv_f32m2(c0, bb, VL), VL);
    if constexpr (ROWS > 1) __riscv_vse32_v_f32m2(out+(size_t)1*N+n0, __riscv_vfadd_vv_f32m2(c1, bb, VL), VL);
    if constexpr (ROWS > 2) __riscv_vse32_v_f32m2(out+(size_t)2*N+n0, __riscv_vfadd_vv_f32m2(c2, bb, VL), VL);
    if constexpr (ROWS > 3) __riscv_vse32_v_f32m2(out+(size_t)3*N+n0, __riscv_vfadd_vv_f32m2(c3, bb, VL), VL);
    if constexpr (ROWS > 4) __riscv_vse32_v_f32m2(out+(size_t)4*N+n0, __riscv_vfadd_vv_f32m2(c4, bb, VL), VL);
    if constexpr (ROWS > 5) __riscv_vse32_v_f32m2(out+(size_t)5*N+n0, __riscv_vfadd_vv_f32m2(c5, bb, VL), VL);
    if constexpr (ROWS > 6) __riscv_vse32_v_f32m2(out+(size_t)6*N+n0, __riscv_vfadd_vv_f32m2(c6, bb, VL), VL);
    if constexpr (ROWS > 7) __riscv_vse32_v_f32m2(out+(size_t)7*N+n0, __riscv_vfadd_vv_f32m2(c7, bb, VL), VL);
    if constexpr (ROWS > 8) __riscv_vse32_v_f32m2(out+(size_t)8*N+n0, __riscv_vfadd_vv_f32m2(c8, bb, VL), VL);
    if constexpr (ROWS > 9) __riscv_vse32_v_f32m2(out+(size_t)9*N+n0, __riscv_vfadd_vv_f32m2(c9, bb, VL), VL);
    if constexpr (ROWS > 10) __riscv_vse32_v_f32m2(out+(size_t)10*N+n0, __riscv_vfadd_vv_f32m2(c10, bb, VL), VL);
    if constexpr (ROWS > 11) __riscv_vse32_v_f32m2(out+(size_t)11*N+n0, __riscv_vfadd_vv_f32m2(c11, bb, VL), VL);
    if constexpr (ROWS > 12) __riscv_vse32_v_f32m2(out+(size_t)12*N+n0, __riscv_vfadd_vv_f32m2(c12, bb, VL), VL);
    if constexpr (ROWS > 13) __riscv_vse32_v_f32m2(out+(size_t)13*N+n0, __riscv_vfadd_vv_f32m2(c13, bb, VL), VL);
}

template <int R>
static inline void mm_pack_dispatch(float* out, const float* x, const float* wp,
                                    const float* bias, int rows, int N, int K, int n0) {
    if constexpr (R > 1) {
        if (rows < R) {
            mm_pack_dispatch<R-1>(out, x, wp, bias, rows, N, K, n0);
            return;
        }
    }
    mm_pack_rows16_rvv<R>(out, x, wp, bias, N, K, n0);
}

void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K) {
    constexpr int NR = 4;
    const int nblocks = (N+NR-1)/NR;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int b=0; b<nblocks; b++) {
        const int n0 = b*NR;
        if (n0+NR <= N) {
            mm_block4_rvv(out, x, W+(size_t)n0*K, W+(size_t)(n0+1)*K, W+(size_t)(n0+2)*K,
                          W+(size_t)(n0+3)*K, seq, N, K, bias, n0);
        } else {
            for (int j=0; n0+j < N; j++) {
                const float* wr = W+(size_t)(n0+j)*K;
                for (int t=0; t<seq; t++) {
                    const float s = simd_dot(x+(size_t)t*K, wr, K);
                    out[(size_t)t*N+n0+j] = bias ? s+bias[n0+j] : s;
                }
            }
        }
    }
}

void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K) {
    // Callers guarantee N % 16 == 0 (pack_weights16 only emits full blocks).
    constexpr int MR = TCPU_RVV_MR;
    static_assert(MR >= 1 && MR <= 14, "MR accumulators + 1 weight group must fit 16 LMUL=2 groups");
    const int nblocks = N/16;
    const int mtiles  = (seq+MR-1)/MR;
#if defined(_OPENMP)
    // Dynamic whole-panel chunks by default, as on NEON: an idle thread steals a
    // 16-wide weight panel from a straggler; TCPU_GEMM_SCHED=static reverts.
    if (!gemm_force_static()) {
        #pragma omp parallel for schedule(dynamic, mtiles) collapse(2)
        for (int b=0; b<nblocks; b++) {
            for (int m=0; m<mtiles; m++) {
                const int t0   = m*MR;
                const int rows = seq-t0 < MR ? seq-t0 : MR;
                mm_pack_dispatch<MR>(out+(size_t)t0*N, x+(size_t)t0*K,
                                     Wp+(size_t)b*K*16, bias, rows, N, K, b*16);
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++) {
            const int t0   = m*MR;
            const int rows = seq-t0 < MR ? seq-t0 : MR;
            mm_pack_dispatch<MR>(out+(size_t)t0*N, x+(size_t)t0*K,
                                 Wp+(size_t)b*K*16, bias, rows, N, K, b*16);
        }
    }
}

} // namespace tcpu

#endif // TCPU_HAL_RVV
