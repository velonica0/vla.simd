/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../arch.h"

// ---------------------------------------------------------------------------
// RISC-V Vector (RVV 1.0) path, tuned on the SpacemiT K3 (X100 cores, VLEN 256).
// Same helper surface as the AVX2 / NEON headers (simd_dot / simd_axpy / exp) so
// the shared op bodies take the SIMD path under one HAVE_SIMD guard. RVV is
// vector-length agnostic: loops strip-mine with vsetvl and need no scalar tail.
//
// exp_ps is the same Cephes polynomial and constants as exp256_ps / the NEON
// exp_ps, so the activations land in the same tolerance class on every backend.
// ---------------------------------------------------------------------------
#include <riscv_vector.h>
#include <cstddef>

namespace tcpu {

// Dot product: LMUL=4 (32 fp32 lanes at VLEN 256) with two independent
// accumulator groups to cover vfmacc latency, tail-undisturbed so the final
// partial strip leaves the inactive accumulator lanes intact.
static inline float simd_dot(const float* a, const float* b, int n) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    const size_t nn = (size_t)n;
    for (; i + 2*vlmax <= nn; i += 2*vlmax) {
        acc0 = __riscv_vfmacc_vv_f32m4(acc0, __riscv_vle32_v_f32m4(a+i, vlmax),
                                       __riscv_vle32_v_f32m4(b+i, vlmax), vlmax);
        acc1 = __riscv_vfmacc_vv_f32m4(acc1, __riscv_vle32_v_f32m4(a+i+vlmax, vlmax),
                                       __riscv_vle32_v_f32m4(b+i+vlmax, vlmax), vlmax);
    }
    while (i < nn) {
        const size_t vl = __riscv_vsetvl_e32m4(nn - i);
        acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, __riscv_vle32_v_f32m4(a+i, vl),
                                          __riscv_vle32_v_f32m4(b+i, vl), vl);
        i += vl;
    }
    acc0 = __riscv_vfadd_vv_f32m4(acc0, acc1, vlmax);
    const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, z, vlmax));
}

static inline void simd_axpy(float* o, float alpha, const float* v, int n) {
    for (size_t i = 0, nn = (size_t)n; i < nn;) {
        const size_t vl = __riscv_vsetvl_e32m4(nn - i);
        vfloat32m4_t ov = __riscv_vle32_v_f32m4(o+i, vl);
        ov = __riscv_vfmacc_vf_f32m4(ov, alpha, __riscv_vle32_v_f32m4(v+i, vl), vl);
        __riscv_vse32_v_f32m4(o+i, ov, vl);
        i += vl;
    }
}

// Cephes-style exp (~1-2 ulp), the RVV twin of exp256_ps. Lanes below -87 flush
// to exactly 0.0f (mirrors libm underflow), as on the other backends.
static inline vfloat32m4_t exp_ps(vfloat32m4_t x, size_t vl) {
    const vbool8_t under = __riscv_vmfle_vf_f32m4_b8(x, -87.0f, vl);
    x = __riscv_vfmin_vf_f32m4(x, 88.3762626647949f, vl);
    x = __riscv_vfmax_vf_f32m4(x, -88.3762626647950f, vl);

    // fx = floor(x*log2(e) + 0.5): round-down conversion, then back to float
    vfloat32m4_t fx = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    fx = __riscv_vfmacc_vf_f32m4(fx, 1.44269504088896341f, x, vl);
    const vint32m4_t fi = __riscv_vfcvt_x_f_v_i32m4_rm(fx, __RISCV_FRM_RDN, vl);
    fx = __riscv_vfcvt_f_x_v_f32m4(fi, vl);

    x = __riscv_vfnmsac_vf_f32m4(x, 0.693359375f, fx, vl);
    x = __riscv_vfnmsac_vf_f32m4(x, -2.12194440e-4f, fx, vl);

    vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(1.9875691500e-4f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, x, __riscv_vfmv_v_f_f32m4(1.3981999507e-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, x, __riscv_vfmv_v_f_f32m4(8.3334519073e-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, x, __riscv_vfmv_v_f_f32m4(4.1665795894e-2f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, x, __riscv_vfmv_v_f_f32m4(1.6666665459e-1f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, x, __riscv_vfmv_v_f_f32m4(5.0000001201e-1f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, __riscv_vfmul_vv_f32m4(x, x, vl),
                                __riscv_vfadd_vf_f32m4(x, 1.0f, vl), vl);

    const vint32m4_t n = __riscv_vadd_vx_i32m4(fi, 127, vl);
    const vfloat32m4_t pow2n = __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vsll_vx_i32m4(n, 23, vl));
    y = __riscv_vfmul_vv_f32m4(y, pow2n, vl);
    return __riscv_vfmerge_vfm_f32m4(y, 0.0f, under, vl);
}

} // namespace tcpu
