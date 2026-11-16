/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// RISC-V Vector masked attention (SpacemiT K3 tuning: X100, VLEN 256).
//
// K is transposed once per call into kt = [n_kv][head_dim][seq_k] (keys
// contiguous). A tile of QT queries of one head then computes its score rows as
// sum_d q[d] * kt[d][:] with vfmacc.vf over a strip of keys - no horizontal
// reduction per (query, key) pair, which is what a dot-per-key loop costs on RVV
// (vfredusum is a long-latency op). Each kt strip load is shared by the QT rows.
// Softmax is vectorized per row (Cephes exp_ps, the tolerance class every SIMD
// backend ships; blocked keys flush to weight exactly 0). AV walks keys with the
// head_dim strip of each V row loaded once and FMA'd into all QT outputs; keys
// whose weight is zero in every row of the tile are skipped. The dense op comes
// from hal/common/attn.cpp (this path with a zero mask).

#include "../arch.h"
#if TCPU_HAL_RVV

#include "../simd.h"
#include "../../ops/lm_ops.h"
#include <cstddef>
#include <limits>
#include <vector>
using std::size_t;

namespace tcpu {

namespace {

constexpr int QT = 4;    // queries per register micro-tile
constexpr int QB = 16;   // queries per cache block: one pass over K^T / V serves QB/QT tiles
constexpr size_t KS = 128;   // keys per QK slice (head_dim x 128 floats of K^T, L1-resident)

// Score rows for `rows` (<= QT) queries against all keys of one kv head.
// sc is QT x seq_k. Applies scale and the mask; blocked keys become -inf.
inline void qk_tile(float* sc, const float* const* q, int rows, const float* kt,
                    int seq_k, int head_dim, float scale, const float* const* mrow,
                    size_t j0, size_t j1) {
    const float NINF  = -std::numeric_limits<float>::infinity();
    const float BLOCK = std::numeric_limits<float>::lowest();
    for (size_t j = j0; j < j1;) {
        const size_t vl = __riscv_vsetvl_e32m4(j1 - j);
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t a1 = a0, a2 = a0, a3 = a0;
        const float* kp = kt + j;
        for (int d = 0; d < head_dim; d++, kp += seq_k) {
            const vfloat32m4_t kv = __riscv_vle32_v_f32m4(kp, vl);
            a0 = __riscv_vfmacc_vf_f32m4(a0, q[0][d], kv, vl);
            if (rows > 1) a1 = __riscv_vfmacc_vf_f32m4(a1, q[1][d], kv, vl);
            if (rows > 2) a2 = __riscv_vfmacc_vf_f32m4(a2, q[2][d], kv, vl);
            if (rows > 3) a3 = __riscv_vfmacc_vf_f32m4(a3, q[3][d], kv, vl);
        }
        for (int r = 0; r < rows; r++) {
            const vfloat32m4_t a = r == 0 ? a0 : r == 1 ? a1 : r == 2 ? a2 : a3;
            vfloat32m4_t s = __riscv_vfmul_vf_f32m4(a, scale, vl);
            if (mrow[r]) {
                const vfloat32m4_t m = __riscv_vle32_v_f32m4(mrow[r] + j, vl);
                s = __riscv_vfadd_vv_f32m4(s, m, vl);
                const vbool8_t blk = __riscv_vmfeq_vf_f32m4_b8(m, BLOCK, vl);
                s = __riscv_vfmerge_vfm_f32m4(s, NINF, blk, vl);
            }
            __riscv_vse32_v_f32m4(sc + (size_t)r*seq_k + j, s, vl);
        }
        j += vl;
    }
}

// In-place softmax numerators of one score row; returns 1/sum. All-blocked rows
// become uniform weights, matching the scalar reference.
inline float softmax_row(float* sc, int seq_k) {
    const float NINF = -std::numeric_limits<float>::infinity();
    vfloat32m1_t mx = __riscv_vfmv_v_f_f32m1(NINF, 1);
    for (size_t i = 0, n = (size_t)seq_k; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        mx = __riscv_vfredmax_vs_f32m4_f32m1(__riscv_vle32_v_f32m4(sc + i, vl), mx, vl);
        i += vl;
    }
    const float maxs = __riscv_vfmv_f_s_f32m1_f32(mx);
    if (maxs == NINF) {
        for (int t = 0; t < seq_k; t++) sc[t] = 1.0f;
        return 1.0f/(float)seq_k;
    }
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (size_t i = 0, n = (size_t)seq_k; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        const vfloat32m4_t e = exp_ps(__riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(sc + i, vl),
                                                             maxs, vl), vl);
        __riscv_vse32_v_f32m4(sc + i, e, vl);
        acc = __riscv_vfredusum_vs_f32m4_f32m1(e, acc, vl);
        i += vl;
    }
    return 1.0f/__riscv_vfmv_f_s_f32m1_f32(acc);
}

// o[r][d0:d1] = sum_t w[r][t] * V[t][d0:d1] (weights already normalized). vs is
// the packed V strip for [d0, d1): seq_k rows of DS floats, contiguous.
inline void av_tile(float* const* o, const float* w, int rows, const float* vs, size_t DS,
                    int seq_k, size_t d0, size_t d1) {
    {
        const size_t d  = d0;
        const size_t vl = __riscv_vsetvl_e32m4(d1 - d0);
        vfloat32m4_t o0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t o1 = o0, o2 = o0, o3 = o0;
        const float* w1 = w + seq_k;
        const float* w2 = w + 2*(size_t)seq_k;
        const float* w3 = w + 3*(size_t)seq_k;
        for (int t = 0; t < seq_k; t++) {
            const float x0 = w[t];
            const float x1 = rows > 1 ? w1[t] : 0.0f;
            const float x2 = rows > 2 ? w2[t] : 0.0f;
            const float x3 = rows > 3 ? w3[t] : 0.0f;
            if (x0 == 0.0f && x1 == 0.0f && x2 == 0.0f && x3 == 0.0f) continue;
            const vfloat32m4_t v = __riscv_vle32_v_f32m4(vs + (size_t)t*DS, vl);
            o0 = __riscv_vfmacc_vf_f32m4(o0, x0, v, vl);
            if (rows > 1) o1 = __riscv_vfmacc_vf_f32m4(o1, x1, v, vl);
            if (rows > 2) o2 = __riscv_vfmacc_vf_f32m4(o2, x2, v, vl);
            if (rows > 3) o3 = __riscv_vfmacc_vf_f32m4(o3, x3, v, vl);
        }
        __riscv_vse32_v_f32m4(o[0] + d, o0, vl);
        if (rows > 1) __riscv_vse32_v_f32m4(o[1] + d, o1, vl);
        if (rows > 2) __riscv_vse32_v_f32m4(o[2] + d, o2, vl);
        if (rows > 3) __riscv_vse32_v_f32m4(o[3] + d, o3, vl);
    }
}

} // namespace

void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask, const float*) {
    const int group  = n_q/n_kv;

    // V packed per kv head into head_dim strips of DS floats, keys contiguous:
    // vp[kv][strip][t][DS]. Read in place, V's rows sit n_kv*head_dim floats
    // apart (2 KB at ACT's shape), so the AV walk over keys aliases a few L1
    // sets and every micro-tile re-fetches V from the cluster-shared L2 -
    // measured slower at 2 threads than at 1 on the K3.
    const size_t DS = __riscv_vsetvlmax_e32m4();
    const int nstrip = (int)(((size_t)head_dim+DS-1)/DS);
    static thread_local std::vector<float> kt_buf, vp_buf;
    const size_t kt_n = (size_t)n_kv*head_dim*seq_k;
    const size_t vp_n = (size_t)n_kv*nstrip*seq_k*DS;
    if (kt_buf.size() < kt_n) kt_buf.resize(kt_n);
    if (vp_buf.size() < vp_n) vp_buf.resize(vp_n);
    float* kt = kt_buf.data();
    float* vp = vp_buf.data();

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    // kt[kv][d][t] = K[t][kv][d]. Each thread owns whole kt rows: splitting by
    // key instead has every thread writing into the same lines (false sharing,
    // measured 2x off the 8-thread op time on the K3).
#if defined(_OPENMP)
    #pragma omp for schedule(static) collapse(2)
#endif
    for (int kv=0; kv<n_kv; kv++)
        for (int d=0; d<head_dim; d++) {
            const float* src = K+(size_t)kv*head_dim+d;
            float* dst = kt+((size_t)kv*head_dim+d)*seq_k;
            const ptrdiff_t stride = (ptrdiff_t)n_kv*head_dim*sizeof(float);
            for (size_t t = 0, n = (size_t)seq_k; t < n;) {
                const size_t vl = __riscv_vsetvl_e32m4(n - t);
                __riscv_vse32_v_f32m4(dst+t, __riscv_vlse32_v_f32m4(src+t*(size_t)n_kv*head_dim,
                                                                   stride, vl), vl);
                t += vl;
            }
        }

#if defined(_OPENMP)
    #pragma omp for schedule(static) collapse(2)
#endif
    for (int kv=0; kv<n_kv; kv++)
        for (int st=0; st<nstrip; st++) {
            const size_t d0 = (size_t)st*DS;
            const size_t w  = (size_t)head_dim-d0 < DS ? (size_t)head_dim-d0 : DS;
            float* dst = vp+((size_t)kv*nstrip+st)*seq_k*DS;
            for (int t=0; t<seq_k; t++) {
                const float* src = V+((size_t)t*n_kv+kv)*head_dim+d0;
                __riscv_vse32_v_f32m4(dst+(size_t)t*DS, __riscv_vle32_v_f32m4(src, w), w);
            }
        }

    // Blocked like the NEON dense path: QK walks key slices outermost so each
    // K^T slice serves every micro-tile of the block from L1; AV walks head_dim
    // strips outermost so each V column strip stays in L2 across the block.
    // Without it every 4-query tile re-streams the head's whole K^T and V panels
    // (~300 KB at ACT's seq 602), and the 8-thread op is bandwidth-bound.
    const int qblocks = (seq_q+QB-1)/QB;
    std::vector<float> sc((size_t)QB*seq_k);   // per-thread score block
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 1) collapse(2)
#endif
    for (int h=0; h<n_q; h++) {
        for (int blk=0; blk<qblocks; blk++) {
            const int kv  = h/group;
            const int b0  = blk*QB;
            const int nbq = seq_q-b0 < QB ? seq_q-b0 : QB;
            const int ntile = (nbq+QT-1)/QT;
            const float* q[QB];
            const float* mrow[QB];
            float* o[QB];
            for (int r=0; r<QB; r++) {
                const int t1 = b0 + (r < nbq ? r : 0);
                q[r]    = Q+((size_t)t1*n_q+h)*head_dim;
                mrow[r] = mask ? mask+(size_t)t1*seq_k : nullptr;
                o[r]    = out+((size_t)t1*n_q+h)*head_dim;
            }
            const float* ktp = kt+(size_t)kv*head_dim*seq_k;
            for (size_t j=0; j<(size_t)seq_k; j+=KS) {
                const size_t j1 = j+KS < (size_t)seq_k ? j+KS : (size_t)seq_k;
                for (int m=0; m<ntile; m++) {
                    const int rows = nbq-m*QT < QT ? nbq-m*QT : QT;
                    qk_tile(sc.data()+(size_t)m*QT*seq_k, q+m*QT, rows, ktp, seq_k, head_dim,
                            scale, mrow+m*QT, j, j1);
                }
            }
            for (int r=0; r<nbq; r++) {
                float* row = sc.data()+(size_t)r*seq_k;
                const float inv = softmax_row(row, seq_k);
                for (size_t i = 0, n = (size_t)seq_k; i < n;) {
                    const size_t vl = __riscv_vsetvl_e32m4(n - i);
                    __riscv_vse32_v_f32m4(row+i, __riscv_vfmul_vf_f32m4(
                                          __riscv_vle32_v_f32m4(row+i, vl), inv, vl), vl);
                    i += vl;
                }
            }
            for (int st=0; st<nstrip; st++) {
                const size_t d  = (size_t)st*DS;
                const size_t d1 = d+DS < (size_t)head_dim ? d+DS : (size_t)head_dim;
                const float* vs = vp+((size_t)kv*nstrip+st)*seq_k*DS;
                for (int m=0; m<ntile; m++) {
                    const int rows = nbq-m*QT < QT ? nbq-m*QT : QT;
                    av_tile(o+m*QT, sc.data()+(size_t)m*QT*seq_k, rows, vs, DS, seq_k, d, d1);
                }
            }
        }
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_RVV
