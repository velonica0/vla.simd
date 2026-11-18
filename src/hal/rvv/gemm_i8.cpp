/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// SpacemiT IME integer matrix kernel (vmadot) for the W8A8 GEMM on the K3.
//
// vmadot vd, vs1, vs2 (vtype e8/m1, vl = M*KT): vd, an MxM int32 tile held
// row-major in a register pair, accumulates vs1 * vs2^T, where vs1 holds M rows
// of KT int8 activations and vs2 M rows of KT int8 weights. The tile follows
// the vector length: 4x4 with KT = 8 on the X100 cores (VLEN 256, 128
// multiply-adds per instruction) and 8x8 with KT = 16 on the A100 cores (VLEN
// 1024, 1024 per instruction). One micro-kernel serves both, since only vl
// differs; the two clusters differ only in how the operands are loaded (wide
// loads on the A100 cores, one register at a time on the X100 cores). This is
// the only file built with _xsmtvdot.
//
// pack_weights_i8 appends a second copy of the quantized weights in tile order,
// [N/(2M)][KS][2][M][KT] with KS = ceil(K/KT), after the row-sum table of the
// portable layout (i8_ime_bytes, i8_ime_pack). Activations are packed per call
// into [ceil(seq/(4M))][KS][4][M][KT]. Accumulation is int32 and the epilogue
// is the reference's fma(acc*wscale, ascale, bias), so results equal
// dense_linear_i8_ref bit for bit.

#include "../arch.h"
#if TCPU_HAL_RVV && TCPU_RVV_IME_I8

#include "../common/env.h"
#include "../../ops/quant_ops.h"
#include <riscv_vector.h>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
using std::size_t;

namespace tcpu {

namespace {

// 4 activation tiles x 2 weight tiles over ks k-steps; c receives the 8 int32
// tiles as [4][2][M*M], and with init != 0 they accumulate onto what c holds. a: [ks][4][vl1] int8, b: [ks][2][vl1] int8, vl1 = M*KT.
// Two k-steps per iteration, so the 8 activation tiles come from one e8/m8 load
// and the 4 weight tiles from one e8/m4 load: the loop is bound by its loads.
void i8_tile(int32_t* c, const int8_t* a, const int8_t* b, size_t ks, size_t vl1, size_t vlc, size_t init) {
    const size_t vl2 = 2*vl1, vl4 = 4*vl1, vl8 = 8*vl1, odd = ks & 1, st = vlc*sizeof(int32_t);
    size_t n2 = ks/2;
    __asm__ volatile(
        "vsetvli  zero, %[vlc], e32, m2, ta, ma \n\t"
        "beqz %[init], 8f \n\t"
        "mv t1, %[c] \n\t"
        "vle32.v  v16, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v18, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v20, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v22, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v24, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v26, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v28, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v30, (t1) \n\t"
        "j 9f \n\t"
        "8: \n\t"
        "vmv.v.i  v16, 0 \n\t" "vmv.v.i  v18, 0 \n\t" "vmv.v.i  v20, 0 \n\t" "vmv.v.i  v22, 0 \n\t"
        "vmv.v.i  v24, 0 \n\t" "vmv.v.i  v26, 0 \n\t" "vmv.v.i  v28, 0 \n\t" "vmv.v.i  v30, 0 \n\t"
        "9: \n\t"
        "beqz %[n2], 2f \n\t"
        "1: \n\t"
        "vsetvli  zero, %[vl8], e8, m8, ta, ma \n\t"
        "vle8.v   v0, (%[a]) \n\t" "add %[a], %[a], %[vl8] \n\t"
        "vsetvli  zero, %[vl4], e8, m4, ta, ma \n\t"
        "vle8.v   v8, (%[b]) \n\t" "add %[b], %[b], %[vl4] \n\t"
        "vsetvli  zero, %[vl1], e8, m1, ta, ma \n\t"
        "vmadot v16, v0, v8 \n\t"  "vmadot v18, v0, v9 \n\t"
        "vmadot v20, v1, v8 \n\t"  "vmadot v22, v1, v9 \n\t"
        "vmadot v24, v2, v8 \n\t"  "vmadot v26, v2, v9 \n\t"
        "vmadot v28, v3, v8 \n\t"  "vmadot v30, v3, v9 \n\t"
        "vmadot v16, v4, v10 \n\t" "vmadot v18, v4, v11 \n\t"
        "vmadot v20, v5, v10 \n\t" "vmadot v22, v5, v11 \n\t"
        "vmadot v24, v6, v10 \n\t" "vmadot v26, v6, v11 \n\t"
        "vmadot v28, v7, v10 \n\t" "vmadot v30, v7, v11 \n\t"
        "addi %[n2], %[n2], -1 \n\t"
        "bnez %[n2], 1b \n\t"
        "2: \n\t"
        "beqz %[odd], 3f \n\t"
        "vsetvli  zero, %[vl4], e8, m4, ta, ma \n\t"
        "vle8.v   v0, (%[a]) \n\t"
        "vsetvli  zero, %[vl2], e8, m2, ta, ma \n\t"
        "vle8.v   v8, (%[b]) \n\t"
        "vsetvli  zero, %[vl1], e8, m1, ta, ma \n\t"
        "vmadot v16, v0, v8 \n\t"  "vmadot v18, v0, v9 \n\t"
        "vmadot v20, v1, v8 \n\t"  "vmadot v22, v1, v9 \n\t"
        "vmadot v24, v2, v8 \n\t"  "vmadot v26, v2, v9 \n\t"
        "vmadot v28, v3, v8 \n\t"  "vmadot v30, v3, v9 \n\t"
        "3: \n\t"
        "vsetvli  zero, %[vlc], e32, m2, ta, ma \n\t"
        "vse32.v  v16, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v18, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v20, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v22, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v24, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v26, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v28, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v30, (%[c]) \n\t"
        : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [n2] "+r"(n2)
        : [vl1] "r"(vl1), [vl2] "r"(vl2), [vl4] "r"(vl4), [vl8] "r"(vl8), [vlc] "r"(vlc),
          [odd] "r"(odd), [st] "r"(st), [init] "r"(init)
        : "cc", "memory", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31");
}

// The same tile with one e8/m1 load per operand tile and no vsetvli in the loop.
// The X100 cores run this about 20% faster than the wide loads (185 against 153
// G multiply-adds/s per core); the A100 cores run it three times slower.
void i8_tile_narrow(int32_t* c, const int8_t* a, const int8_t* b, size_t ks, size_t vl1, size_t vlc, size_t init) {
    const size_t st = vlc*sizeof(int32_t);
    __asm__ volatile(
        "vsetvli  zero, %[vlc], e32, m2, ta, ma \n\t"
        "beqz %[init], 8f \n\t"
        "mv t1, %[c] \n\t"
        "vle32.v  v16, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v18, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v20, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v22, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v24, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v26, (t1) \n\t" "add t1, t1, %[st] \n\t"
        "vle32.v  v28, (t1) \n\t" "add t1, t1, %[st] \n\t" "vle32.v  v30, (t1) \n\t"
        "j 9f \n\t"
        "8: \n\t"
        "vmv.v.i  v16, 0 \n\t" "vmv.v.i  v18, 0 \n\t" "vmv.v.i  v20, 0 \n\t" "vmv.v.i  v22, 0 \n\t"
        "vmv.v.i  v24, 0 \n\t" "vmv.v.i  v26, 0 \n\t" "vmv.v.i  v28, 0 \n\t" "vmv.v.i  v30, 0 \n\t"
        "9: \n\t"
        "vsetvli  zero, %[vl1], e8, m1, ta, ma \n\t"
        "1: \n\t"
        "vle8.v v8, (%[b]) \n\t" "add %[b], %[b], %[vl1] \n\t"
        "vle8.v v9, (%[b]) \n\t" "add %[b], %[b], %[vl1] \n\t"
        "vle8.v v0, (%[a]) \n\t" "add %[a], %[a], %[vl1] \n\t"
        "vle8.v v1, (%[a]) \n\t" "add %[a], %[a], %[vl1] \n\t"
        "vle8.v v2, (%[a]) \n\t" "add %[a], %[a], %[vl1] \n\t"
        "vle8.v v3, (%[a]) \n\t" "add %[a], %[a], %[vl1] \n\t"
        "vmadot v16, v0, v8 \n\t"  "vmadot v18, v0, v9 \n\t"
        "vmadot v20, v1, v8 \n\t"  "vmadot v22, v1, v9 \n\t"
        "vmadot v24, v2, v8 \n\t"  "vmadot v26, v2, v9 \n\t"
        "vmadot v28, v3, v8 \n\t"  "vmadot v30, v3, v9 \n\t"
        "addi %[ks], %[ks], -1 \n\t"
        "bnez %[ks], 1b \n\t"
        "vsetvli  zero, %[vlc], e32, m2, ta, ma \n\t"
        "vse32.v  v16, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v18, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v20, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v22, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v24, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v26, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v28, (%[c]) \n\t" "add %[c], %[c], %[st] \n\t"
        "vse32.v  v30, (%[c]) \n\t"
        : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [ks] "+r"(ks)
        : [vl1] "r"(vl1), [vlc] "r"(vlc), [st] "r"(st), [init] "r"(init)
        : "cc", "memory", "t1", "v0", "v1", "v2", "v3", "v8", "v9",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31");
}

inline void i8_tile_for(int M, int32_t* c, const int8_t* a, const int8_t* b, size_t ks, size_t vl1, size_t vlc,
                        size_t init = 0) {
    if (M == 4) i8_tile_narrow(c, a, b, ks, vl1, vlc, init);
    else i8_tile(c, a, b, ks, vl1, vlc, init);
}

sigjmp_buf probe_jmp;
void on_sigill(int) { siglongjmp(probe_jmp, 1); }

// Tile rows M (4 or 8) when every thread has the same supported vector length
// and vmadot gives the right answer there; 0 otherwise.
int detect() {
    if (!hal::env::ime_i8()) return 0;
    const size_t vlen = __riscv_vlenb()*8;
    bool same = true;
#if defined(_OPENMP)
    #pragma omp parallel reduction(&& : same)
    same = same && __riscv_vlenb()*8 == vlen;
#endif
    const int M = vlen == 1024 ? 8 : vlen == 256 ? 4 : 0;
    if (!same || !M) return 0;
    const size_t vl1 = (size_t)M*2*M, vlc = (size_t)M*M;
    // 3 k-steps: the paired loop and the odd tail both run
    std::vector<int8_t> a(3*4*vl1), b(3*2*vl1);
    std::vector<int32_t> c(8*vlc);
    for (size_t i=0; i<a.size(); i++) a[i] = (int8_t)((i*37 + 11) % 255 - 127);
    for (size_t i=0; i<b.size(); i++) b[i] = (int8_t)((i*53 + 5) % 255 - 127);
    struct sigaction sa = {}, old = {};
    sa.sa_handler = on_sigill;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, &old);
    volatile bool ran = false;
    if (sigsetjmp(probe_jmp, 1) == 0) {
        i8_tile_for(M, c.data(), a.data(), b.data(), 3, vl1, vlc);
        ran = true;
    }
    sigaction(SIGILL, &old, nullptr);
    bool ok = ran;
    const int KT = 2*M;
    for (int t=0; t<4 && ok; t++) for (int h=0; h<2 && ok; h++)
        for (int i=0; i<M && ok; i++) for (int j=0; j<M && ok; j++) {
            int32_t s = 0;
            for (int ks=0; ks<3; ks++) for (int k=0; k<KT; k++)
                s += (int32_t)a[((size_t)ks*4 + t)*vl1 + i*KT + k]*(int32_t)b[((size_t)ks*2 + h)*vl1 + j*KT + k];
            ok = c[((size_t)t*2 + h)*vlc + i*M + j] == s;
        }
    if (hal::env::ime_i8_verbose() || !ok)
        std::fprintf(stderr, "vla_simd: IME int8 GEMM %s (vmadot %dx%dx%d, VLEN %zu)\n",
                     ok ? "on" : ran ? "off (vmadot gave a wrong result)" : "off (vmadot raised SIGILL)",
                     M, M, KT, vlen);
    return ok ? M : 0;
}

int tile_m() {
    static const int v = detect();
    return v;
}

inline size_t ks_of(int K, int KT) { return (size_t)(K + KT - 1)/KT; }

// xq [rows][Kp4] -> one block of 4 activation tiles per k-step: [KS][4][M][KT].
// Rows past `rows` and k past Kp4 are zero.
void pack_block(int8_t* d, const int8_t* xq, int rows, int Kp4, int M, int KT, size_t KS) {
    const size_t step = (size_t)4*M*KT;
    const int full = Kp4/KT, rem = Kp4 - full*KT;
    if (rows < 4*M || rem) std::memset(d, 0, KS*step);
    for (int r=0; r<rows; r++) {
        const int8_t* s = xq + (size_t)r*Kp4;
        int8_t* dr = d + (size_t)r*KT;   // (t, i) = (r / M, r % M) is r*KT into a k-step
        if (KT == 16) for (int ks=0; ks<full; ks++) std::memcpy(dr + ks*step, s + (size_t)ks*16, 16);
        else          for (int ks=0; ks<full; ks++) std::memcpy(dr + ks*step, s + (size_t)ks*8, 8);
        if (rem) std::memcpy(dr + full*step, s + (size_t)full*KT, rem);
    }
}

} // namespace

bool int8_gemm_available() { return tile_m() != 0; }

size_t i8_ime_bytes(int N, int K) {
    // room for either tile size: KS*KT <= K rounded up to 16
    return (size_t)N*(((size_t)K + 15) & ~(size_t)15);
}

// Wq: the portable [N/16][Kp/4][16][4] panels. dst: [N/(2M)][KS][2][M][KT].
void i8_ime_pack(int8_t* dst, const int8_t* Wq, int N, int K) {
    const int M = tile_m();
    if (!M) return;
    const int KT = 2*M, KG = i8_kpad(K)/4;
    const size_t KS = ks_of(K, KT);
    std::memset(dst, 0, i8_ime_bytes(N, K));
    for (int n=0; n<N; n++) {
        const int b = n/16, j = n%16;
        const int nb = n/(2*M), h = (n/M) & 1, r = n%M;
        for (int k=0; k<K; k++)
            dst[((nb*KS + k/KT)*2 + h)*(size_t)M*KT + r*KT + k%KT] =
                Wq[((size_t)(b*KG + k/4)*16 + j)*4 + (k%4)];
    }
}

void dense_linear_i8_pre(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K) {
    const int M = tile_m();
    if (!M) { dense_linear_i8_ref(out, xq, ascale, Wq, wscale, bias, seq, N, K); return; }
    const int KT = 2*M, Kp4 = i8_kpad(K), RB = 4*M;
    const size_t KS = ks_of(K, KT), vl1 = (size_t)M*KT, vlc = (size_t)M*M;
    const int MB = (seq + RB - 1)/RB, NB = N/(2*M);
    // tile-order weights sit after the portable panels and their row sums
    const int8_t* Wt = Wq + (size_t)N*Kp4 + (size_t)N*sizeof(int32_t);
    const size_t astep = KS*4*vl1, wstep = KS*2*vl1;

    static thread_local std::vector<int8_t> at;
    if (at.size() < (size_t)MB*astep) at.resize((size_t)MB*astep);
    int8_t* A = at.data();
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*K > (size_t)hal::env::omp_min())
#endif
    for (int mb=0; mb<MB; mb++)
        pack_block(A + (size_t)mb*astep, xq + (size_t)mb*RB*Kp4, seq - mb*RB < RB ? seq - mb*RB : RB,
                   Kp4, M, KT, KS);

    // element e = i*M + j of a tile: i is the token row, j the output column
    uint32_t idx_j[64], idx_i[64];
    for (int e=0; e<M*M; e++) { idx_j[e] = (uint32_t)(e % M); idx_i[e] = (uint32_t)(e / M); }

    // Blocked like a BLAS GEMM, because the kernel reads a byte of operands per
    // five multiply-adds and runs several times slower once those bytes come
    // from L2 instead of L1 (64 KB on both clusters). A cell is a group of GR
    // activation blocks (64 rows) and a group of weight panels; inside it K is
    // cut into chunks of about 512 bytes, so the group's activation slice (32
    // KB) stays in L1 while each panel's slice (4 or 8 KB) sweeps it. With more
    // than one chunk the int32 tiles wait in a per-thread buffer between chunks.
    const int GR = 64/RB;                                     // activation blocks per row group
    const int NG = (MB + GR - 1)/GR;
    const size_t kcs_max = (size_t)512/KT;                    // k-steps per chunk
    const size_t nchunk = (KS + kcs_max - 1)/kcs_max;
    const size_t kcs = ((KS + nchunk - 1)/nchunk + 1) & ~(size_t)1;
    int nt = 1;
#if defined(_OPENMP)
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    // panels per cell: about four cells per thread, and a weight slice of at most 256 KB
    long PG = ((long)NB*NG + 4L*nt - 1)/(4L*nt);
    const long pg_max = (long)((size_t)256*1024/(2*vl1*kcs));
    if (PG > pg_max) PG = pg_max;
    if (PG > NB) PG = NB;
    if (PG < 1) PG = 1;
    const int NPG = (int)((NB + PG - 1)/PG);
    const size_t ctile = 8*vlc;                               // int32 per kernel call

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) collapse(2) if((long)NG*NPG > 1)
#endif
    for (int g=0; g<NG; g++) {
        for (int pg=0; pg<NPG; pg++) {
            const int mb0 = g*GR, nmb = MB - mb0 < GR ? MB - mb0 : GR;
            const int nb0 = (int)(pg*PG), nnb = NB - nb0 < PG ? NB - nb0 : (int)PG;
            static thread_local std::vector<int32_t> cbuf;
            int32_t c1[8*64];
            int32_t* C = c1;
            if (nchunk > 1) {
                if (cbuf.size() < (size_t)nnb*nmb*ctile) cbuf.resize((size_t)PG*GR*ctile);
                C = cbuf.data();
            }
            float f[8*64];
            for (size_t ch=0; ch<nchunk; ch++) {
                const size_t k0 = ch*kcs, kn = KS - k0 < kcs ? KS - k0 : kcs;
                const bool last = ch + 1 == nchunk;
                for (int nbi=0; nbi<nnb; nbi++) {
                    const int nb = nb0 + nbi, n0 = nb*2*M;
                    const int8_t* w = Wt + (size_t)nb*wstep + k0*2*vl1;
                    for (int mbi=0; mbi<nmb; mbi++) {
                        const int mb = mb0 + mbi;
                        int32_t* c = nchunk > 1 ? C + ((size_t)nbi*nmb + mbi)*ctile : C;
                        i8_tile_for(M, c, A + (size_t)mb*astep + k0*4*vl1, w, kn, vl1, vlc, ch != 0);
                        if (!last) continue;

                        const int t0 = mb*RB, rows = seq - t0 < RB ? seq - t0 : RB;
                        float asp[32];
                        for (int i=0; i<RB; i++) asp[i] = i < rows ? ascale[t0+i] : 0.0f;
                        const vuint32m2_t vj = __riscv_vle32_v_u32m2(idx_j, vlc);
                        const vuint32m2_t vi = __riscv_vle32_v_u32m2(idx_i, vlc);
                        const vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vlc);
                        const vfloat32m2_t ws0 = __riscv_vrgather_vv_f32m2(__riscv_vle32_v_f32m2(wscale + n0, M), vj, vlc);
                        const vfloat32m2_t ws1 = __riscv_vrgather_vv_f32m2(__riscv_vle32_v_f32m2(wscale + n0 + M, M), vj, vlc);
                        const vfloat32m2_t bs0 = bias ? __riscv_vrgather_vv_f32m2(__riscv_vle32_v_f32m2(bias + n0, M), vj, vlc) : zero;
                        const vfloat32m2_t bs1 = bias ? __riscv_vrgather_vv_f32m2(__riscv_vle32_v_f32m2(bias + n0 + M, M), vj, vlc) : zero;
                        for (int t=0; t<4; t++) {
                            const vfloat32m2_t sa = __riscv_vrgather_vv_f32m2(__riscv_vle32_v_f32m2(asp + t*M, M), vi, vlc);
                            const vfloat32m2_t v0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(c + ((size_t)t*2    )*vlc, vlc), vlc);
                            const vfloat32m2_t v1 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(c + ((size_t)t*2 + 1)*vlc, vlc), vlc);
                            // fma(acc*wscale, ascale, bias), as dense_linear_i8_ref
                            __riscv_vse32_v_f32m2(f + ((size_t)t*2    )*vlc,
                                __riscv_vfmadd_vv_f32m2(__riscv_vfmul_vv_f32m2(v0, ws0, vlc), sa, bs0, vlc), vlc);
                            __riscv_vse32_v_f32m2(f + ((size_t)t*2 + 1)*vlc,
                                __riscv_vfmadd_vv_f32m2(__riscv_vfmul_vv_f32m2(v1, ws1, vlc), sa, bs1, vlc), vlc);
                        }
                        for (int r=0; r<rows; r++) {
                            const int t = r/M, i = r%M;
                            float* o = out + (size_t)(t0 + r)*N + n0;
                            const float* f0 = f + ((size_t)t*2    )*vlc + (size_t)i*M;
                            const float* f1 = f + ((size_t)t*2 + 1)*vlc + (size_t)i*M;
                            // constant sizes, so the copies are inlined
                            if (M == 8) { std::memcpy(o, f0, 32); std::memcpy(o + 8, f1, 32); }
                            else        { std::memcpy(o, f0, 16); std::memcpy(o + 4, f1, 16); }
                        }
                    }
                }
            }
        }
    }
}

} // namespace tcpu

#endif // TCPU_HAL_RVV && TCPU_RVV_IME_I8
