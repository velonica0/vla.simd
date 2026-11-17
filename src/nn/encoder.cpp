/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "encoder.h"
#include "../ops/lm_ops.h"
#include <cstdio>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

void Prof::tic() { if (on) t0 = now_ms(); }
void Prof::toc(double& acc) { if (on) acc += now_ms()-t0; }
void Prof::report() const {
    if (!on) return;
    double sum = assemble+mask+ln+qkv+attn+proj+mlp+res;
    std::printf("  [octo tf] assemble %6.1f  mask %6.1f  ln %6.1f  qkv %6.1f  attn %6.1f"
                "  proj %6.1f  mlp %6.1f  res %6.1f | other %6.1f | wall %6.1f ms\n",
                assemble, mask, ln, qkv, attn, proj, mlp, res, wall-sum, wall);
}

void EncoderLayer::forward(float* x, int total, const float* mask, Scratch& s, Prof* prof) const {
    if (s.h.size()  < (size_t)total*d  ) s.h.resize ((size_t)total*d);
    if (s.ff.size() < (size_t)total*mlp) s.ff.resize((size_t)total*mlp);

    // Per layer, not per backend: an int8 wo or w2 has no fused epilogue, and
    // taking it anyway costs an allocation and a second pass per call.
    const bool fuse_attn = attn.wo.add_ok();
    const bool fuse_mlp  = w2.add_ok();

    if (prof) prof->tic();
    layernorm(s.h.data(), x, ln1_s, ln1_b, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (fuse_attn) {
        // residual fused into the wo epilogue: x += proj (bit-identical add)
        attn.forward(x, s.h.data(), total, mask, -1, s, prof, true);
    } else {
        // proj lands back in h: x_norm (h) is consumed before the projection writes
        attn.forward(s.h.data(), s.h.data(), total, mask, -1, s, prof);

        if (prof) prof->tic();
        for (size_t i=0; i<(size_t)total*d; i++)
            x[i] += s.h[i];
        if (prof) prof->toc(prof->res);
    }

    if (prof) prof->tic();
    layernorm(s.h.data(), x, ln2_s, ln2_b, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (prof) prof->tic();
    if (gelu == Gelu::Tanh) {
        w1.forward_gelu(s.ff.data(), s.h.data(), total);
    } else {
        // no fused erf epilogue: plain GEMM then the activation in place
        w1.forward(s.ff.data(), s.h.data(), total);
        gelu_erf(s.ff.data(), total*mlp);
    }
    if (fuse_mlp) {
        w2.forward_add(x, s.ff.data(), total);
    } else {
        w2.forward(s.h.data(), s.ff.data(), total);
    }
    if (prof) prof->toc(prof->mlp);

    if (!fuse_mlp) {
        if (prof) prof->tic();
        for (size_t i=0; i<(size_t)total*d; i++)
            x[i] += s.h[i];
        if (prof) prof->toc(prof->res);
    }
}

bool EncoderLayer::prepare_f16() {
    ln_tiles.clear();
    const bool ok = ime_f16_enabled() && gelu == Gelu::Tanh && d%16 == 0 && mlp%16 == 0 &&
                    attn.head_dim%16 == 0 && attn.head_dim <= 256 && attn.heads*attn.head_dim == d &&
                    attn.wq.f16_ok() && attn.wk.f16_ok() && attn.wv.f16_ok() && attn.wo.f16_ok() &&
                    w1.f16_ok() && w2.f16_ok() && ln1_s && ln1_b && ln2_s && ln2_b;
    if (!ok) return false;
    const size_t n = (size_t)d*8;
    ln_tiles.resize(4*n);
    ime_f16_bias_tiles(ln_tiles.data(), ln1_s, d);
    ime_f16_bias_tiles(ln_tiles.data() + n, ln1_b, d);
    ime_f16_bias_tiles(ln_tiles.data() + 2*n, ln2_s, d);
    ime_f16_bias_tiles(ln_tiles.data() + 3*n, ln2_b, d);
    return true;
}

void EncoderLayer::forward_f16(uint16_t* x16, int total, Scratch& s, Prof* prof) const {
    const size_t nd = ime_f16_size(total, d), nm = ime_f16_size(total, mlp), n = (size_t)d*8;
    if (s.h16.size() < nd) s.h16.resize(nd);
    if (s.q16.size() < nd) s.q16.resize(nd);
    if (s.k16.size() < nd) s.k16.resize(nd);
    if (s.v16.size() < nd) s.v16.resize(nd);
    if (s.a16.size() < nd) s.a16.resize(nd);
    if (s.ff16.size() < nm) s.ff16.resize(nm);
    const float* lt = ln_tiles.data();

    if (prof) prof->tic();
    ime_f16_layernorm(s.h16.data(), x16, lt, lt + n, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (prof) prof->tic();
    const ImeF16Job qkv[3] = {attn.wq.f16_job(s.q16.data()), attn.wk.f16_job(s.k16.data()),
                              attn.wv.f16_job(s.v16.data())};
    ime_f16_linear(qkv, 3, s.h16.data(), total, d);
    if (prof) prof->toc(prof->qkv);

    if (prof) prof->tic();
    ime_f16_attention(s.a16.data(), s.q16.data(), s.k16.data(), s.v16.data(), total,
                      attn.heads, attn.head_dim, attn.scale);
    if (prof) prof->toc(prof->attn);

    if (prof) prof->tic();
    const ImeF16Job proj = attn.wo.f16_job(s.h16.data());
    ime_f16_linear(&proj, 1, s.a16.data(), total, d);
    if (prof) prof->toc(prof->proj);

    if (prof) prof->tic();
    ime_f16_add(x16, s.h16.data(), total, d);
    if (prof) prof->toc(prof->res);

    if (prof) prof->tic();
    ime_f16_layernorm(s.h16.data(), x16, lt + 2*n, lt + 3*n, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (prof) prof->tic();
    const ImeF16Job fc1 = w1.f16_job(s.ff16.data());
    ime_f16_linear(&fc1, 1, s.h16.data(), total, d);
    ime_f16_gelu_tanh(s.ff16.data(), total, mlp);
    const ImeF16Job fc2 = w2.f16_job(s.h16.data());
    ime_f16_linear(&fc2, 1, s.ff16.data(), total, mlp);
    if (prof) prof->toc(prof->mlp);

    if (prof) prof->tic();
    ime_f16_add(x16, s.h16.data(), total, d);
    if (prof) prof->toc(prof->res);
}

void EncoderLayer::forward_last_row(float* x, int ro, int total, const float* mask, Scratch& s) const {
    if (s.h.size()  < (size_t)total*d) s.h.resize ((size_t)total*d);
    if (s.ff.size() < (size_t)mlp    ) s.ff.resize((size_t)mlp);

    const bool fuse_attn = attn.wo.add_ok();
    const bool fuse_mlp  = w2.add_ok();
    float* xr = x+(size_t)ro*d;

    layernorm(s.h.data(), x, ln1_s, ln1_b, total, d, ln_eps);
    if (fuse_attn) {
        attn.forward(xr, s.h.data(), total, mask, ro, s, nullptr, true);
    } else {
        attn.forward(s.h.data(), s.h.data(), total, mask, ro, s, nullptr);

        for (int i=0; i<d; i++)
            xr[i] += s.h[i];
    }

    layernorm(s.h.data(), xr, ln2_s, ln2_b, 1, d, ln_eps);
    if (gelu == Gelu::Tanh) {
        w1.forward_gelu(s.ff.data(), s.h.data(), 1);
    } else {
        w1.forward(s.ff.data(), s.h.data(), 1);
        gelu_erf(s.ff.data(), mlp);
    }
    if (fuse_mlp) {
        w2.forward_add(xr, s.ff.data(), 1);
    } else {
        w2.forward(s.h.data(), s.ff.data(), 1);

        for (int i=0; i<d; i++)
            xr[i] += s.h[i];
    }
}

} // namespace nn
} // namespace tcpu
