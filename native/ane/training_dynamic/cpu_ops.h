// cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, embedding
#pragma once
#include "config.h"

static float *g_rms_tmp = NULL;

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)jishui_malloc(S*4);
    float *ss = (float*)jishui_calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)jishui_malloc(S*4);
    float *ss = (float*)jishui_calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)jishui_malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)jishui_calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot);
}

// LayerNorm reference path used by Jishui-200M.  ANE kernels deliberately
// receive already-normalized activations, so keeping this on the host also
// makes the normalization semantics explicit and easy to compare with MLX.
static void layernorm_forward(float *out, const float *x, const float *g,
                              const float *b, int d, int S) {
    const float invd = 1.0f / (float)d;
    for (int t = 0; t < S; t++) {
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += x[i*S + t];
        mean *= invd;
        float var = 0.0f;
        for (int i = 0; i < d; i++) {
            float z = x[i*S + t] - mean;
            var += z * z;
        }
        float inv = 1.0f / sqrtf(var * invd + NORM_EPS);
        for (int i = 0; i < d; i++) {
            float xhat = (x[i*S + t] - mean) * inv;
            out[i*S + t] = xhat * g[i] + b[i];
        }
    }
}

static void layernorm_backward(float *dx, float *dg, float *db,
                               const float *dy, const float *x,
                               const float *g, int d, int S) {
    const float invd = 1.0f / (float)d;
    for (int t = 0; t < S; t++) {
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += x[i*S + t];
        mean *= invd;
        float var = 0.0f;
        for (int i = 0; i < d; i++) {
            float z = x[i*S + t] - mean;
            var += z * z;
        }
        float inv = 1.0f / sqrtf(var * invd + NORM_EPS);
        float sum_dyg = 0.0f;
        float sum_dygx = 0.0f;
        for (int i = 0; i < d; i++) {
            float xhat = (x[i*S + t] - mean) * inv;
            float dyi = dy[i*S + t];
            float dyg = dyi * g[i];
            sum_dyg += dyg;
            sum_dygx += dyg * xhat;
            dg[i] += dyi * xhat;
            db[i] += dyi;
        }
        for (int i = 0; i < d; i++) {
            float xhat = (x[i*S + t] - mean) * inv;
            float dyg = dy[i*S + t] * g[i];
            dx[i*S + t] = inv * invd *
                ((float)d * dyg - sum_dyg - xhat * sum_dygx);
        }
    }
}

static void norm_forward(float *out, const float *x, const float *g,
                         const float *b, int d, int S) {
#if USE_LAYER_NORM
    layernorm_forward(out, x, g, b, d, S);
#else
    (void)b;
    rmsnorm(out, x, g, d, S);
#endif
}

static void norm_backward(float *dx, float *dg, float *db, const float *dy,
                          const float *x, const float *g, int d, int S) {
#if USE_LAYER_NORM
    layernorm_backward(dx, dg, db, dy, x, g, d, S);
#else
    (void)db;
    rmsnorm_bwd(dx, dg, dy, x, g, d, S);
#endif
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
    }
}

// Cross-entropy loss for token-major logits [S,V].  Keeping each vocabulary
// row contiguous avoids the cache-hostile gather/scatter required by [V,S].
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        const float *row = logits + (size_t)t * V;
        float *drow = dlogits + (size_t)t * V;
        memcpy(drow, row, (size_t)V * sizeof(float));
        float maxv; vDSP_maxv(drow, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(drow, 1, &neg_max, drow, 1, (vDSP_Length)V);
        int n = V; vvexpf(drow, drow, &n);
        float sum; vDSP_sve(drow, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(drow, 1, &inv_sum, drow, 1, (vDSP_Length)V);
        int tgt = targets[t];
        total_loss -= logf(drow[tgt] + 1e-10f);
        drow[tgt] -= 1.0f;
        vDSP_vsmul(drow, 1, &invS, drow, 1, (vDSP_Length)V);
    }
    return total_loss / S;
}

// Vocab compaction: build mapping from full 32K vocab to compact vocab
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_identity(int vocab) {
    VocabMap vm;
    vm.compact_vocab = vocab;
    vm.full_to_compact = (int*)jishui_malloc((size_t)vocab * sizeof(int));
    vm.compact_to_full = (int*)jishui_malloc((size_t)vocab * sizeof(int));
    for (int v = 0; v < vocab; v++) {
        vm.full_to_compact[v] = v;
        vm.compact_to_full[v] = v;
    }
    return vm;
}

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)jishui_malloc((size_t)full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    // Scan for used tokens
    for (size_t i = 0; i < n_tokens; i++) {
        vm.full_to_compact[data[i]] = 0;  // mark as used
    }
    // Assign compact IDs
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)jishui_malloc((size_t)cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

// Create compact embedding from full embedding
static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)jishui_malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

// Scatter compact embed gradients back to full embed
static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        for (int d = 0; d < dim; d++)
            full_gembed[fv*dim + d] += compact_gembed[c*dim + d];
    }
}

// Update full embed from compact embed (after adam)
static void vocab_update_full(float *full_embed, const float *compact_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(full_embed + vm->compact_to_full[c]*dim, compact_embed + c*dim, dim*4);
}

static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

// RoPE backward (in-place): inverse rotation on dQ/dK gradients
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
            for (int p = 0; p < seq; p++) {
                float theta = p * freq;
                float cos_t = cosf(theta), sin_t = sinf(theta);
#if ROPE_TRADITIONAL
                int idx0 = (h * hd + 2 * i) * seq + p;
                int idx1 = (h * hd + 2 * i + 1) * seq + p;
                float v0 = dx[idx0], v1 = dx[idx1];
                dx[idx0] = v0 * cos_t + v1 * sin_t;
                dx[idx1] = -v0 * sin_t + v1 * cos_t;
#else
                int idx0 = (h * hd + i) * seq + p;
                int idx1 = (h * hd + hd/2 + i) * seq + p;
                float v0 = dx[idx0], v1 = dx[idx1];
                dx[idx0] = v0 * cos_t + v1 * sin_t;
                dx[idx1] = -v0 * sin_t + v1 * cos_t;
#endif
            }
        }
    }
}
