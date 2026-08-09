// config.h — Model-agnostic structs, derived sizes, ANE init
// Model-specific dims come from models/*.h, selected via -DMODEL_HEADER
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arm_neon.h>

// Include selected model config
// MODEL_HEADER is set by Makefile via -include models/xxx.h
#ifndef MODEL_NAME
#error "No model selected. Build with: make MODEL=qwen3_06b (or stories110m)"
#endif

#ifndef USE_LAYER_NORM
#define USE_LAYER_NORM 0
#endif
#ifndef RES_ALPHA
#define RES_ALPHA (1.0f / sqrtf(2.0f * NLAYERS))
#endif
#ifndef NORM_EPS
#define NORM_EPS 1.0e-5f
#endif
#ifndef ADAM_EPS
#define ADAM_EPS 1.0e-8f
#endif
#ifndef ROPE_TRADITIONAL
#define ROPE_TRADITIONAL 1
#endif

// The frozen NPY shards, ANE index and native record loader all use uint16
// token IDs.  Refuse larger model vocabularies instead of silently truncating
// IDs (the Qwen GQA header is retained as a future MIL reference only).
#if VOCAB > 65536
#error "Native ANE data path supports vocabularies up to 65536 uint16 IDs"
#endif

// Derived weight sizes per layer (GQA-aware)
#define WQ_SZ (Q_DIM*DIM)
#define WK_SZ (KV_DIM*DIM)
#define WV_SZ (KV_DIM*DIM)
#define WO_SZ (DIM*Q_DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 4*DIM)

// Attention score channels for SDPA backward
#define SCORE_CH (HEADS*SEQ)

// Per-layer weights
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_att_b, *rms_ffn, *rms_ffn_b;
} LayerWeights;

// Adam optimizer state
typedef struct { float *m, *v; size_t n; } AdamState;
typedef struct {
    AdamState Wq, Wk, Wv, Wo, W1, W2, W3;
    AdamState rms_att, rms_att_b, rms_ffn, rms_ffn_b;
} LayerAdam;

// Per-layer activations (saved for backward)
typedef struct {
    float *layer_in, *xnorm, *Q, *K, *V, *attn_out, *o_out;
    float *x2, *x2norm, *h1, *h3, *silu_out, *ffn_out;
} LayerActs;

// Per-layer gradients
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *W1, *W2, *W3;
    float *rms_att, *rms_att_b, *rms_ffn, *rms_ffn_b;
} LayerGrads;

// ANE kernel handle
typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

// Per-layer IOSurfaces for pre-staged weights
typedef struct {
    IOSurfaceRef sdpaFwd_in, woFwd_in, ffnFused_in;
    IOSurfaceRef ffnBwdW2t_in, ffnBwdW13t_in, wotBwd_in, qBwd_in, kvBwd_in;
} PerLayerSurfaces;

// Per-layer ANE requests (bound to per-layer IOSurfaces)
typedef struct {
    void *sdpaFwd, *woFwd, *ffnFused;
    void *ffnBwdW2t, *ffnBwdW13t, *wotBwd, *qBwd, *kvBwd;
} PerLayerRequests;

// Checkpoint header
typedef struct {
    int magic, version, step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
    int kv_heads, head_dim, q_dim;
    int accum_steps, warmup_steps;
    float max_lr, min_lr_frac;
    float adam_b1, adam_b2, adam_eps, weight_decay;
    float grad_clip, loss_scale, res_alpha;
} CkptHdr;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
// Apple Silicon uses 16 KiB VM pages.  Buffers shared with Metal must keep a
// stable page-aligned allocation for the entire command-buffer lifetime.
#define JISHUI_SHARED_ALIGNMENT ((size_t)16384)
static size_t jishui_shared_bytes(size_t bytes) {
    if (bytes > SIZE_MAX - (JISHUI_SHARED_ALIGNMENT - 1)) {
        fprintf(stderr, "allocation size overflow: %zu bytes\n", bytes);
        abort();
    }
    return (bytes + JISHUI_SHARED_ALIGNMENT - 1) & ~(JISHUI_SHARED_ALIGNMENT - 1);
}
static size_t jishui_allocation_bytes(size_t count, size_t size) {
    if (size != 0 && count > SIZE_MAX / size) {
        fprintf(stderr, "allocation size overflow: %zu x %zu bytes\n", count, size);
        abort();
    }
    return count * size;
}
static void *jishui_malloc(size_t bytes) {
    void *ptr = malloc(bytes ? bytes : 1);
    if (!ptr) {
        fprintf(stderr, "allocation failed: %zu bytes\n", bytes);
        abort();
    }
    return ptr;
}
static void *jishui_calloc(size_t count, size_t size) {
    size_t bytes = jishui_allocation_bytes(count, size);
    void *ptr = calloc(count ? count : 1, size ? size : 1);
    if (!ptr) {
        fprintf(stderr, "allocation failed: %zu bytes\n", bytes);
        abort();
    }
    return ptr;
}
static void *jishui_aligned_malloc(size_t bytes) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, JISHUI_SHARED_ALIGNMENT, jishui_shared_bytes(bytes)) != 0) abort();
    return ptr;
}
static void *jishui_aligned_calloc(size_t count, size_t size) {
    size_t bytes = jishui_allocation_bytes(count, size);
    void *ptr = jishui_aligned_malloc(bytes);
    memset(ptr, 0, jishui_shared_bytes(bytes));
    return ptr;
}
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)jishui_calloc(n,4); s.v=(float*)jishui_calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)jishui_malloc(WQ_SZ*4); w.Wk=(float*)jishui_malloc(WK_SZ*4);
    w.Wv=(float*)jishui_malloc(WV_SZ*4); w.Wo=(float*)jishui_malloc(WO_SZ*4);
    w.W1=(float*)jishui_malloc(W1_SZ*4); w.W2=(float*)jishui_malloc(W2_SZ*4); w.W3=(float*)jishui_malloc(W3_SZ*4);
    w.rms_att=(float*)jishui_aligned_malloc(DIM*4); w.rms_att_b=(float*)jishui_aligned_malloc(DIM*4);
    w.rms_ffn=(float*)jishui_aligned_malloc(DIM*4); w.rms_ffn_b=(float*)jishui_aligned_malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);
    free(w->rms_att);free(w->rms_att_b);free(w->rms_ffn);free(w->rms_ffn_b);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WK_SZ); a.Wv=adam_alloc(WV_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_att_b=adam_alloc(DIM);
    a.rms_ffn=adam_alloc(DIM); a.rms_ffn_b=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_att_b);
    adam_free(&a->rms_ffn);adam_free(&a->rms_ffn_b);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)jishui_aligned_malloc(SEQ*DIM*4);
    a.xnorm=(float*)jishui_aligned_malloc(SEQ*DIM*4);
    a.Q=(float*)jishui_aligned_malloc(SEQ*Q_DIM*4); a.K=(float*)jishui_aligned_malloc(SEQ*KV_DIM*4); a.V=(float*)jishui_aligned_malloc(SEQ*KV_DIM*4);
    a.attn_out=(float*)jishui_aligned_malloc(SEQ*Q_DIM*4); a.o_out=(float*)jishui_malloc(SEQ*DIM*4);
    a.x2=(float*)jishui_aligned_malloc(SEQ*DIM*4); a.x2norm=(float*)jishui_aligned_malloc(SEQ*DIM*4);
    a.h1=(float*)jishui_aligned_malloc(SEQ*HIDDEN*4); a.h3=(float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
    a.silu_out=(float*)jishui_aligned_malloc(SEQ*HIDDEN*4); a.ffn_out=(float*)jishui_malloc(SEQ*DIM*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);
    free(a->Q);free(a->K);free(a->V);
    free(a->attn_out);free(a->o_out);free(a->x2);free(a->x2norm);
    free(a->h1);free(a->h3);free(a->silu_out);free(a->ffn_out);
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)jishui_calloc(WQ_SZ,4); g.Wk=(float*)jishui_calloc(WK_SZ,4);
    g.Wv=(float*)jishui_calloc(WV_SZ,4); g.Wo=(float*)jishui_calloc(WO_SZ,4);
    g.W1=(float*)jishui_calloc(W1_SZ,4); g.W2=(float*)jishui_calloc(W2_SZ,4); g.W3=(float*)jishui_calloc(W3_SZ,4);
    g.rms_att=(float*)jishui_calloc(DIM,4); g.rms_att_b=(float*)jishui_calloc(DIM,4);
    g.rms_ffn=(float*)jishui_calloc(DIM,4); g.rms_ffn_b=(float*)jishui_calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WK_SZ*4);
    memset(g->Wv,0,WV_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_att_b,0,DIM*4);
    memset(g->rms_ffn,0,DIM*4);memset(g->rms_ffn_b,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);
    free(g->rms_att);free(g->rms_att_b);free(g->rms_ffn);free(g->rms_ffn_b);
}
