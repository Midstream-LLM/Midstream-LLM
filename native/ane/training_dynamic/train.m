// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "ane_data.h"
#include "metal_ops.h"
#include <dirent.h>
#include <errno.h>

typedef enum {
    ANE_DATA_FLAT = 0,
    ANE_DATA_RECORDS = 1,
    ANE_DATA_INDEX = 2,
} AneDataMode;

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T → dx_ffn (HIDDEN → DIM)
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
} DynLayerKernels;

static void free_dynamic_kernels(DynLayerKernels *dk) {
    free_kern(dk->sdpaFwd); free_kern(dk->woFwd); free_kern(dk->ffnFused);
    free_kern(dk->ffnBwdW2t); free_kern(dk->ffnBwdW13t); free_kern(dk->wotBwd);
    free_kern(dk->sdpaBwd1); free_kern(dk->sdpaBwd2);
    free_kern(dk->qBwd); free_kern(dk->kvBwd);
    memset(dk, 0, sizeof(*dk));
}

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    memset(dk, 0, sizeof(*dk));
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;

    // SDPA forward (no Wo): [1, DIM, 1, SDPA_FWD_SP] → [1, sdpa_out_ch, 1, SEQ]
    printf("  Compiling sdpaFwd (GQA)...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) goto fail;

    // Wo forward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) goto fail;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) goto fail;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) goto fail;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) goto fail;

    // Wo^T backward: [1, DIM, 1, SEQ+Q_DIM] → [1, Q_DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
    if (!dk->wotBwd) goto fail;

    // SDPA bwd1 (weight-free, has mask): [1, 4*Q_DIM, 1, SEQ] → [1, Q_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1 (GQA)...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) goto fail;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+2*Q_DIM, 1, SEQ] → [1, 2*Q_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2 (GQA)...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) goto fail;

    // Q backward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) goto fail;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) goto fail;

    return true;
fail:
    free_dynamic_kernels(dk);
    return false;
}

// ===== Checkpoint =====
typedef struct {
    int step;
    char path[PATH_MAX];
} CheckpointEntry;

static int checkpoint_entry_cmp(const void *a, const void *b) {
    const CheckpointEntry *ea = (const CheckpointEntry *)a;
    const CheckpointEntry *eb = (const CheckpointEntry *)b;
    return (ea->step > eb->step) - (ea->step < eb->step);
}

static void split_checkpoint_base(const char *base, char *dir, size_t dir_cap,
                                  char *name, size_t name_cap) {
    const char *slash = strrchr(base, '/');
    if (!slash) {
        snprintf(dir, dir_cap, ".");
        snprintf(name, name_cap, "%s", base);
        return;
    }
    size_t dir_len = (size_t)(slash - base);
    if (dir_len == 0) dir_len = 1;  // root directory
    if (dir_len >= dir_cap) dir_len = dir_cap - 1;
    memcpy(dir, base, dir_len); dir[dir_len] = '\0';
    snprintf(name, name_cap, "%s", slash + 1);
}

static void prune_checkpoint_files(const char *base, int keep) {
    if (keep < 1) return;
    char dir[PATH_MAX], name[PATH_MAX];
    split_checkpoint_base(base, dir, sizeof(dir), name, sizeof(name));
    DIR *dp = opendir(dir);
    if (!dp) return;
    char prefix[PATH_MAX];
    snprintf(prefix, sizeof(prefix), "%s.step_", name);
    size_t prefix_len = strlen(prefix);
    CheckpointEntry *entries = NULL;
    size_t n = 0, cap = 0;
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (strncmp(entry->d_name, prefix, prefix_len) != 0) continue;
        const char *digits = entry->d_name + prefix_len;
        if (!*digits) continue;
        char *end = NULL;
        unsigned long parsed = strtoul(digits, &end, 10);
        if (!end || *end != '\0' || parsed > INT_MAX) continue;
        if (n == cap) {
            size_t next = cap ? cap * 2 : 8;
            CheckpointEntry *grown = (CheckpointEntry *)realloc(entries, next * sizeof(*entries));
            if (!grown) break;
            entries = grown; cap = next;
        }
        entries[n].step = (int)parsed;
        snprintf(entries[n].path, sizeof(entries[n].path), "%s/%s", dir, entry->d_name);
        n++;
    }
    closedir(dp);
    qsort(entries, n, sizeof(*entries), checkpoint_entry_cmp);
    if (n > (size_t)keep) {
        for (size_t i = 0; i < n - (size_t)keep; i++) unlink(entries[i].path);
    }
    free(entries);
}

static bool resolve_checkpoint_path(const char *requested, char *resolved, size_t cap) {
    if (access(requested, R_OK) == 0) {
        if (snprintf(resolved, cap, "%s", requested) >= (int)cap) return false;
        return true;
    }
    char dir[PATH_MAX], name[PATH_MAX];
    split_checkpoint_base(requested, dir, sizeof(dir), name, sizeof(name));
    DIR *dp = opendir(dir);
    if (!dp) return false;
    char prefix[PATH_MAX];
    snprintf(prefix, sizeof(prefix), "%s.step_", name);
    size_t prefix_len = strlen(prefix);
    int best_step = -1;
    char best[PATH_MAX] = {0};
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (strncmp(entry->d_name, prefix, prefix_len) != 0) continue;
        const char *digits = entry->d_name + prefix_len;
        char *end = NULL;
        unsigned long parsed = strtoul(digits, &end, 10);
        if (!*digits || !end || *end != '\0' || parsed > INT_MAX || (int)parsed <= best_step) continue;
        best_step = (int)parsed;
        snprintf(best, sizeof(best), "%s/%s", dir, entry->d_name);
    }
    closedir(dp);
    if (best_step < 0 || snprintf(resolved, cap, "%s", best) >= (int)cap) return false;
    return true;
}

static bool save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t, int keep,
                            int accum_steps, int warmup_steps, float max_lr, float min_lr_frac,
                            float adam_b1, float adam_b2, float adam_eps, float weight_decay,
                            float grad_clip, float loss_scale, float res_alpha,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, float *rms_final_b,
                            AdamState *arms_final, AdamState *arms_final_b,
                            float *embed, AdamState *aembed) {
    char destination[PATH_MAX], temporary[PATH_MAX];
    if (snprintf(destination, sizeof(destination), "%s.step_%08d", path, step) >= (int)sizeof(destination) ||
        snprintf(temporary, sizeof(temporary), "%s.tmp", destination) >= (int)sizeof(temporary)) {
        printf("checkpoint path is too long\n"); return false;
    }
    FILE *f = fopen(temporary, "wb");
    if (!f) { perror(temporary); return false; }
    bool ok = true;
#define WRITE_CKPT(ptr, size, count) do { if (fwrite((ptr), (size), (count), f) != (count)) ok = false; } while (0)
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 6;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    h.accum_steps = accum_steps; h.warmup_steps = warmup_steps;
    h.max_lr = max_lr; h.min_lr_frac = min_lr_frac;
    h.adam_b1 = adam_b1; h.adam_b2 = adam_b2; h.adam_eps = adam_eps;
    h.weight_decay = weight_decay; h.grad_clip = grad_clip;
    h.loss_scale = loss_scale; h.res_alpha = res_alpha;
    WRITE_CKPT(&h, sizeof(h), 1);
    for (int L = 0; L < NLAYERS; L++) {
        WRITE_CKPT(lw[L].Wq,4,WQ_SZ); WRITE_CKPT(lw[L].Wk,4,WK_SZ);
        WRITE_CKPT(lw[L].Wv,4,WV_SZ); WRITE_CKPT(lw[L].Wo,4,WO_SZ);
        WRITE_CKPT(lw[L].W1,4,W1_SZ); WRITE_CKPT(lw[L].W2,4,W2_SZ); WRITE_CKPT(lw[L].W3,4,W3_SZ);
        WRITE_CKPT(lw[L].rms_att,4,DIM); WRITE_CKPT(lw[L].rms_att_b,4,DIM);
        WRITE_CKPT(lw[L].rms_ffn,4,DIM); WRITE_CKPT(lw[L].rms_ffn_b,4,DIM);
        WRITE_CKPT(la[L].Wq.m,4,WQ_SZ); WRITE_CKPT(la[L].Wq.v,4,WQ_SZ);
        WRITE_CKPT(la[L].Wk.m,4,WK_SZ); WRITE_CKPT(la[L].Wk.v,4,WK_SZ);
        WRITE_CKPT(la[L].Wv.m,4,WV_SZ); WRITE_CKPT(la[L].Wv.v,4,WV_SZ);
        WRITE_CKPT(la[L].Wo.m,4,WO_SZ); WRITE_CKPT(la[L].Wo.v,4,WO_SZ);
        WRITE_CKPT(la[L].W1.m,4,W1_SZ); WRITE_CKPT(la[L].W1.v,4,W1_SZ);
        WRITE_CKPT(la[L].W2.m,4,W2_SZ); WRITE_CKPT(la[L].W2.v,4,W2_SZ);
        WRITE_CKPT(la[L].W3.m,4,W3_SZ); WRITE_CKPT(la[L].W3.v,4,W3_SZ);
        WRITE_CKPT(la[L].rms_att.m,4,DIM); WRITE_CKPT(la[L].rms_att.v,4,DIM);
        WRITE_CKPT(la[L].rms_att_b.m,4,DIM); WRITE_CKPT(la[L].rms_att_b.v,4,DIM);
        WRITE_CKPT(la[L].rms_ffn.m,4,DIM); WRITE_CKPT(la[L].rms_ffn.v,4,DIM);
        WRITE_CKPT(la[L].rms_ffn_b.m,4,DIM); WRITE_CKPT(la[L].rms_ffn_b.v,4,DIM);
    }
    WRITE_CKPT(rms_final,4,DIM); WRITE_CKPT(rms_final_b,4,DIM);
    WRITE_CKPT(arms_final->m,4,DIM); WRITE_CKPT(arms_final->v,4,DIM);
    WRITE_CKPT(arms_final_b->m,4,DIM); WRITE_CKPT(arms_final_b->v,4,DIM);
    WRITE_CKPT(embed,4,VOCAB*DIM);
    WRITE_CKPT(aembed->m,4,VOCAB*DIM); WRITE_CKPT(aembed->v,4,VOCAB*DIM);
#undef WRITE_CKPT
    if (fflush(f) != 0 || fsync(fileno(f)) != 0) ok = false;
    if (fclose(f) != 0) ok = false;
    if (!ok || rename(temporary, destination) != 0) {
        if (ok) perror("checkpoint rename");
        unlink(temporary);
        return false;
    }
    prune_checkpoint_files(path, keep);
    printf("  [checkpoint file: %s]\n", destination);
    return true;
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             int *accum_steps, int *warmup_steps,
                             float *max_lr, float *min_lr_frac,
                             float *adam_b1, float *adam_b2, float *adam_eps, float *weight_decay,
                             float *grad_clip, float *loss_scale,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, float *rms_final_b,
                             AdamState *arms_final, AdamState *arms_final_b,
                             float *embed, AdamState *aembed) {
    char resolved[PATH_MAX];
    if (!resolve_checkpoint_path(path, resolved, sizeof(resolved))) return false;
    FILE *f = fopen(resolved, "rb");
    if (!f) return false;
    struct stat checkpoint_stat;
    if (fstat(fileno(f), &checkpoint_stat) != 0) { fclose(f); return false; }
    uint64_t expected = sizeof(CkptHdr);
    uint64_t layer_floats = (uint64_t)WQ_SZ * 3 + WK_SZ * 3 + WV_SZ * 3 +
        WO_SZ * 3 + W1_SZ * 3 + W2_SZ * 3 + W3_SZ * 3 + (uint64_t)DIM * 12;
    expected += (uint64_t)NLAYERS * layer_floats * sizeof(float);
    expected += (uint64_t)DIM * 6 * sizeof(float);  // final norms + Adam states
    expected += (uint64_t)VOCAB * DIM * 3 * sizeof(float);  // tied embed + Adam m/v
    if ((uint64_t)checkpoint_stat.st_size != expected) {
        fclose(f); printf("invalid checkpoint size: %s\n", resolved); return false;
    }
    CkptHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return false; }
    if (h.magic != 0x424C5A54 || h.version != 6 ||
        h.n_layers != NLAYERS || h.vocab_size != VOCAB || h.dim != DIM ||
        h.hidden_dim != HIDDEN || h.n_heads != HEADS || h.kv_heads != KV_HEADS ||
        h.head_dim != HD || h.q_dim != Q_DIM || h.seq_len != SEQ ||
        fabsf(h.res_alpha - RES_ALPHA) > 1.0e-8f || h.accum_steps < 1 ||
        h.warmup_steps < 0 || h.step < 0 || h.total_steps < h.step ||
        h.cum_steps != h.step || h.adam_t < 0 || h.adam_t > h.cum_steps ||
        !isfinite(h.lr) || h.lr <= 0.0f || !isfinite(h.loss) ||
        !isfinite(h.cum_train) || h.cum_train < 0.0 ||
        !isfinite(h.cum_wall) || h.cum_wall < 0.0 ||
        !isfinite(h.max_lr) || h.max_lr <= 0.0f ||
        !isfinite(h.min_lr_frac) || h.min_lr_frac < 0.0f || h.min_lr_frac > 1.0f ||
        !isfinite(h.adam_b1) || h.adam_b1 < 0.0f || h.adam_b1 >= 1.0f ||
        !isfinite(h.adam_b2) || h.adam_b2 < 0.0f || h.adam_b2 >= 1.0f ||
        !isfinite(h.adam_eps) || h.adam_eps <= 0.0f ||
        !isfinite(h.weight_decay) || h.weight_decay < 0.0f ||
        !isfinite(h.grad_clip) || h.grad_clip < 0.0f ||
        !isfinite(h.loss_scale) || h.loss_scale <= 0.0f) {
        fclose(f); return false;
    }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    *accum_steps = h.accum_steps; *warmup_steps = h.warmup_steps;
    *max_lr = h.max_lr; *min_lr_frac = h.min_lr_frac;
    *adam_b1 = h.adam_b1; *adam_b2 = h.adam_b2; *adam_eps = h.adam_eps;
    *weight_decay = h.weight_decay; *grad_clip = h.grad_clip; *loss_scale = h.loss_scale;
#define READ_CKPT(ptr, size, count) do { \
    if (fread((ptr), (size), (count), f) != (count)) { \
        printf("checkpoint read failed: %s\n", resolved); fclose(f); return false; \
    } \
} while (0)
    for (int L = 0; L < NLAYERS; L++) {
        READ_CKPT(lw[L].Wq,4,WQ_SZ); READ_CKPT(lw[L].Wk,4,WK_SZ);
        READ_CKPT(lw[L].Wv,4,WV_SZ); READ_CKPT(lw[L].Wo,4,WO_SZ);
        READ_CKPT(lw[L].W1,4,W1_SZ); READ_CKPT(lw[L].W2,4,W2_SZ); READ_CKPT(lw[L].W3,4,W3_SZ);
        READ_CKPT(lw[L].rms_att,4,DIM); READ_CKPT(lw[L].rms_att_b,4,DIM);
        READ_CKPT(lw[L].rms_ffn,4,DIM); READ_CKPT(lw[L].rms_ffn_b,4,DIM);
        READ_CKPT(la[L].Wq.m,4,WQ_SZ); READ_CKPT(la[L].Wq.v,4,WQ_SZ);
        READ_CKPT(la[L].Wk.m,4,WK_SZ); READ_CKPT(la[L].Wk.v,4,WK_SZ);
        READ_CKPT(la[L].Wv.m,4,WV_SZ); READ_CKPT(la[L].Wv.v,4,WV_SZ);
        READ_CKPT(la[L].Wo.m,4,WO_SZ); READ_CKPT(la[L].Wo.v,4,WO_SZ);
        READ_CKPT(la[L].W1.m,4,W1_SZ); READ_CKPT(la[L].W1.v,4,W1_SZ);
        READ_CKPT(la[L].W2.m,4,W2_SZ); READ_CKPT(la[L].W2.v,4,W2_SZ);
        READ_CKPT(la[L].W3.m,4,W3_SZ); READ_CKPT(la[L].W3.v,4,W3_SZ);
        READ_CKPT(la[L].rms_att.m,4,DIM); READ_CKPT(la[L].rms_att.v,4,DIM);
        READ_CKPT(la[L].rms_att_b.m,4,DIM); READ_CKPT(la[L].rms_att_b.v,4,DIM);
        READ_CKPT(la[L].rms_ffn.m,4,DIM); READ_CKPT(la[L].rms_ffn.v,4,DIM);
        READ_CKPT(la[L].rms_ffn_b.m,4,DIM); READ_CKPT(la[L].rms_ffn_b.v,4,DIM);
    }
    READ_CKPT(rms_final,4,DIM); READ_CKPT(rms_final_b,4,DIM);
    READ_CKPT(arms_final->m,4,DIM); READ_CKPT(arms_final->v,4,DIM);
    READ_CKPT(arms_final_b->m,4,DIM); READ_CKPT(arms_final_b->v,4,DIM);
    READ_CKPT(embed,4,VOCAB*DIM);
    READ_CKPT(aembed->m,4,VOCAB*DIM); READ_CKPT(aembed->v,4,VOCAB*DIM);
#undef READ_CKPT
    if (fclose(f) != 0) return false;
    printf("  [checkpoint loaded: %s]\n", resolved);
    return true;
}

int main(int argc, char *argv[]) {
    int exit_code = 0;
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        int requested_optimizer_steps = -1;
        bool steps_explicit = false;
        bool microsteps_explicit = false;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.95f, adam_eps=ADAM_EPS, wd=0.1f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        int log_interval = 10;
        int save_interval = 100;
        int keep_checkpoints = 3;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f;
        float res_alpha = RES_ALPHA;
        float min_lr_frac = 0.1f;

        bool do_resume = false, from_scratch = false, record_mode = false;
        bool checkpoint_enabled = true;
        bool use_metal_norm = false, use_metal_silu = false, use_metal_io = false;
        bool metal_shadow = false;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *index_path = NULL;
        const char *data_dir = DEFAULT_DATA_PATH;
        const char *ckpt_path = CKPT_PATH;
        const char *parity_path = NULL;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [--scratch|--resume] [--index INDEX --data-dir DATASET | --records --data FILE | --data FILE]\n", argv[0]);
                printf("       --steps N | --optimizer-steps N  --accum N  --save-interval N(updates)  --keep-checkpoints N\n");
                printf("       --checkpoint PREFIX | --no-checkpoint  --log-interval N  --lr LR  --warmup N(updates)\n");
                printf("       --metal-norm  --metal-silu  --metal-io  --metal-shadow  --parity-report FILE\n");
                return 0;
            }
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) {
                total_steps = atoi(argv[++i]); steps_explicit = true; microsteps_explicit = true;
            }
            else if (strcmp(argv[i], "--optimizer-steps") == 0 && i+1<argc) {
                requested_optimizer_steps = atoi(argv[++i]); steps_explicit = true;
            }
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--log-interval") == 0 && i+1<argc) log_interval = atoi(argv[++i]);
            else if (strcmp(argv[i], "--save-interval") == 0 && i+1<argc) save_interval = atoi(argv[++i]);
            else if (strcmp(argv[i], "--keep-checkpoints") == 0 && i+1<argc) keep_checkpoints = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--records") == 0) record_mode = true;
            else if (strcmp(argv[i], "--index") == 0 && i+1<argc) index_path = argv[++i];
            else if (strcmp(argv[i], "--data-dir") == 0 && i+1<argc) data_dir = argv[++i];
            else if (strcmp(argv[i], "--checkpoint") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--no-checkpoint") == 0) checkpoint_enabled = false;
            else if (strcmp(argv[i], "--parity-report") == 0 && i+1<argc) parity_path = argv[++i];
            else if (strcmp(argv[i], "--metal-norm") == 0) use_metal_norm = true;
            else if (strcmp(argv[i], "--metal-silu") == 0) use_metal_silu = true;
            else if (strcmp(argv[i], "--metal-io") == 0) use_metal_io = true;
            else if (strcmp(argv[i], "--metal-shadow") == 0) {
                use_metal_norm = true;
                use_metal_io = true;
                metal_shadow = true;
            }
            else {
                printf("Unknown or incomplete option: %s\n", argv[i]);
                return 1;
            }
        }
        AneDataMode data_mode = index_path ? ANE_DATA_INDEX :
            (record_mode ? ANE_DATA_RECORDS : ANE_DATA_FLAT);
        if (data_mode == ANE_DATA_INDEX && record_mode) {
            printf("--index and --records are mutually exclusive\n"); return 1;
        }
        if (do_resume == from_scratch) {
            printf("Choose exactly one of --scratch or --resume\n"); return 1;
        }
        if (microsteps_explicit && requested_optimizer_steps >= 0) {
            printf("--steps and --optimizer-steps are mutually exclusive\n"); return 1;
        }
        if (total_steps < 1 || accum_steps < 1 || warmup_steps < 0 || log_interval < 1 ||
            save_interval < 0 || keep_checkpoints < 1 || !isfinite(max_lr) || max_lr <= 0.0f ||
            !isfinite(grad_clip) || grad_clip < 0.0f) {
            printf("Invalid training, schedule, or checkpoint interval\n"); return 1;
        }
        if (requested_optimizer_steps >= 0) {
            if (requested_optimizer_steps < 1) {
                printf("--optimizer-steps must be positive\n"); return 1;
            }
            if (requested_optimizer_steps > INT_MAX / accum_steps) {
                printf("Requested optimizer budget is too large\n"); return 1;
            }
            total_steps = requested_optimizer_steps * accum_steps;
        }
        if (use_metal_norm && !USE_LAYER_NORM) {
            printf("--metal-norm currently supports LayerNorm models only\n");
            return 1;
        }
        FILE *parity_report = NULL;
        if (parity_path) {
            parity_report = fopen(parity_path, do_resume ? "a" : "w");
            if (!parity_report) {
                perror(parity_path);
                return 1;
            }
            setvbuf(parity_report, NULL, _IOLBF, 0);
            fprintf(parity_report,
                    "{\"record\":\"meta\",\"model\":\"%s\",\"seq\":%d,"
                    "\"layers\":%d,\"accum\":%d,\"metal_norm\":%s,"
                    "\"metal_silu\":%s,\"metal_io\":%s,\"metal_shadow\":%s}\n",
                    MODEL_NAME, SEQ, NLAYERS, accum_steps,
                    use_metal_norm ? "true" : "false",
                    use_metal_silu ? "true" : "false",
                    use_metal_io ? "true" : "false",
                    metal_shadow ? "true" : "false");
        }
        if ((use_metal_norm || use_metal_silu || use_metal_io) && !metal_ops_init()) return 1;
        metal_ops_set_shadow(metal_shadow);
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)jishui_aligned_malloc(DIM*4);
        float *rms_final_b = (float*)jishui_aligned_malloc(DIM*4);
        float *embed = (float*)jishui_malloc(VOCAB*DIM*4);
        float *grms_final = (float*)jishui_calloc(DIM, 4);
        float *grms_final_b = (float*)jishui_calloc(DIM, 4);
        float *gembed = (float*)jishui_calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState arms_final_b = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            int checkpoint_total_steps = total_steps;
            resuming = load_checkpoint(ckpt_path, &start_step, &checkpoint_total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                &accum_steps, &warmup_steps, &max_lr, &min_lr_frac,
                &adam_b1, &adam_b2, &adam_eps, &wd, &grad_clip, &loss_scale,
                lw, la, rms_final, rms_final_b, &arms_final, &arms_final_b, embed, &aembed);
            if (resuming && !steps_explicit) total_steps = checkpoint_total_steps;
            if (resuming && requested_optimizer_steps >= 0) {
                if (requested_optimizer_steps > (INT_MAX - start_step) / accum_steps) {
                    printf("Requested resumed optimizer budget is too large\n"); return 1;
                }
                total_steps = start_step + requested_optimizer_steps * accum_steps;
            }
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
            else {
                printf("Unable to resume from checkpoint prefix %s\n", ckpt_path);
                return 1;
            }
        }
        if (resuming && total_steps <= start_step) {
            printf("Training target %d must be greater than checkpoint step %d\n",
                   total_steps, start_step);
            return 1;
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training: %s (%d layers, GQA %d/%d heads) ===\n",
                   MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
            printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
                   DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
            double xformer_m = (double)NLAYERS*(WQ_SZ + WK_SZ + WV_SZ + (double)WO_SZ + W1_SZ + W2_SZ + W3_SZ + 4.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            printf("Kernels: 10 compiled (sdpaFwd+woFwd, ffnFused, ffnBwdW2t+W13t, wotBwd, sdpaBwd1+2, qBwd+kvBwd)\n");
            printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);
            double fwd_flops = 2.0*NLAYERS*((double)WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;
            printf("FLOPs/step: fwd=%.1fM total=%.1fM\n", fwd_flops/1e6, total_flops/1e6);
            if (from_scratch) {
                printf("  Training from scratch (random init)\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_qd=1.0f/sqrtf(Q_DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = res_alpha;
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){
                        lw[L].rms_att[i]=1.0f; lw[L].rms_att_b[i]=0.0f;
                        lw[L].rms_ffn[i]=1.0f; lw[L].rms_ffn_b[i]=0.0f;
                    }
                }
                for(int i=0;i<DIM;i++){ rms_final[i]=1.0f; rms_final_b[i]=0.0f; }
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            } else {
                printf("  ERROR: Pretrained weight loading not implemented for Qwen3. Use --scratch.\n");
                return 1;
            }
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)jishui_malloc(WQ_SZ*4); Wkt_buf[L]=(float*)jishui_malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)jishui_malloc(WV_SZ*4); Wot_buf[L]=(float*)jishui_malloc(WO_SZ*4);
            W1t_buf[L]=(float*)jishui_malloc(W1_SZ*4); W2t_buf[L]=(float*)jishui_malloc(W2_SZ*4);
            W3t_buf[L]=(float*)jishui_malloc(W3_SZ*4);
            // Wq is [Q_DIM, DIM] → Wq^T is [DIM, Q_DIM] (staged as [DIM channels, Q_DIM spatial])
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            // Wk is [KV_DIM, DIM] → Wk^T is [DIM, KV_DIM]
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            // Wv is [KV_DIM, DIM] → Wv^T is [DIM, KV_DIM]
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            // Wo is [DIM, Q_DIM] → Wo^T is [Q_DIM, DIM]
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // Data source.  Index mode mmaps the original NPY shards and samples
        // document-aware windows on demand; flat/record modes retain the
        // small-file benchmark path used by the upstream ANE example.
        int data_fd = -1;
        void *data_mapping = NULL;
        size_t data_len = 0;
        uint16_t *token_data = NULL;
        size_t n_tokens = 0;
        size_t n_records = 0;
        AneTrainIndex train_index;
        memset(&train_index, 0, sizeof(train_index));
        if (data_mode == ANE_DATA_INDEX) {
#if !DISABLE_VOCAB_COMPACTION
            printf("--index requires DISABLE_VOCAB_COMPACTION for this model\n");
            return 1;
#endif
            if (!index_path || !ane_index_load(&train_index, index_path, data_dir)) {
                printf("Unable to load ANE train index %s\n", index_path ? index_path : "(null)");
                ane_index_free(&train_index);
                return 1;
            }
            printf("ANE index: %zu train docs across %d NPY shards (data=%s)\n",
                   train_index.n_docs, train_index.n_shards, data_dir);
            for (int c = 1; c <= 6; c++) {
                if (train_index.group_n[c])
                    printf("  category %d: %zu docs, %llu tokens, p=%.4f\n", c,
                           train_index.group_n[c],
                           (unsigned long long)train_index.group_total[c],
                           train_index.group_prob[c]);
            }
        } else {
            data_fd = open(data_path, O_RDONLY);
            if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
            struct stat st;
            if (fstat(data_fd, &st) != 0) {
                printf("Cannot stat token data file: %s\n", data_path);
                close(data_fd); return 1;
            }
            if (S_ISDIR(st.st_mode)) {
                printf("%s is a directory; use --index INDEX --data-dir DATASET\n", data_path);
                close(data_fd); return 1;
            }
            if (st.st_size <= 0) {
                printf("Invalid token data file: %s\n", data_path);
                close(data_fd); return 1;
            }
            data_len = (size_t)st.st_size;
            data_mapping = mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
            if (data_mapping == MAP_FAILED) {
                printf("mmap failed for %s\n", data_path);
                close(data_fd); return 1;
            }
            token_data = (uint16_t *)data_mapping;
            if (data_len & 1) {
                printf("Token data size is not even\n");
                munmap(data_mapping, data_len); close(data_fd); return 1;
            }
            n_tokens = data_len / sizeof(uint16_t);
            for (size_t i = 0; i < n_tokens; i++) {
                if (token_data[i] >= VOCAB) {
                    printf("Token id %u at offset %zu exceeds vocab %d\n",
                           (unsigned)token_data[i], i, VOCAB);
                    munmap(data_mapping, data_len); close(data_fd); return 1;
                }
            }
            if (data_mode == ANE_DATA_RECORDS) {
                size_t record_bytes = (size_t)(SEQ + 1) * sizeof(uint16_t);
                if (data_len == 0 || data_len % record_bytes != 0) {
                    printf("Record data size is not a multiple of %zu bytes\n", record_bytes);
                    munmap(data_mapping, data_len); close(data_fd); return 1;
                }
                n_records = data_len / record_bytes;
                printf("ANE records: %zu records, width=%d (%.1f MB)\n",
                       n_records, SEQ + 1, data_len/1e6);
            } else {
                printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);
            }
        }

        // Vocab compaction
        VocabMap vm;
#if DISABLE_VOCAB_COMPACTION
        vm = vocab_map_identity(VOCAB);
#else
        if (data_mode == ANE_DATA_INDEX) {
            printf("Index mode cannot build a compact vocab; use a Jishui model header\n");
            ane_index_free(&train_index);
            return 1;
        }
        vm = vocab_map_build(token_data, n_tokens, VOCAB);
#endif
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);

        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = (float*)jishui_calloc((size_t)CV*DIM, 4);

        // ===== Compile all kernels ONCE =====
        printf("Compiling 10 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 10 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        printf("Allocating per-layer IOSurfaces...\n");
        PerLayerSurfaces pls[NLAYERS] = {0};
        PerLayerRequests plr[NLAYERS] = {0};
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
            pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
            if (!pls[L].sdpaFwd_in || !pls[L].woFwd_in || !pls[L].ffnFused_in ||
                !pls[L].ffnBwdW2t_in || !pls[L].ffnBwdW13t_in || !pls[L].wotBwd_in ||
                !pls[L].qBwd_in || !pls[L].kvBwd_in || !plr[L].sdpaFwd ||
                !plr[L].woFwd || !plr[L].ffnFused || !plr[L].ffnBwdW2t ||
                !plr[L].ffnBwdW13t || !plr[L].wotBwd || !plr[L].qBwd || !plr[L].kvBwd) {
                printf("Per-layer IOSurface/request allocation failed at layer %d\n", L);
                return 1;
            }
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
        }
        printf("Per-layer weight staging complete\n\n");

        // Gradient + work buffers (GQA: Q has Q_DIM, K/V have KV_DIM)
        float *dy = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *dffn = (float*)jishui_malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *dx2 = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *dx_attn = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *dq = (float*)jishui_malloc(SEQ*Q_DIM*4);     // Q_DIM for Q grads
        float *dk_buf = (float*)jishui_malloc(SEQ*KV_DIM*4); // KV_DIM for K grads
        float *dv = (float*)jishui_malloc(SEQ*KV_DIM*4);     // KV_DIM for V grads
        float *da_buf = (float*)jishui_malloc(SEQ*Q_DIM*4);  // Q_DIM for attn grads
        float *x_cur = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *x_final = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *dx_norm_scratch = (float*)jishui_aligned_calloc(SEQ*DIM,4);
        float *logits = (float*)jishui_malloc(SEQ*CV*4);
        float *dlogits = (float*)jishui_malloc(SEQ*CV*4);
        float *dh1 = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)jishui_malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)jishui_malloc(SEQ*HIDDEN*4);
        // GQA tile/reduce buffers
        float *k_tiled = (float*)jishui_malloc(SEQ*Q_DIM*4);  // KV_DIM → Q_DIM
        float *v_tiled = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *dq_full = (float*)jishui_malloc(SEQ*Q_DIM*4);  // from sdpaBwd2
        float *dk_full = (float*)jishui_malloc(SEQ*Q_DIM*4);  // from sdpaBwd2 (needs reduce)
        float *dv_full = (float*)jishui_malloc(SEQ*Q_DIM*4);  // from sdpaBwd1 (needs reduce)
        float *dx_kv = (float*)jishui_malloc(SEQ*DIM*4);
        float *dx2_scaled = (float*)jishui_malloc(SEQ*DIM*4);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();
        dispatch_semaphore_t dw_slots = dispatch_semaphore_create(2);
        if (!dw_q || !dw_grp || !dw_slots) {
            printf("Unable to create dW dispatch resources\n"); return 1;
        }

        float last_loss = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        int accumulated_microsteps = 0;
        int last_checkpoint_step = start_step;
        int64_t remaining_microsteps = (int64_t)total_steps - start_step;
        int64_t remaining_updates_64 = (remaining_microsteps + accum_steps - 1) / accum_steps;
        if (remaining_updates_64 > INT_MAX - adam_t) {
            printf("Optimizer schedule is too large\n"); return 1;
        }
        int total_optimizer_steps = adam_t + (int)remaining_updates_64;
        bool training_failed = false;
        uint64_t t_wall_start = mach_absolute_time();
        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t_step = mach_absolute_time();
            AneRng sample_rng;
            ane_rng_seed(&sample_rng, UINT64_C(0x4a4953485549) ^ (uint64_t)(uint32_t)step);

            // Sample data
            uint16_t *input_tokens;
            uint16_t *target_tokens_raw;
            if (data_mode == ANE_DATA_INDEX) {
                if (!ane_index_sample(&train_index, &sample_rng)) {
                    printf("ANE index sampler failed at step %d\n", step);
                    training_failed = true;
                    break;
                }
                input_tokens = train_index.sample;
                target_tokens_raw = train_index.sample + 1;
            } else if (data_mode == ANE_DATA_RECORDS) {
                size_t rec = (size_t)(ane_rng_uniform(&sample_rng) * (double)n_records);
                input_tokens = token_data + rec * (SEQ + 1);
                target_tokens_raw = input_tokens + 1;
            } else {
                if (n_tokens < (size_t)SEQ + 1) {
                    printf("Token data is shorter than seq length %d\n", SEQ);
                    training_failed = true;
                    break;
                }
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(ane_rng_uniform(&sample_rng) * (double)(max_pos + 1));
                if (pos > max_pos) pos = max_pos;
                input_tokens = token_data + pos;
                target_tokens_raw = token_data + pos + 1;
            }

            uint16_t ctargets[SEQ];
            bool token_error = false;
            for (int t = 0; t < SEQ; t++) {
                if (input_tokens[t] >= VOCAB || target_tokens_raw[t] >= VOCAB ||
                    vm.full_to_compact[target_tokens_raw[t]] < 0) {
                    printf("token id %u is outside the active vocab at step %d\n",
                           (unsigned)(input_tokens[t] >= VOCAB ? input_tokens[t] : target_tokens_raw[t]), step);
                    token_error = true;
                    break;
                }
                ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];
            }
            if (token_error) {
                training_failed = true;
                break;
            }

            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0;
            double t_cls_fwd=0, t_xent=0, t_cls_bwd=0, t_dw_copy=0;

            // ===== FORWARD (28 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // Pre-attention norm (CPU or Metal)
                t0 = mach_absolute_time();
                if (use_metal_norm) {
                    if (!metal_layernorm_forward(xnorm_buf, x_cur, lw[L].rms_att, lw[L].rms_att_b)) return 1;
                } else {
                    norm_forward(xnorm_buf, x_cur, lw[L].rms_att, lw[L].rms_att_b, DIM, SEQ);
                }
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                // SDPA forward (ANE): xnorm + Wq,Wk,Wv → attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM], xnorm[DIM]
                t0 = mach_absolute_time();
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd)) return 1;
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Unpack SDPA caches and stage Wo activations through Metal or CPU.
                t0 = mach_absolute_time();
                if (use_metal_io) {
                    if (!metal_sdpa_unpack_and_pack(dk.sdpaFwd->ioOut, pls[L].woFwd_in,
                                                    ac->attn_out, ac->Q, ac->K, ac->V))
                        return 1;
                } else {
                    IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                    _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                    int off = 0;
                    cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                    cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                    cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                    cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ);
                    IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                    write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out);
                }
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Wo forward (ANE): attn_out[Q_DIM] -> o_out[DIM]
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.woFwd, plr[L].woFwd)) return 1;
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.woFwd->ioOut, ac->o_out, DIM, SEQ);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Scaled residual + pre-FFN norm
                t0 = mach_absolute_time();
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                if (use_metal_norm) {
                    if (!metal_layernorm_forward(ac->x2norm, ac->x2, lw[L].rms_ffn, lw[L].rms_ffn_b)) return 1;
                } else {
                    norm_forward(ac->x2norm, ac->x2, lw[L].rms_ffn, lw[L].rms_ffn_b, DIM, SEQ);
                }
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Fused FFN (ANE)
                t0 = mach_absolute_time();
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnFused, plr[L].ffnFused)) return 1;
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                if (use_metal_io) {
                    if (!metal_ffn_unpack_output(dk.ffnFused->ioOut, x_cur, ac->h1,
                                                 ac->h3, ac->silu_out))
                        return 1;
                } else {
                    IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                    _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                    int off = 0;
                    cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                    cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                    cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                    cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
                    IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                }
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
            }

            // Final norm + classifier + loss
            t0 = mach_absolute_time();
            if (use_metal_norm) {
                if (!metal_layernorm_forward(x_final, x_cur, rms_final, rms_final_b)) return 1;
            } else {
                norm_forward(x_final, x_cur, rms_final, rms_final_b, DIM, SEQ);
            }
            t_rms += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            // Token-major logits [SEQ,CV] keep each softmax row contiguous.
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, x_final, SEQ, cembed, DIM, 0.0f, logits, CV);
            t_cls_fwd += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_xent += tb_ms(mach_absolute_time() - t0);
            if (!isfinite(loss)) {
                printf("Non-finite loss at microstep %d; refusing to update Adam\n", step + 1);
                return 1;
            }
            last_loss = loss;

            // ===== BACKWARD =====
            vDSP_vsmul(dlogits, 1, &loss_scale, dlogits, 1, (vDSP_Length)(SEQ*CV));

            // Classifier backward
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, CV, 0.0f, dy, SEQ);
            t_cls_bwd += tb_ms(mach_absolute_time() - t0);

            // dEmbed async
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits, CV, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final norm backward
            if (use_metal_norm) {
                if (!metal_layernorm_backward(dx_norm_scratch, grms_final, grms_final_b,
                                              dy, x_cur, rms_final)) return 1;
            } else {
                norm_backward(dx_norm_scratch, grms_final, grms_final_b, dy, x_cur, rms_final, DIM, SEQ);
            }
            memcpy(dy, dx_norm_scratch, SEQ*DIM*4);

            // ===== BACKWARD (28 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
                write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SiLU derivative (Accelerate or fused Metal)
                t0 = mach_absolute_time();
                if (use_metal_silu) {
                    if (!metal_silu_backward(dh1, dh3, ac->h1, ac->h3, dsilu)) return 1;
                } else {
                    int n = HIDDEN*SEQ;
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // sig
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);
                    vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);
                }
                t_silu += tb_ms(mach_absolute_time() - t0);

                // dh1@W1^T + dh3@W3^T → dx_ffn (ANE)
                t0 = mach_absolute_time();
                write_ffn_bwd_w13t_acts(pls[L].ffnBwdW13t_in, dh1, dh3);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dW FFN async
                t0 = mach_absolute_time();
                dispatch_semaphore_wait(dw_slots, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *capt_dffn = (float*)jishui_malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_dh1 = (float*)jishui_malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)jishui_malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                const float *saved_silu = ac->silu_out;
                const float *saved_x2n = ac->x2norm;
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, saved_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, saved_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, saved_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_dh1); free(capt_dh3);
                    dispatch_semaphore_signal(dw_slots);
                });

                // Pre-FFN norm backward
                t0 = mach_absolute_time();
                memset(dx2, 0, SEQ*DIM*4);
                if (use_metal_norm) {
                    if (!metal_layernorm_backward(dx2, gr->rms_ffn, gr->rms_ffn_b,
                                                  dx_ffn, ac->x2, lw[L].rms_ffn)) return 1;
                } else {
                    norm_backward(dx2, gr->rms_ffn, gr->rms_ffn_b, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                }
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward (ANE): alpha*dx2 @ Wo → da[Q_DIM]
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                t0 = mach_absolute_time();
                write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.wotBwd, plr[L].wotBwd)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.wotBwd->ioOut, da_buf, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWo async: gr->Wo[DIM,Q_DIM] += dx2_scaled[DIM,SEQ] @ attn_out^T[SEQ,Q_DIM]
                t0 = mach_absolute_time();
                dispatch_semaphore_wait(dw_slots, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *capt_do = (float*)jishui_malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
                const float *saved_attn = ac->attn_out;
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                1.0f, capt_do, SEQ, saved_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
                    free(capt_do);
                    dispatch_semaphore_signal(dw_slots);
                });

                // GQA: tile K,V from KV_DIM → Q_DIM for SDPA backward
                t0 = mach_absolute_time();
                gqa_tile_kv(k_tiled, ac->K, SEQ);
                gqa_tile_kv(v_tiled, ac->V, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 1: Q[Q_DIM],K_tiled[Q_DIM],V_tiled[Q_DIM],da[Q_DIM] → dV_full[Q_DIM],probs,dp
                t0 = mach_absolute_time();
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,       ac->Q,    Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, Q_DIM,   k_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 2*Q_DIM, v_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 3*Q_DIM, da_buf,   Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval(dk.sdpaBwd1)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q[Q_DIM],K_tiled[Q_DIM] → dQ[Q_DIM],dK_full[Q_DIM]
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, Q_DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,       ac->Q,   Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+Q_DIM, k_tiled, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval(dk.sdpaBwd2)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA backward outputs
                t0 = mach_absolute_time();
                io_read_fp16(dk.sdpaBwd2->ioOut, dq_full, 0,     Q_DIM, SEQ);  // dQ at full HEADS
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_full, Q_DIM, Q_DIM, SEQ);  // dK at full HEADS
                io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,     Q_DIM, SEQ);  // dV at full HEADS
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // GQA: reduce dK, dV from Q_DIM (HEADS) → KV_DIM (KV_HEADS)
                gqa_reduce_kv(dk_buf, dk_full, SEQ);
                gqa_reduce_kv(dv, dv_full, SEQ);
                // dQ stays at Q_DIM — no reduction needed
                memcpy(dq, dq_full, SEQ*Q_DIM*4);

                // RoPE backward on dQ[Q_DIM] and dK[KV_DIM]
                rope_backward_inplace(dq, SEQ, Q_DIM, HD);
                rope_backward_inplace(dk_buf, SEQ, KV_DIM, HD);

                if (L == 0 && step % 10 == 0) {
                    float dqmx, dkmx, dvmx;
                    vDSP_maxmgv(dq, 1, &dqmx, (vDSP_Length)(SEQ*Q_DIM));
                    vDSP_maxmgv(dk_buf, 1, &dkmx, (vDSP_Length)(SEQ*KV_DIM));
                    vDSP_maxmgv(dv, 1, &dvmx, (vDSP_Length)(SEQ*KV_DIM));
                    printf("    L0 sdpa_bwd: |dq|=%.6f |dk|=%.6f |dv|=%.6f\n", dqmx, dkmx, dvmx);
                }

                // dWq/dWk/dWv async
                // dWq[Q_DIM,DIM] += dq[Q_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWk[KV_DIM,DIM] += dk[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWv[KV_DIM,DIM] += dv[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                t0 = mach_absolute_time();
                dispatch_semaphore_wait(dw_slots, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *capt_dq = (float*)jishui_malloc(SEQ*Q_DIM*4); memcpy(capt_dq, dq, SEQ*Q_DIM*4);
                float *capt_dk = (float*)jishui_malloc(SEQ*KV_DIM*4); memcpy(capt_dk, dk_buf, SEQ*KV_DIM*4);
                float *capt_dv = (float*)jishui_malloc(SEQ*KV_DIM*4); memcpy(capt_dv, dv, SEQ*KV_DIM*4);
                const float *saved_xn = ac->xnorm;
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, saved_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, saved_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, saved_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv);
                    dispatch_semaphore_signal(dw_slots);
                });

                // Q backward (ANE): dq[Q_DIM] @ Wq → dx_q[DIM]
                t0 = mach_absolute_time();
                write_q_bwd_acts(pls[L].qBwd_in, dq);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.qBwd, plr[L].qBwd)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // KV backward (ANE): dk[KV_DIM]@Wk + dv[KV_DIM]@Wv → dx_kv[DIM]
                t0 = mach_absolute_time();
                write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.kvBwd, plr[L].kvBwd)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dx_attn = dx_q + dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];

                // Pre-attention norm backward
                t0 = mach_absolute_time();
                if (use_metal_norm) {
                    if (!metal_layernorm_backward(dx_norm_scratch, gr->rms_att, gr->rms_att_b,
                                                  dx_attn, ac->layer_in, lw[L].rms_att)) return 1;
                } else {
                    norm_backward(dx_norm_scratch, gr->rms_att, gr->rms_att_b, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                }
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_norm_scratch[i] + dx2[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;
            accumulated_microsteps++;

            if (step % log_interval == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls_fwd=%.1f xent=%.1f cls_bwd=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd,
                       t_cls_fwd, t_xent, t_cls_bwd, t_cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if (accumulated_microsteps == accum_steps || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float wq0_before = lw[0].Wq[0];
                float gsc = 1.0f / (accumulated_microsteps * loss_scale);

                // Scale gradients
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++) g->Wq[i]*=gsc;
                    for(size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
                    for(size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
                    for(size_t i=0;i<WO_SZ;i++) g->Wo[i]*=gsc;
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){
                        g->rms_att[i]*=gsc; g->rms_att_b[i]*=gsc;
                        g->rms_ffn[i]*=gsc; g->rms_ffn_b[i]*=gsc;
                    }
                }
                for(int i=0;i<DIM;i++){ grms_final[i]*=gsc; grms_final_b[i]*=gsc; }
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // Global gradient norm
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att_b,1,g->rms_att_b,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn_b,1,g->rms_ffn_b,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                }
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(grms_final_b,1,grms_final_b,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); grad_norm_sq+=s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if (!isfinite(grad_norm_sq) || grad_norm_sq < 0.0f || !isfinite(grad_norm)) {
                    printf("Non-finite gradient norm at microstep %d; refusing to update Adam\n", step + 1);
                    return 1;
                }
                uint32_t parity_token = input_tokens[0];
                size_t parity_embed_index = (size_t)parity_token * DIM;
                float parity_grad_wq0 = grads[0].Wq[0];
                float parity_grad_w10 = grads[0].W1[0];
                float parity_grad_last_w20 = grads[NLAYERS-1].W2[0];
                float parity_grad_norm0 = grms_final[0];
                float parity_grad_embed0 = gembed[parity_embed_index];
                if ((step+1) % 10 == 0) {
                    float attn_sq=0, ffn_sq=0, embed_sq=0;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L]; float s;
                        vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); attn_sq+=s;
                        vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); ffn_sq+=s;
                    }
                    { float s;
                      vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); embed_sq=s;
                    }
                    printf("  grad_norm=%.4f  attn=%.4f ffn=%.4f embed=%.4f\n",
                           grad_norm, sqrtf(attn_sq), sqrtf(ffn_sq), sqrtf(embed_sq));
                }

                // Gradient clipping
                float clip_scale = 1.0f;
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_att_b,1,&clip_scale,g->rms_att_b,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn_b,1,&clip_scale,g->rms_ffn_b,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(grms_final_b,1,&clip_scale,grms_final_b,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
                }

                // Warmup and cosine are expressed in optimizer updates, matching MLX.
                adam_t++;
                if (warmup_steps > 0 && adam_t <= warmup_steps) {
                    lr = max_lr * ((float)adam_t) / warmup_steps;
                } else {
                    int decay_updates = total_optimizer_steps - warmup_steps;
                    int decay_index = adam_t - warmup_steps - 1;
                    // MLX cosine_decay evaluates local steps [0, decay_updates-1]
                    // for a run with decay_updates updates. The final update is
                    // therefore just above min_lr; min_lr is reached only if a
                    // caller continues beyond the configured budget.
                    float decay_ratio = decay_updates <= 0 ? 1.0f :
                        fminf(1.0f, fmaxf(0.0f, (float)decay_index / (float)decay_updates));
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Adam update
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_att_b, g->rms_att_b, &la[L].rms_att_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn_b, g->rms_ffn_b, &la[L].rms_ffn_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);

                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);

                    // Re-stage weights
                    stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
                    stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
                    stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
                    stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
                    stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
                    stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
                    stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
                    stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(rms_final_b, grms_final_b, &arms_final_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                free(cembed);
                cembed = vocab_compact_embed(embed, &vm, DIM);

                if (parity_report) {
                    fprintf(parity_report,
                            "{\"record\":\"update\",\"optimizer_step\":%d,"
                            "\"microstep\":%d,\"probe_token\":%u,\"loss\":%.9e,"
                            "\"lr\":%.9e,\"grad_norm\":%.9e,\"clip_scale\":%.9e,"
                            "\"grad_l0_wq0\":%.9e,\"grad_l0_w1_0\":%.9e,"
                            "\"grad_last_w2_0\":%.9e,\"grad_final_norm0\":%.9e,"
                            "\"grad_embed0\":%.9e,\"weight_l0_wq0\":%.9e,"
                            "\"weight_l0_w1_0\":%.9e,\"weight_last_w2_0\":%.9e,"
                            "\"weight_final_norm0\":%.9e,\"weight_embed0\":%.9e,"
                            "\"adam_m_l0_wq0\":%.9e,\"adam_v_l0_wq0\":%.9e,"
                            "\"adam_m_l0_w1_0\":%.9e,\"adam_v_l0_w1_0\":%.9e,"
                            "\"adam_m_last_w2_0\":%.9e,\"adam_v_last_w2_0\":%.9e,"
                            "\"adam_m_final_norm0\":%.9e,\"adam_v_final_norm0\":%.9e,"
                            "\"adam_m_embed0\":%.9e,\"adam_v_embed0\":%.9e}\n",
                            adam_t, step + 1, parity_token, last_loss, lr, grad_norm,
                            clip_scale, parity_grad_wq0, parity_grad_w10,
                            parity_grad_last_w20, parity_grad_norm0, parity_grad_embed0,
                            lw[0].Wq[0], lw[0].W1[0], lw[NLAYERS-1].W2[0],
                            rms_final[0], embed[parity_embed_index],
                            la[0].Wq.m[0], la[0].Wq.v[0],
                            la[0].W1.m[0], la[0].W1.v[0],
                            la[NLAYERS-1].W2.m[0], la[NLAYERS-1].W2.v[0],
                            arms_final.m[0], arms_final.v[0],
                            aembed.m[parity_embed_index], aembed.v[parity_embed_index]);
                }

                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM*4);
                memset(grms_final_b, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);
                accumulated_microsteps = 0;

                if (step % log_interval == 0 || step == total_steps - 1) {
                    printf("  optimizer_step=%d lr=%.2e Wq[0] delta=%+.6e\n",
                           adam_t, lr, lw[0].Wq[0] - wq0_before);
                }

                // Checkpoints are optimizer-consistent, so the interval is in
                // completed Adam updates rather than raw microsteps.
                if (checkpoint_enabled && save_interval > 0 && adam_t % save_interval == 0) {
                    if (last_loss < best_loss) best_loss = last_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    bool checkpoint_ok = save_checkpoint(ckpt_path, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        keep_checkpoints, accum_steps, warmup_steps, max_lr, min_lr_frac,
                        adam_b1, adam_b2, adam_eps, wd, grad_clip, loss_scale, res_alpha,
                        lw, la, rms_final, rms_final_b,
                        &arms_final, &arms_final_b, embed, &aembed);
                    if (checkpoint_ok) last_checkpoint_step = step + 1;
                    printf("  [%s under %s, keep=%d, best_loss=%.4f]\n",
                           checkpoint_ok ? "ckpt saved" : "checkpoint save failed",
                           ckpt_path, keep_checkpoints, best_loss);
                    if (!checkpoint_ok) {
                        training_failed = true;
                        break;
                    }
                }
            }
        }

        // Always leave a resumable point when a short run ends between save
        // intervals.  The file is written only after all pending dW work has
        // completed, so it represents a fully updated optimizer state.
        int final_step = start_step + total_steps_done;
        if (!training_failed && checkpoint_enabled && total_steps_done > 0 &&
            final_step != last_checkpoint_step) {
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            double wall = tb_ms(mach_absolute_time() - t_wall_start);
            if (save_checkpoint(ckpt_path, final_step, total_steps,
                                lr, last_loss, total_train_ms + cum_train,
                                wall + cum_wall, total_steps_done + cum_steps, adam_t,
                                keep_checkpoints, accum_steps, warmup_steps, max_lr, min_lr_frac,
                                adam_b1, adam_b2, adam_eps, wd, grad_clip, loss_scale, res_alpha,
                                lw, la, rms_final, rms_final_b,
                                &arms_final, &arms_final_b, embed, &aembed)) {
                printf("  [final ckpt saved under %s, keep=%d]\n", ckpt_path, keep_checkpoints);
            } else {
                printf("  [final checkpoint save failed]\n");
                training_failed = true;
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms,
               total_steps_done ? total_train_ms/total_steps_done : 0.0);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);
        if (parity_report && fclose(parity_report) != 0)
            perror("parity report close");

        // Cleanup
        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
        if (use_metal_norm || use_metal_silu || use_metal_io) metal_ops_shutdown();
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free_per_layer(pls, plr);
        free_dynamic_kernels(&dk);
        free(da_buf); free(k_tiled); free(v_tiled);
        free(dq_full); free(dk_full); free(dv_full);
        free(dq); free(dk_buf); free(dv);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(dx_attn); free(dx_kv); free(dx2_scaled);
        free(x_cur); free(x_final); free(xnorm_buf); free(dx_norm_scratch);
        free(logits); free(dlogits);
        free(dh1); free(dh3); free(dsilu); free(silu_tmp); free(silu_tmp2);
        free(rms_final); free(rms_final_b); free(grms_final); free(grms_final_b);
        adam_free(&arms_final); adam_free(&arms_final_b); adam_free(&aembed);
        free(embed); free(gembed); free(cembed); free(gcembed);
        free(vm.full_to_compact); free(vm.compact_to_full);
        if (data_mode == ANE_DATA_INDEX) {
            ane_index_free(&train_index);
        } else {
            if (data_mapping && data_mapping != MAP_FAILED) munmap(data_mapping, data_len);
            if (data_fd >= 0) close(data_fd);
        }
        if (training_failed) exit_code = 1;
    }
    return exit_code;
}
