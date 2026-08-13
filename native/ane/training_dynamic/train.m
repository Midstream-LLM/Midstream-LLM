// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "ane_data.h"
#include "metal_ops.h"
#include <dirent.h>
#include <errno.h>
#include <time.h>

typedef enum {
    ANE_DATA_FLAT = 0,
    ANE_DATA_RECORDS = 1,
    ANE_DATA_INDEX = 2,
} AneDataMode;

static bool hybrid_tensor_map_open(HybridTensorMap *map, const char *path,
                                   uint32_t expected_kind, bool writable) {
    memset(map, 0, sizeof(*map));
    map->fd = -1;
    int flags = writable ? O_RDWR : O_RDONLY;
    map->fd = open(path, flags);
    if (map->fd < 0) { perror(path); return false; }
    struct stat st;
    size_t expected_bytes = HYBRID_HEADER_BYTES + (size_t)HYBRID_PARAM_COUNT * 2;
    if (fstat(map->fd, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size != (uint64_t)expected_bytes) {
        printf("invalid hybrid tensor file size: %s\n", path);
        close(map->fd); map->fd = -1; return false;
    }
    int protection = PROT_READ | (writable ? PROT_WRITE : 0);
    map->mapping = mmap(NULL, expected_bytes, protection, MAP_SHARED, map->fd, 0);
    if (map->mapping == MAP_FAILED) {
        map->mapping = NULL; perror("hybrid mmap"); close(map->fd); map->fd = -1;
        return false;
    }
    const HybridTensorHeader *header = (const HybridTensorHeader *)map->mapping;
    if (memcmp(header->magic, HYBRID_MAGIC, 8) != 0 ||
        header->version != HYBRID_VERSION ||
        header->header_bytes != HYBRID_HEADER_BYTES || header->kind != expected_kind ||
        header->parameter_count != HYBRID_PARAM_COUNT || header->n_layers != NLAYERS ||
        header->vocab_size != VOCAB || header->dim != DIM ||
        header->hidden_dim != HIDDEN || header->n_heads != HEADS ||
        header->kv_heads != KV_HEADS) {
        printf("hybrid tensor header does not match %s: %s\n", MODEL_NAME, path);
        munmap(map->mapping, expected_bytes); close(map->fd);
        map->mapping = NULL; map->fd = -1; return false;
    }
    map->bytes = expected_bytes;
    map->values = (_Float16 *)((uint8_t *)map->mapping + HYBRID_HEADER_BYTES);
    return true;
}

static void hybrid_tensor_map_close(HybridTensorMap *map) {
    if (map->mapping) munmap(map->mapping, map->bytes);
    if (map->fd >= 0) close(map->fd);
    memset(map, 0, sizeof(*map)); map->fd = -1;
}

static bool hybrid_control_open(HybridControl **control, int *fd, const char *path) {
    *control = NULL; *fd = open(path, O_RDWR);
    if (*fd < 0) { perror(path); return false; }
    struct stat st;
    if (fstat(*fd, &st) != 0 || st.st_size != HYBRID_CONTROL_BYTES) {
        printf("invalid hybrid control file size: %s\n", path);
        close(*fd); *fd = -1; return false;
    }
    void *mapping = mmap(NULL, HYBRID_CONTROL_BYTES, PROT_READ|PROT_WRITE,
                         MAP_SHARED, *fd, 0);
    if (mapping == MAP_FAILED) {
        perror("hybrid control mmap"); close(*fd); *fd = -1; return false;
    }
    HybridControl *c = (HybridControl *)mapping;
    if (memcmp(c->magic, HYBRID_MAGIC, 8) != 0 || c->version != HYBRID_VERSION ||
        c->bytes != HYBRID_CONTROL_BYTES) {
        printf("hybrid control header does not match\n");
        munmap(mapping, HYBRID_CONTROL_BYTES); close(*fd); *fd = -1; return false;
    }
    *control = c;
    return true;
}

static void hybrid_control_close(HybridControl *control, int fd) {
    if (control) munmap(control, HYBRID_CONTROL_BYTES);
    if (fd >= 0) close(fd);
}

static void transpose_weight(float *dst, const float *src, int rows, int cols);
static bool hybrid_import_weights(const HybridTensorMap *map, LayerWeights *lw,
                                  float *rms_final, float *rms_final_b, float *embed);

static uint32_t hybrid_control_load(const volatile uint32_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

static void hybrid_control_store(volatile uint32_t *value, uint32_t next) {
    __atomic_store_n(value, next, __ATOMIC_RELEASE);
}

static void hybrid_control_set_status(HybridControl *control, uint32_t status) {
    hybrid_control_store(&control->status, status);
    msync((void *)control, HYBRID_CONTROL_BYTES, MS_ASYNC);
}

static bool hybrid_wait_for_generation(HybridControl *control, uint32_t previous,
                                       uint32_t *generation, uint32_t *microsteps,
                                       uint64_t *record_base,
                                       int timeout_seconds) {
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    for (;;) {
        uint32_t command = hybrid_control_load(&control->command);
        uint32_t current = hybrid_control_load(&control->generation);
        if (command == HYBRID_CMD_STOP) return false;
        if (command == HYBRID_CMD_RUN && current > previous) {
            uint32_t count = hybrid_control_load(&control->microsteps);
            if (count == 0) {
                printf("Hybrid generation %u requested zero microsteps\n", current);
                hybrid_control_set_status(control, HYBRID_STATUS_ERROR);
                return false;
            }
            *generation = current;
            *microsteps = count;
            *record_base = control->record_base;
            hybrid_control_set_status(control, HYBRID_STATUS_RUNNING);
            return true;
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - started.tv_sec >= timeout_seconds) {
            printf("Timed out waiting for hybrid generation %u\n", previous + 1);
            hybrid_control_set_status(control, HYBRID_STATUS_ERROR);
            return false;
        }
        usleep(1000);
    }
}

static bool hybrid_restage_weights(const HybridTensorMap *weights, LayerWeights *lw,
                                   float *rms_final, float *rms_final_b, float *embed,
                                   float *Wqt_buf[], float *Wkt_buf[], float *Wvt_buf[],
                                   float *Wot_buf[], float *W1t_buf[],
                                   float *W3t_buf[], PerLayerSurfaces *pls) {
    if (!hybrid_import_weights(weights, lw, rms_final, rms_final_b, embed)) return false;
    for (int L = 0; L < NLAYERS; L++) {
        transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
        transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
        transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
        transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
        transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
        transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        if (!stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]) ||
            !stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]) ||
            !stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2) ||
            !stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2) ||
            !stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3) ||
            !stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo) ||
            !stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq) ||
            !stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv))
            return false;
        if (pls[L].qkvBwd_in)
            if (!stage_qkv_bwd_weights(pls[L].qkvBwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv))
                return false;
    }
    return true;
}

static void hybrid_copy_f16_to_f32(float *destination, const _Float16 **source,
                                   size_t count) {
    cvt_f16_f32(destination, *source, (int)count);
    *source += count;
}

static void hybrid_copy_f32_to_f16(_Float16 **destination, const float *source,
                                   size_t count) {
    cvt_f32_f16(*destination, source, (int)count);
    *destination += count;
}

// Canonical order mirrors mlx.utils.tree_flatten(model.parameters()).
static bool hybrid_import_weights(const HybridTensorMap *map, LayerWeights *lw,
                                  float *rms_final, float *rms_final_b, float *embed) {
    const _Float16 *source = map->values;
    hybrid_copy_f16_to_f32(embed, &source, (size_t)VOCAB*DIM);
    for (int L = 0; L < NLAYERS; L++) {
        hybrid_copy_f16_to_f32(lw[L].rms_att, &source, DIM);
        hybrid_copy_f16_to_f32(lw[L].rms_att_b, &source, DIM);
        hybrid_copy_f16_to_f32(lw[L].Wq, &source, WQ_SZ);
        hybrid_copy_f16_to_f32(lw[L].Wk, &source, WK_SZ);
        hybrid_copy_f16_to_f32(lw[L].Wv, &source, WV_SZ);
        hybrid_copy_f16_to_f32(lw[L].Wo, &source, WO_SZ);
        hybrid_copy_f16_to_f32(lw[L].rms_ffn, &source, DIM);
        hybrid_copy_f16_to_f32(lw[L].rms_ffn_b, &source, DIM);
        hybrid_copy_f16_to_f32(lw[L].W1, &source, W1_SZ); // gate_proj
        hybrid_copy_f16_to_f32(lw[L].W3, &source, W3_SZ); // up_proj
        hybrid_copy_f16_to_f32(lw[L].W2, &source, W2_SZ); // down_proj
    }
    hybrid_copy_f16_to_f32(rms_final, &source, DIM);
    hybrid_copy_f16_to_f32(rms_final_b, &source, DIM);
    return (uint64_t)(source - map->values) == HYBRID_PARAM_COUNT;
}

static bool hybrid_export_gradients(HybridTensorMap *map, LayerGrads *grads,
                                    float *grms_final, float *grms_final_b,
                                    float *gembed) {
    _Float16 *destination = map->values;
    hybrid_copy_f32_to_f16(&destination, gembed, (size_t)VOCAB*DIM);
    for (int L = 0; L < NLAYERS; L++) {
        hybrid_copy_f32_to_f16(&destination, grads[L].rms_att, DIM);
        hybrid_copy_f32_to_f16(&destination, grads[L].rms_att_b, DIM);
        hybrid_copy_f32_to_f16(&destination, grads[L].Wq, WQ_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].Wk, WK_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].Wv, WV_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].Wo, WO_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].rms_ffn, DIM);
        hybrid_copy_f32_to_f16(&destination, grads[L].rms_ffn_b, DIM);
        hybrid_copy_f32_to_f16(&destination, grads[L].W1, W1_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].W3, W3_SZ);
        hybrid_copy_f32_to_f16(&destination, grads[L].W2, W2_SZ);
    }
    hybrid_copy_f32_to_f16(&destination, grms_final, DIM);
    hybrid_copy_f32_to_f16(&destination, grms_final_b, DIM);
    atomic_thread_fence(memory_order_release);
    return (uint64_t)(destination - map->values) == HYBRID_PARAM_COUNT;
}

static bool benchmark_barrier(const char *ready_path, const char *start_path,
                              int timeout_seconds) {
    if (!ready_path && !start_path) return true;
    if (!ready_path || !start_path) {
        printf("--ready-file and --start-file must be used together\n");
        return false;
    }
    int fd = open(ready_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        printf("Cannot create ready file %s: %s\n", ready_path, strerror(errno));
        return false;
    }
    char message[64];
    int length = snprintf(message, sizeof(message), "pid=%d\n", getpid());
    bool ok = length > 0 && write(fd, message, (size_t)length) == length;
    if (close(fd) != 0) ok = false;
    if (!ok) {
        printf("Cannot write ready file %s\n", ready_path);
        return false;
    }

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    while (access(start_path, F_OK) != 0) {
        if (errno != ENOENT) {
            printf("Cannot inspect start file %s: %s\n", start_path, strerror(errno));
            return false;
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - started.tv_sec >= timeout_seconds) {
            printf("Timed out waiting for start file %s\n", start_path);
            return false;
        }
        usleep(10000);
    }
    return true;
}

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → transient DIM buffer (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T → dx_ffn (HIDDEN → DIM)
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
    Kern *qkvBwd;      // optional MHA fusion: (dq,dk,dv) → (dx_q,dx_kv)
} DynLayerKernels;

static void free_dynamic_kernels(DynLayerKernels *dk) {
    free_kern(dk->sdpaFwd); free_kern(dk->woFwd); free_kern(dk->ffnFused);
    free_kern(dk->ffnBwdW2t); free_kern(dk->ffnBwdW13t); free_kern(dk->wotBwd);
    free_kern(dk->sdpaBwd1); free_kern(dk->sdpaBwd2);
    free_kern(dk->qBwd); free_kern(dk->kvBwd); free_kern(dk->qkvBwd);
    memset(dk, 0, sizeof(*dk));
}

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ANE normally returns finite fp16 values, but an experimental graph must not
// be allowed to inject NaN/Inf into the residual stream or optimizer state.
static bool f32_buffer_all_finite(const float *values, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (!isfinite(values[i])) return false;
    return true;
}

// Optional, opt-in diagnostics for numerical parity work.  The normal
// trainer never opens these files, so this has no effect on production
// execution or checkpoint contents.  Each tensor is written as native
// little-endian fp32 with the same channel-first layout used by the native
// kernels.
static void trace_dump_f32(const char *directory, const char *name,
                           const float *values, size_t count) {
    if (!directory || !name || !values) return;
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%s/%s.f32", directory, name);
    if (length <= 0 || (size_t)length >= sizeof(path)) return;
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "trace open failed: %s\n", path);
        return;
    }
    size_t written = fwrite(values, sizeof(float), count, file);
    if (written != count)
        fprintf(stderr, "trace write failed: %s (%zu/%zu)\n", path, written, count);
    fclose(file);
}

static void trace_norm(FILE *file, int step, int layer, const char *phase,
                       const char *name, const float *values, size_t count) {
    if (!file || !phase || !name || !values || count == 0) return;
    float minimum = values[0], maximum = values[0];
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        float value = values[i];
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        sum_sq += (double)value * (double)value;
    }
    fprintf(file, "%d,%d,%s,%s,%.9g,%.9g,%.9g,%.9g\n", step, layer,
            phase, name, sqrt(sum_sq), minimum, maximum, (double)count);
    fflush(file);
}

static void trace_layer_f32(const char *directory, int step, int layer,
                            const char *phase, const char *name,
                            const float *values, size_t count) {
    if (!directory || !phase || !name) return;
    char label[PATH_MAX];
    int length = snprintf(label, sizeof(label), "step_%08d_l%02d_%s_%s",
                          step, layer, phase, name);
    if (length <= 0 || (size_t)length >= sizeof(label)) return;
    trace_dump_f32(directory, label, values, count);
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk, bool request_qkv_fused) {
    memset(dk, 0, sizeof(*dk));
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM;

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

#if Q_DIM == KV_DIM && HEADS == KV_HEADS
    if (request_qkv_fused) {
        // Keep the already-validated split kernels alive. A MIL compiler or
        // ANE loader rejection of this experimental graph is a normal
        // fallback condition, not a fatal trainer error.
        printf("  Compiling qkvBwd (experimental MHA fusion)...\n");
        dk->qkvBwd = compile_kern_mil_w(gen_qkv_bwd_dynamic(), @{},
            Q_DIM*QKV_BWD_SP*2, 2*DIM*SEQ*2);
        if (!dk->qkvBwd)
            printf("  qkvBwd unavailable; falling back to qBwd + kvBwd\n");
    }
#else
    (void)request_qkv_fused;
#endif

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
        float ffn_bwd_scale = 1.0f;
        float ffn_w13_bwd_scale = 1.0f;
        float sdpa_bwd_scale = 1.0f;
        float qkv_bwd_scale = 1.0f;
        float wo_bwd_scale = 1.0f;
        float res_alpha = RES_ALPHA;
        float min_lr_frac = 0.1f;

        bool do_resume = false, from_scratch = false, record_mode = false;
        bool checkpoint_enabled = true;
        bool use_metal_norm = false, use_metal_silu = false, use_metal_io = false;
        bool metal_shadow = false;
        bool request_qkv_fused = false;
        bool qkv_fused_shadow = false;
        bool legacy_sdpa_bwd = false;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *index_path = NULL;
        const char *data_dir = DEFAULT_DATA_PATH;
        const char *ckpt_path = CKPT_PATH;
        const char *parity_path = NULL;
        const char *ready_path = NULL, *start_path = NULL;
        int barrier_timeout = 300;
        bool hybrid_mode = false;
        const char *hybrid_weights_path = NULL;
        const char *hybrid_gradients_path = NULL;
        const char *hybrid_control_path = NULL;
        int hybrid_timeout = 300;
        const char *trace_dir = NULL;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [--scratch|--resume] [--index INDEX --data-dir DATASET | --records --data FILE | --data FILE]\n", argv[0]);
                printf("       --steps N | --optimizer-steps N  --accum N  --save-interval N(updates)  --keep-checkpoints N\n");
                printf("       --checkpoint PREFIX | --no-checkpoint  --log-interval N  --lr LR  --warmup N(updates)\n");
                printf("       --loss-scale X (global fp16 backward loss scale; default 256)\n");
                printf("       --metal-norm  --metal-silu  --metal-io  --metal-shadow  --parity-report FILE\n");
                printf("       --ffn-bwd-scale X (diagnostic: scale dffn before ANE W2 backward)\n");
                printf("       --ffn-w13-bwd-scale X (diagnostic: scale dh1/dh3 before ANE W1/W3 backward)\n");
                printf("       --sdpa-bwd-scale X (diagnostic: scale da through ANE SDPA backward)\n");
                printf("       --qkv-bwd-scale X (diagnostic: scale dq/dk/dv through ANE projection backward)\n");
                printf("       --wo-bwd-scale X (diagnostic: scale dx2 through ANE Wo backward)\n");
                printf("       --npu-fuse-qkv-bwd [--npu-fuse-qkv-bwd-shadow]\n");
                printf("       --legacy-sdpa-bwd (diagnostic: pre-optimization SDPA I/O path)\n");
                printf("       --ready-file FILE --start-file FILE [--barrier-timeout SECONDS]\n");
                printf("       --hybrid-weights FILE --hybrid-gradients FILE --hybrid-control FILE\n");
                printf("       --trace-dir DIRECTORY (optional fp32 parity diagnostics)\n");
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
            else if (strcmp(argv[i], "--loss-scale") == 0 && i+1<argc) loss_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--ffn-bwd-scale") == 0 && i+1<argc) ffn_bwd_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--ffn-w13-bwd-scale") == 0 && i+1<argc) ffn_w13_bwd_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--sdpa-bwd-scale") == 0 && i+1<argc) sdpa_bwd_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--qkv-bwd-scale") == 0 && i+1<argc) qkv_bwd_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--wo-bwd-scale") == 0 && i+1<argc) wo_bwd_scale = atof(argv[++i]);
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
            else if (strcmp(argv[i], "--npu-fuse-qkv-bwd") == 0) request_qkv_fused = true;
            else if (strcmp(argv[i], "--npu-fuse-qkv-bwd-shadow") == 0) {
                request_qkv_fused = true;
                qkv_fused_shadow = true;
            }
            else if (strcmp(argv[i], "--legacy-sdpa-bwd") == 0) legacy_sdpa_bwd = true;
            else if (strcmp(argv[i], "--ready-file") == 0 && i+1<argc) ready_path = argv[++i];
            else if (strcmp(argv[i], "--start-file") == 0 && i+1<argc) start_path = argv[++i];
            else if (strcmp(argv[i], "--barrier-timeout") == 0 && i+1<argc) barrier_timeout = atoi(argv[++i]);
            else if (strcmp(argv[i], "--hybrid-weights") == 0 && i+1<argc) {
                hybrid_mode = true; hybrid_weights_path = argv[++i];
            }
            else if (strcmp(argv[i], "--hybrid-gradients") == 0 && i+1<argc) {
                hybrid_mode = true; hybrid_gradients_path = argv[++i];
            }
            else if (strcmp(argv[i], "--hybrid-control") == 0 && i+1<argc) {
                hybrid_mode = true; hybrid_control_path = argv[++i];
            }
            else if (strcmp(argv[i], "--hybrid-timeout") == 0 && i+1<argc) hybrid_timeout = atoi(argv[++i]);
            else if (strcmp(argv[i], "--trace-dir") == 0 && i+1<argc) trace_dir = argv[++i];
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
            save_interval < 0 || keep_checkpoints < 1 || barrier_timeout < 1 || hybrid_timeout < 1 ||
            !isfinite(max_lr) || max_lr <= 0.0f ||
            !isfinite(grad_clip) || grad_clip < 0.0f ||
            !isfinite(loss_scale) || loss_scale <= 0.0f ||
            !isfinite(ffn_bwd_scale) || ffn_bwd_scale <= 0.0f ||
            !isfinite(ffn_w13_bwd_scale) || ffn_w13_bwd_scale <= 0.0f ||
            !isfinite(sdpa_bwd_scale) || sdpa_bwd_scale <= 0.0f ||
            !isfinite(qkv_bwd_scale) || qkv_bwd_scale <= 0.0f ||
            !isfinite(wo_bwd_scale) || wo_bwd_scale <= 0.0f) {
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
        if (hybrid_mode && (!hybrid_weights_path || !hybrid_gradients_path || !hybrid_control_path)) {
            printf("hybrid mode requires --hybrid-weights, --hybrid-gradients, and --hybrid-control\n");
            return 1;
        }
        if (hybrid_mode && !record_mode) {
            printf("hybrid mode currently requires --records\n");
            return 1;
        }
        if (hybrid_mode) {
            // In hybrid mode MLX owns accumulation, clipping, AdamW and the
            // schedule. The native worker computes one unscaled gradient per
            // control generation and never mutates the optimizer state.
            checkpoint_enabled = false;
            accum_steps = 1;
            warmup_steps = 0;
            // The coordinator ends the worker with HYBRID_CMD_STOP. Keep the
            // local loop open until then; this also makes --steps harmless for
            // a protocol run instead of silently truncating the handshake.
            total_steps = INT_MAX;
        }
        FILE *parity_report = NULL;
        FILE *trace_norms = NULL;
        if (trace_dir) {
            if (mkdir(trace_dir, 0700) != 0 && errno != EEXIST) {
                fprintf(stderr, "trace directory creation failed: %s (%s)\n",
                        trace_dir, strerror(errno));
                return 1;
            }
            char trace_path[PATH_MAX];
            int trace_length = snprintf(trace_path, sizeof(trace_path),
                                        "%s/norms.csv", trace_dir);
            if (trace_length <= 0 || (size_t)trace_length >= sizeof(trace_path) ||
                !(trace_norms = fopen(trace_path, "w"))) {
                fprintf(stderr, "trace norms open failed: %s\n", trace_dir);
                return 1;
            }
            fprintf(trace_norms, "step,layer,phase,name,l2,min,max,count\n");
        }
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
                    "\"metal_silu\":%s,\"metal_io\":%s,\"metal_shadow\":%s,"
                    "\"loss_scale\":%.9g,"
                    "\"ffn_bwd_scale\":%.9g,\"ffn_w13_bwd_scale\":%.9g,"
                    "\"wo_bwd_scale\":%.9g,\"sdpa_bwd_scale\":%.9g,"
                    "\"qkv_bwd_scale\":%.9g,\"legacy_sdpa_bwd\":%s,"
                    "\"qkv_fused\":%s,\"qkv_fused_shadow\":%s}\n",
                    MODEL_NAME, SEQ, NLAYERS, accum_steps,
                    use_metal_norm ? "true" : "false",
                    use_metal_silu ? "true" : "false",
                    use_metal_io ? "true" : "false",
                    metal_shadow ? "true" : "false",
                    loss_scale,
                    ffn_bwd_scale, ffn_w13_bwd_scale, wo_bwd_scale,
                    sdpa_bwd_scale, qkv_bwd_scale,
                    legacy_sdpa_bwd ? "true" : "false",
                    request_qkv_fused ? "true" : "false",
                    qkv_fused_shadow ? "true" : "false");
        }
        if ((use_metal_norm || use_metal_silu || use_metal_io) && !metal_ops_init()) return 1;
        metal_ops_set_shadow(metal_shadow);
        if (!rope_backward_init(SEQ, HD)) return 1;
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

        HybridTensorMap hybrid_weights = {.fd = -1}, hybrid_gradients = {.fd = -1};
        HybridControl *hybrid_control = NULL;
        int hybrid_control_fd = -1;

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
            printf("Kernels: dynamic set (actual count is printed after compilation)\n");
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

        if (hybrid_mode) {
            if (!hybrid_tensor_map_open(&hybrid_weights, hybrid_weights_path,
                                        HYBRID_KIND_WEIGHTS, false) ||
                !hybrid_tensor_map_open(&hybrid_gradients, hybrid_gradients_path,
                                        HYBRID_KIND_GRADIENTS, true) ||
                !hybrid_control_open(&hybrid_control, &hybrid_control_fd,
                                     hybrid_control_path)) {
                hybrid_tensor_map_close(&hybrid_weights);
                hybrid_tensor_map_close(&hybrid_gradients);
                hybrid_control_close(hybrid_control, hybrid_control_fd);
                return 1;
            }
            if (!hybrid_import_weights(&hybrid_weights, lw, rms_final, rms_final_b, embed)) {
                printf("hybrid weight import failed\n");
                hybrid_tensor_map_close(&hybrid_weights);
                hybrid_tensor_map_close(&hybrid_gradients);
                hybrid_control_close(hybrid_control, hybrid_control_fd);
                return 1;
            }
            printf("Hybrid worker: shared fp16 weights/gradients enabled\n");
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)jishui_malloc(WQ_SZ*4); Wkt_buf[L]=(float*)jishui_malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)jishui_malloc(WV_SZ*4); Wot_buf[L]=(float*)jishui_malloc(WO_SZ*4);
            W1t_buf[L]=(float*)jishui_malloc(W1_SZ*4);
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
            // Hybrid records are filled by the coordinator between control
            // generations. A shared read-only mapping observes those flushed
            // updates without reopening the fixed-size record stream.
            int map_flags = hybrid_mode ? MAP_SHARED : MAP_PRIVATE;
            data_mapping = mmap(NULL, data_len, PROT_READ, map_flags, data_fd, 0);
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

        if (hybrid_mode && CV != VOCAB) {
            printf("hybrid mode requires identity vocabulary mapping\n");
            return 1;
        }
        // Jishui's production index uses the identity vocabulary map. Keep
        // the tied embedding as the classifier matrix instead of rebuilding a
        // full fp32 copy after every Adam update. Compact-vocab models retain
        // the old separate matrix and scatter path.
        bool identity_vocab = (CV == VOCAB);
        float *cembed = identity_vocab ? embed : vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = identity_vocab ? NULL : (float*)jishui_calloc((size_t)CV*DIM, 4);

        // ===== Compile all kernels ONCE =====
        printf("Compiling dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk, request_qkv_fused)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled %d kernels in %.0fms (shared across all %d layers)\n",
               g_compile_count, compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        printf("Allocating per-layer IOSurfaces...\n");
        PerLayerSurfaces pls[NLAYERS] = {0};
        PerLayerRequests plr[NLAYERS] = {0};
        // qkvBwd is optional.  A surface/request allocation failure must
        // disable only this experiment and leave the validated split path
        // usable; the rest of the per-layer resources remain mandatory.
        const bool mha_shape = (Q_DIM == KV_DIM && HEADS == KV_HEADS);
        bool qkv_fusion_available = request_qkv_fused && mha_shape && dk.qkvBwd;
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
            pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);
            pls[L].qkvBwd_in     = qkv_fusion_available
                ? make_surface(Q_DIM*QKV_BWD_SP*2) : NULL;

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
            plr[L].qkvBwd    = (pls[L].qkvBwd_in && qkv_fusion_available)
                ? make_request(dk.qkvBwd, pls[L].qkvBwd_in) : NULL;
            if (qkv_fusion_available &&
                (!pls[L].qkvBwd_in || !plr[L].qkvBwd)) {
                printf("Optional qkvBwd resource allocation failed at layer %d; using split path\n", L);
                if (plr[L].qkvBwd) {
                    CFRelease(plr[L].qkvBwd);
                    plr[L].qkvBwd = NULL;
                }
                if (pls[L].qkvBwd_in) {
                    CFRelease(pls[L].qkvBwd_in);
                    pls[L].qkvBwd_in = NULL;
                }
                // Release optional resources allocated for earlier layers as
                // well, so a partial allocation cannot retain a large ANE
                // surface set after the fallback is selected.
                for (int j = 0; j < L; j++) {
                    if (plr[j].qkvBwd) {
                        CFRelease(plr[j].qkvBwd);
                        plr[j].qkvBwd = NULL;
                    }
                    if (pls[j].qkvBwd_in) {
                        CFRelease(pls[j].qkvBwd_in);
                        pls[j].qkvBwd_in = NULL;
                    }
                }
                qkv_fusion_available = false;
            }
            if (!pls[L].sdpaFwd_in || !pls[L].woFwd_in || !pls[L].ffnFused_in ||
                !pls[L].ffnBwdW2t_in || !pls[L].ffnBwdW13t_in || !pls[L].wotBwd_in ||
                !pls[L].qBwd_in || !pls[L].kvBwd_in ||
                !plr[L].sdpaFwd ||
                !plr[L].woFwd || !plr[L].ffnFused || !plr[L].ffnBwdW2t ||
                !plr[L].ffnBwdW13t || !plr[L].wotBwd || !plr[L].qBwd || !plr[L].kvBwd) {
                printf("Per-layer IOSurface/request allocation failed at layer %d\n", L);
                // This can occur after earlier layers already own ANE
                // surfaces/requests. Release the complete partial set before
                // returning so a failed optional experiment cannot strand a
                // large allocation in the process.
                free_per_layer(pls, plr);
                return 1;
            }
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            if (!stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]) ||
                !stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]) ||
                !stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2) ||
                !stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2) ||
                !stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3) ||
                !stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo) ||
                !stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq) ||
                !stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv) ||
                (pls[L].qkvBwd_in &&
                 !stage_qkv_bwd_weights(pls[L].qkvBwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv))) {
                printf("Per-layer IOSurface weight staging failed at layer %d\n", L);
                free_per_layer(pls, plr);
                return 1;
            }
        }
        printf("Per-layer weight staging complete\n\n");

        bool use_qkv_fused = qkv_fusion_available;
        bool qkv_fusion_disabled = false;
        if (request_qkv_fused) {
            printf("QKV backward fusion: %s%s\n",
                   use_qkv_fused
                       ? "enabled" : "unavailable; split fallback",
                   qkv_fused_shadow ? " (shadow split reference)" : "");
        }

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
        // `acts[0].layer_in` owns the embedding output. Each following layer
        // writes its result directly into `acts[L+1].layer_in`, so forward no
        // longer copies every residual stream into a second buffer. The final
        // residual remains separate from the final norm output.
        float *x_cur = acts[0].layer_in;
        float *final_residual = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *x_final = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        // Both norm backward implementations write every element. Keep this
        // as a second gradient buffer so the reverse pass can swap pointers
        // instead of copying a full [SEQ,DIM] result back into `dy`.
        float *dx_norm_scratch = (float*)jishui_aligned_malloc(SEQ*DIM*4);
        float *logits = (float*)jishui_malloc(SEQ*CV*4);
        float *dh1 = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)jishui_aligned_malloc(SEQ*HIDDEN*4);
        // Metal SiLU writes both derivatives directly; CPU-only temporaries
        // are unnecessary on that path.
        float *silu_tmp = use_metal_silu ? NULL : (float*)jishui_malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = use_metal_silu ? NULL : (float*)jishui_malloc(SEQ*HIDDEN*4);
        // Retain the materialization buffers for the diagnostic legacy SDPA
        // path. Production MHA uses direct K/V and output buffers.
        float *k_tiled = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *v_tiled = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *dq_full = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *dk_full = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *dv_full = (float*)jishui_malloc(SEQ*Q_DIM*4);
        float *dx2_scaled = (float*)jishui_malloc(SEQ*DIM*4);
        // dffn and dx_kv are both transient and their lifetimes do not overlap
        // within one layer's reverse pass.
        float *dx_kv = dffn;

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();
        dispatch_semaphore_t dw_slots = dispatch_semaphore_create(2);
        if (!dw_q || !dw_grp || !dw_slots) {
            printf("Unable to create dW dispatch resources\n"); return 1;
        }

        if (!benchmark_barrier(ready_path, start_path, barrier_timeout)) return 1;
        uint64_t t_benchmark_start = mach_absolute_time();

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
        uint32_t hybrid_generation = 0;
        uint32_t hybrid_microsteps = 0;
        uint32_t hybrid_microstep = 0;
        uint64_t hybrid_record_base = 0;
        if (hybrid_mode)
            hybrid_control_set_status(hybrid_control, HYBRID_STATUS_READY);
        uint64_t t_wall_start = mach_absolute_time();
        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t_step = mach_absolute_time();
            if (hybrid_mode) {
                if (hybrid_control_load(&hybrid_control->command) == HYBRID_CMD_STOP) {
                    hybrid_control_set_status(hybrid_control, HYBRID_STATUS_STOPPED);
                    break;
                }
                bool needs_generation = hybrid_microstep == hybrid_microsteps;
                if (needs_generation) {
                    if (!hybrid_wait_for_generation(hybrid_control, hybrid_generation,
                                                    &hybrid_generation,
                                                    &hybrid_microsteps,
                                                    &hybrid_record_base,
                                                    hybrid_timeout)) {
                        if (hybrid_control_load(&hybrid_control->command) == HYBRID_CMD_STOP) {
                            hybrid_control_set_status(hybrid_control, HYBRID_STATUS_STOPPED);
                        } else {
                            training_failed = true;
                        }
                        break;
                    }
                    // `hybrid_wait_for_generation` has just accepted a new
                    // generation. Restage once, then reuse the same snapshot
                    // while accumulating all requested native microsteps.
                    hybrid_microstep = 0;
                    if (!hybrid_restage_weights(&hybrid_weights, lw, rms_final,
                                                rms_final_b, embed, Wqt_buf,
                                                Wkt_buf, Wvt_buf, Wot_buf, W1t_buf,
                                                W3t_buf, pls)) {
                        printf("Hybrid generation %u weight restaging failed\n",
                               hybrid_generation);
                        hybrid_control_set_status(hybrid_control, HYBRID_STATUS_ERROR);
                        training_failed = true;
                        break;
                    }
                }
            }
            AneRng sample_rng;
            ane_rng_seed(&sample_rng, UINT64_C(0x4a4953485549) ^ (uint64_t)(uint32_t)step);

            // Sample data
            uint16_t *input_tokens;
            uint16_t *target_tokens_raw;
            if (hybrid_mode) {
                if (data_mode != ANE_DATA_RECORDS || n_records == 0) {
                    printf("Hybrid worker requires a non-empty records stream\n");
                    hybrid_control_set_status(hybrid_control, HYBRID_STATUS_ERROR);
                    training_failed = true;
                    break;
                }
                // record_base is a record number, not a byte offset. The
                // coordinator chooses it so the MLX and ANE workers consume
                // the exact same [SEQ+1] token record.
                size_t rec = (size_t)((hybrid_record_base + hybrid_microstep) % n_records);
                input_tokens = token_data + rec * (SEQ + 1);
                target_tokens_raw = input_tokens + 1;
            } else if (data_mode == ANE_DATA_INDEX) {
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

            x_cur = acts[0].layer_in;
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0;
            double t_cls_fwd=0, t_xent=0, t_cls_bwd=0, t_dw_copy=0;

            // ===== FORWARD (28 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                // x_cur already aliases this layer's saved residual input.
                // Reuse the next layer's saved input as this layer's output.
                float *x_next = (L + 1 < NLAYERS)
                    ? acts[L + 1].layer_in : final_residual;

                // Pre-attention norm (CPU or Metal)
                t0 = mach_absolute_time();
                if (use_metal_norm) {
                    if (!metal_layernorm_forward(ac->xnorm, x_cur, lw[L].rms_att, lw[L].rms_att_b)) return 1;
                } else {
                    norm_forward(ac->xnorm, x_cur, lw[L].rms_att, lw[L].rms_att_b, DIM, SEQ);
                }
                t_rms += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "forward", "xnorm", ac->xnorm,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "forward", "xnorm",
                                    ac->xnorm, (size_t)SEQ * DIM);

                // SDPA forward (ANE): xnorm + Wq,Wk,Wv → attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM]
                t0 = mach_absolute_time();
                if (!write_sdpa_fwd_acts(pls[L].sdpaFwd_in, ac->xnorm)) return 1;
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
                    void *fwd_base = NULL;
                    if (!io_surface_lock_base(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly,
                                               &fwd_base, "sdpa forward output")) return 1;
                    _Float16 *fwd_out = (_Float16*)fwd_base;
                    int off = 0;
                    cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                    cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                    cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                    cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ);
                    if (!io_surface_unlock_checked(dk.sdpaFwd->ioOut,
                                                   kIOSurfaceLockReadOnly,
                                                   "sdpa forward output")) return 1;
                    if (!write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out)) return 1;
                }
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "forward", "attn_out", ac->attn_out,
                           (size_t)SEQ * Q_DIM);
                trace_norm(trace_norms, step, L, "forward", "Q", ac->Q,
                           (size_t)SEQ * Q_DIM);
                trace_norm(trace_norms, step, L, "forward", "K", ac->K,
                           (size_t)SEQ * KV_DIM);
                trace_norm(trace_norms, step, L, "forward", "V", ac->V,
                           (size_t)SEQ * KV_DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1)) {
                    trace_layer_f32(trace_dir, step, L, "forward", "attn_out",
                                    ac->attn_out, (size_t)SEQ * Q_DIM);
                    trace_layer_f32(trace_dir, step, L, "forward", "Q", ac->Q,
                                    (size_t)SEQ * Q_DIM);
                    trace_layer_f32(trace_dir, step, L, "forward", "K", ac->K,
                                    (size_t)SEQ * KV_DIM);
                    trace_layer_f32(trace_dir, step, L, "forward", "V", ac->V,
                                    (size_t)SEQ * KV_DIM);
                }

                // Wo forward (ANE): attn_out[Q_DIM] -> a transient DIM-wide
                // buffer. The projection output is consumed immediately by
                // the residual and is not needed by backward.
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.woFwd, plr[L].woFwd)) return 1;
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!io_read_dyn(dk.woFwd->ioOut, x_final, DIM, SEQ)) return 1;
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Scaled residual + pre-FFN norm
                t0 = mach_absolute_time();
                vDSP_vsma(x_final, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                if (use_metal_norm) {
                    if (!metal_layernorm_forward(ac->x2norm, ac->x2, lw[L].rms_ffn, lw[L].rms_ffn_b)) return 1;
                } else {
                    norm_forward(ac->x2norm, ac->x2, lw[L].rms_ffn, lw[L].rms_ffn_b, DIM, SEQ);
                }
                t_rms += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "forward", "x2", ac->x2,
                           (size_t)SEQ * DIM);
                trace_norm(trace_norms, step, L, "forward", "x2norm", ac->x2norm,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1)) {
                    trace_layer_f32(trace_dir, step, L, "forward", "x2", ac->x2,
                                    (size_t)SEQ * DIM);
                    trace_layer_f32(trace_dir, step, L, "forward", "x2norm",
                                    ac->x2norm, (size_t)SEQ * DIM);
                }

                // Fused FFN (ANE)
                t0 = mach_absolute_time();
                if (!write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2)) return 1;
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnFused, plr[L].ffnFused)) return 1;
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                if (use_metal_io) {
                    if (!metal_ffn_unpack_output(dk.ffnFused->ioOut, x_next, ac->h1,
                                                 ac->h3, ac->silu_out))
                        return 1;
                } else {
                    void *ffn_base = NULL;
                    if (!io_surface_lock_base(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly,
                                               &ffn_base, "FFN forward output")) return 1;
                    _Float16 *ffn_out = (_Float16*)ffn_base;
                    int off = 0;
                    cvt_f16_f32(x_next,      ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                    cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                    cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                    cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
                    if (!io_surface_unlock_checked(dk.ffnFused->ioOut,
                                                   kIOSurfaceLockReadOnly,
                                                   "FFN forward output")) return 1;
                }
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "forward", "x_next", x_next,
                           (size_t)SEQ * DIM);
                trace_norm(trace_norms, step, L, "forward", "h1", ac->h1,
                           (size_t)SEQ * HIDDEN);
                trace_norm(trace_norms, step, L, "forward", "h3", ac->h3,
                           (size_t)SEQ * HIDDEN);
                trace_norm(trace_norms, step, L, "forward", "silu_out", ac->silu_out,
                           (size_t)SEQ * HIDDEN);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1)) {
                    trace_layer_f32(trace_dir, step, L, "forward", "x_next", x_next,
                                    (size_t)SEQ * DIM);
                    trace_layer_f32(trace_dir, step, L, "forward", "h1", ac->h1,
                                    (size_t)SEQ * HIDDEN);
                    trace_layer_f32(trace_dir, step, L, "forward", "h3", ac->h3,
                                    (size_t)SEQ * HIDDEN);
                    trace_layer_f32(trace_dir, step, L, "forward", "silu_out",
                                    ac->silu_out, (size_t)SEQ * HIDDEN);
                }
                x_cur = x_next;
            }

            // Final norm + classifier + loss. The prior microstep waits for all
            // dW jobs before returning, so these buffers are safe to reuse.
            t0 = mach_absolute_time();
            if (use_metal_norm) {
                if (!metal_layernorm_forward(x_final, final_residual, rms_final, rms_final_b)) return 1;
            } else {
                norm_forward(x_final, final_residual, rms_final, rms_final_b, DIM, SEQ);
            }
            t_rms += tb_ms(mach_absolute_time() - t0);
            trace_norm(trace_norms, step, NLAYERS, "forward", "final_residual",
                       final_residual, (size_t)SEQ * DIM);
            trace_norm(trace_norms, step, NLAYERS, "forward", "x_final",
                       x_final, (size_t)SEQ * DIM);
            if (trace_dir && step == 0) {
                trace_layer_f32(trace_dir, step, NLAYERS, "forward", "final_residual",
                                final_residual, (size_t)SEQ * DIM);
                trace_layer_f32(trace_dir, step, NLAYERS, "forward", "x_final",
                                x_final, (size_t)SEQ * DIM);
            }
            t0 = mach_absolute_time();
            // Token-major logits [SEQ,CV] keep each softmax row contiguous.
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, x_final, SEQ, cembed, DIM, 0.0f, logits, CV);
            t_cls_fwd += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            float loss = cross_entropy_loss_inplace(logits, ctargets, CV, SEQ);
            t_xent += tb_ms(mach_absolute_time() - t0);
            if (!isfinite(loss)) {
                printf("Non-finite loss at microstep %d; refusing to update Adam\n", step + 1);
                return 1;
            }
            last_loss = loss;

            // ===== BACKWARD =====
            // `logits` now stores dlogits in place.
            vDSP_vsmul(logits, 1, &loss_scale, logits, 1, (vDSP_Length)(SEQ*CV));

            // Classifier backward
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, logits, CV, 0.0f, dy, SEQ);
            t_cls_bwd += tb_ms(mach_absolute_time() - t0);

            // dEmbed async
            float *classifier_gembed = identity_vocab ? gembed : gcembed;
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, logits, CV, x_final, SEQ,
                            1.0f, classifier_gembed, DIM);
            });

            // Final norm backward
            if (use_metal_norm) {
                if (!metal_layernorm_backward(dx_norm_scratch, grms_final, grms_final_b,
                                              dy, final_residual, rms_final)) return 1;
            } else {
                norm_backward(dx_norm_scratch, grms_final, grms_final_b, dy, final_residual, rms_final, DIM, SEQ);
            }
            // LayerNorm backward filled the alternate buffer completely;
            // make it the current gradient without a full-buffer memcpy.
            float *gradient_swap = dy;
            dy = dx_norm_scratch;
            dx_norm_scratch = gradient_swap;

            // ===== BACKWARD (28 layers, reverse) =====
            bool qkv_shadow_failed = false;
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                trace_norm(trace_norms, step, L, "backward", "dy_in", dy,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "backward", "dy_in", dy,
                                    (size_t)SEQ * DIM);

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));
                trace_norm(trace_norms, step, L, "backward", "dffn", dffn,
                           (size_t)SEQ * DIM);

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
                if (!write_ffn_bwd_w2t_acts_scaled(pls[L].ffnBwdW2t_in, dffn,
                                                   ffn_bwd_scale)) return 1;
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ)) return 1;
                if (ffn_bwd_scale != 1.0f) {
                    float inv_ffn_bwd_scale = 1.0f / ffn_bwd_scale;
                    vDSP_vsmul(dsilu, 1, &inv_ffn_bwd_scale, dsilu, 1,
                               (vDSP_Length)(SEQ * HIDDEN));
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "backward", "dsilu", dsilu,
                           (size_t)SEQ * HIDDEN);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "backward", "dsilu",
                                    dsilu, (size_t)SEQ * HIDDEN);

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
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1)) {
                    trace_layer_f32(trace_dir, step, L, "backward", "dh1",
                                    dh1, (size_t)SEQ * HIDDEN);
                    trace_layer_f32(trace_dir, step, L, "backward", "dh3",
                                    dh3, (size_t)SEQ * HIDDEN);
                }

                // dh1@W1^T + dh3@W3^T → dx_ffn (ANE)
                t0 = mach_absolute_time();
                if (!write_ffn_bwd_w13t_acts_scaled(pls[L].ffnBwdW13t_in,
                                                    dh1, dh3,
                                                    ffn_w13_bwd_scale)) return 1;
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ)) return 1;
                if (ffn_w13_bwd_scale != 1.0f) {
                    float inv_ffn_w13_bwd_scale = 1.0f / ffn_w13_bwd_scale;
                    vDSP_vsmul(dx_ffn, 1, &inv_ffn_w13_bwd_scale, dx_ffn, 1,
                               (vDSP_Length)(SEQ * DIM));
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "backward", "dx_ffn", dx_ffn,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "backward", "dx_ffn",
                                    dx_ffn, (size_t)SEQ * DIM);

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
                if (use_metal_norm) {
                    if (!metal_layernorm_backward(dx2, gr->rms_ffn, gr->rms_ffn_b,
                                                  dx_ffn, ac->x2, lw[L].rms_ffn)) return 1;
                } else {
                    norm_backward(dx2, gr->rms_ffn, gr->rms_ffn_b, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                }
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "backward", "dx2", dx2,
                           (size_t)SEQ * DIM);

                // Wo^T backward (ANE): alpha*dx2 @ Wo → da[Q_DIM]. The
                // legacy diagnostic path keeps the pre-optimization explicit
                // fp32 scaled buffer and writer; production folds scaling into
                // the fp16 conversion.
                t0 = mach_absolute_time();
                if (legacy_sdpa_bwd) {
                    vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1,
                               (vDSP_Length)(SEQ * DIM));
                    if (!write_wot_bwd_acts_scaled(pls[L].wotBwd_in,
                                                    dx2_scaled,
                                                    wo_bwd_scale)) return 1;
                } else if (!write_wot_bwd_acts_scaled(pls[L].wotBwd_in, dx2,
                                                       res_alpha *
                                                       wo_bwd_scale)) return 1;
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!ane_eval_req(dk.wotBwd, plr[L].wotBwd)) return 1;
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                if (!io_read_dyn(dk.wotBwd->ioOut, da_buf, Q_DIM, SEQ)) return 1;
                if (wo_bwd_scale != 1.0f) {
                    float inv_wo_bwd_scale = 1.0f / wo_bwd_scale;
                    vDSP_vsmul(da_buf, 1, &inv_wo_bwd_scale, da_buf, 1,
                               (vDSP_Length)(SEQ * Q_DIM));
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "backward", "da", da_buf,
                           (size_t)SEQ * Q_DIM);

                // dWo async: gr->Wo[DIM,Q_DIM] += dx2_scaled[DIM,SEQ] @ attn_out^T[SEQ,Q_DIM]
                t0 = mach_absolute_time();
                dispatch_semaphore_wait(dw_slots, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *capt_do = (float*)jishui_malloc(SEQ*DIM*4);
                if (legacy_sdpa_bwd)
                    memcpy(capt_do, dx2_scaled, SEQ * DIM * sizeof(float));
                else
                    vDSP_vsmul(dx2, 1, &res_alpha, capt_do, 1,
                               (vDSP_Length)(SEQ * DIM));
                const float *saved_attn = ac->attn_out;
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                1.0f, capt_do, SEQ, saved_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
                    free(capt_do);
                    dispatch_semaphore_signal(dw_slots);
                });

                // SDPA backward.  The legacy branch intentionally preserves
                // the pre-optimization MHA/GQA materialization and one-lock
                // per segment.  It is useful for isolating regressions in the
                // direct-MHA and batched-I/O path.
                t0 = mach_absolute_time();
                const float *sdpa_k = ac->K;
                const float *sdpa_v = ac->V;
                if (legacy_sdpa_bwd) {
                    gqa_tile_kv(k_tiled, ac->K, SEQ);
                    gqa_tile_kv(v_tiled, ac->V, SEQ);
                    sdpa_k = k_tiled;
                    sdpa_v = v_tiled;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);

                    t0 = mach_absolute_time();
                    if (!io_write_fp16_at(dk.sdpaBwd1->ioIn, 0, ac->Q,
                                          Q_DIM, SEQ) ||
                        !io_write_fp16_at(dk.sdpaBwd1->ioIn, Q_DIM, sdpa_k,
                                          Q_DIM, SEQ) ||
                        !io_write_fp16_at(dk.sdpaBwd1->ioIn, 2 * Q_DIM, sdpa_v,
                                          Q_DIM, SEQ) ||
                        !io_write_fp16_at_scaled(dk.sdpaBwd1->ioIn,
                                                 3 * Q_DIM, da_buf,
                                                 sdpa_bwd_scale,
                                                 Q_DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval(dk.sdpaBwd1)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                    t0 = mach_absolute_time();
                    if (!io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut,
                                 Q_DIM, 2 * SCORE_CH, SEQ) ||
                        !io_write_fp16_at(dk.sdpaBwd2->ioIn, 2 * SCORE_CH,
                                          ac->Q, Q_DIM, SEQ) ||
                        !io_write_fp16_at(dk.sdpaBwd2->ioIn,
                                          2 * SCORE_CH + Q_DIM, sdpa_k,
                                          Q_DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval(dk.sdpaBwd2)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                    t0 = mach_absolute_time();
                    if (!io_read_fp16(dk.sdpaBwd2->ioOut, dq_full, 0,
                                      Q_DIM, SEQ) ||
                        !io_read_fp16(dk.sdpaBwd2->ioOut, dk_full, Q_DIM,
                                      Q_DIM, SEQ) ||
                        !io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,
                                      Q_DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    gqa_reduce_kv(dk_buf, dk_full, SEQ);
                    gqa_reduce_kv(dv, dv_full, SEQ);
                    memcpy(dq, dq_full, (size_t)SEQ * Q_DIM * sizeof(float));
                } else {
#if GQA_RATIO != 1
                    gqa_tile_kv(k_tiled, ac->K, SEQ);
                    gqa_tile_kv(v_tiled, ac->V, SEQ);
                    sdpa_k = k_tiled;
                    sdpa_v = v_tiled;
#endif
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    const int sdpa_bwd1_offsets[4] = {0, Q_DIM, 2*Q_DIM, 3*Q_DIM};
                    const float *sdpa_bwd1_data[4] = {ac->Q, sdpa_k, sdpa_v, da_buf};
                    const float sdpa_bwd1_scales[4] = {1.0f, 1.0f, 1.0f,
                                                       sdpa_bwd_scale};
                    t0 = mach_absolute_time();
                    if (!io_write_fp16_at_many_scaled(
                            dk.sdpaBwd1->ioIn, sdpa_bwd1_offsets,
                            sdpa_bwd1_data, sdpa_bwd1_scales,
                            4, Q_DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval(dk.sdpaBwd1)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                    const int sdpa_bwd2_offsets[2] = {2*SCORE_CH,
                                                      2*SCORE_CH+Q_DIM};
                    const float *sdpa_bwd2_data[2] = {ac->Q, sdpa_k};
                    t0 = mach_absolute_time();
                    if (!io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut,
                                 Q_DIM, 2*SCORE_CH, SEQ) ||
                        !io_write_fp16_at_many(dk.sdpaBwd2->ioIn,
                                               sdpa_bwd2_offsets,
                                               sdpa_bwd2_data, 2, Q_DIM,
                                               SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval(dk.sdpaBwd2)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
#if GQA_RATIO == 1
                    const int sdpa_bwd2_read_offsets[2] = {0, Q_DIM};
                    float *sdpa_bwd2_read_data[2] = {dq, dk_buf};
                    t0 = mach_absolute_time();
                    if (!io_read_fp16_at_many(dk.sdpaBwd2->ioOut,
                                              sdpa_bwd2_read_offsets,
                                              sdpa_bwd2_read_data, 2,
                                              Q_DIM, SEQ) ||
                        !io_read_fp16(dk.sdpaBwd1->ioOut, dv, 0, Q_DIM,
                                      SEQ)) return 1;
#else
                    t0 = mach_absolute_time();
                    const int sdpa_bwd2_read_offsets[2] = {0, Q_DIM};
                    float *sdpa_bwd2_read_data[2] = {dq_full, dk_full};
                    if (!io_read_fp16_at_many(dk.sdpaBwd2->ioOut,
                                              sdpa_bwd2_read_offsets,
                                              sdpa_bwd2_read_data, 2,
                                              Q_DIM, SEQ) ||
                        !io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,
                                      Q_DIM, SEQ)) return 1;
                    memcpy(dq, dq_full, (size_t)SEQ * Q_DIM * sizeof(float));
                    gqa_reduce_kv(dk_buf, dk_full, SEQ);
                    gqa_reduce_kv(dv, dv_full, SEQ);
#endif
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                }

                if (sdpa_bwd_scale != 1.0f) {
                    float inv_sdpa_bwd_scale = 1.0f / sdpa_bwd_scale;
                    vDSP_vsmul(dq, 1, &inv_sdpa_bwd_scale, dq, 1,
                               (vDSP_Length)(SEQ * Q_DIM));
                    vDSP_vsmul(dk_buf, 1, &inv_sdpa_bwd_scale, dk_buf, 1,
                               (vDSP_Length)(SEQ * KV_DIM));
                    vDSP_vsmul(dv, 1, &inv_sdpa_bwd_scale, dv, 1,
                               (vDSP_Length)(SEQ * KV_DIM));
                }

                trace_norm(trace_norms, step, L, "backward", "dq", dq,
                           (size_t)SEQ * Q_DIM);
                trace_norm(trace_norms, step, L, "backward", "dk", dk_buf,
                           (size_t)SEQ * KV_DIM);
                trace_norm(trace_norms, step, L, "backward", "dv", dv,
                           (size_t)SEQ * KV_DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1)) {
                    trace_layer_f32(trace_dir, step, L, "backward", "dy_in", dy,
                                    (size_t)SEQ * DIM);
                    trace_layer_f32(trace_dir, step, L, "backward", "da", da_buf,
                                    (size_t)SEQ * Q_DIM);
                    trace_layer_f32(trace_dir, step, L, "backward", "dq", dq,
                                    (size_t)SEQ * Q_DIM);
                    trace_layer_f32(trace_dir, step, L, "backward", "dk", dk_buf,
                                    (size_t)SEQ * KV_DIM);
                    trace_layer_f32(trace_dir, step, L, "backward", "dv", dv,
                                    (size_t)SEQ * KV_DIM);
                }

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

                bool fused_qkv_ok = false;
                if (use_qkv_fused && !qkv_fusion_disabled &&
                    pls[L].qkvBwd_in && plr[L].qkvBwd) {
                    // Experimental MHA path: one ANE request produces dx_q and
                    // the same fp16-summed dx_kv as the split implementation.
                    t0 = mach_absolute_time();
                    if (!write_qkv_bwd_acts_scaled(pls[L].qkvBwd_in,
                                                   dq, dk_buf, dv,
                                                   qkv_bwd_scale)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (ane_eval_req(dk.qkvBwd, plr[L].qkvBwd)) {
                        t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                        t0 = mach_absolute_time();
                        const int qkv_read_offsets[2] = {0, DIM};
                        float *qkv_read_data[2] = {dx_attn, dx_kv};
                        if (!io_read_fp16_at_many(dk.qkvBwd->ioOut,
                                                  qkv_read_offsets, qkv_read_data,
                                                  2, DIM, SEQ)) return 1;
                        bool fused_finite =
                            f32_buffer_all_finite(dx_attn, (size_t)SEQ * DIM) &&
                            f32_buffer_all_finite(dx_kv, (size_t)SEQ * DIM);
                        t_io_bwd += tb_ms(mach_absolute_time() - t0);
                        if (!fused_finite) {
                            printf("    qkv fusion produced non-finite output; falling back to split path\n");
                            qkv_fusion_disabled = true;
                        } else {
                            fused_qkv_ok = true;
                        }
                    } else {
                        printf("    qkv fusion evaluation failed; falling back to split path\n");
                        qkv_fusion_disabled = true;
                    }

                    if (fused_qkv_ok && qkv_fused_shadow) {
                        // Run the established split graphs as a correctness
                        // oracle. If they disagree, use the oracle result and
                        // keep the process alive for diagnostic parity output.
                        float *split_q = (float *)jishui_malloc((size_t)SEQ * DIM * 4);
                        float *split_kv = (float *)jishui_malloc((size_t)SEQ * DIM * 4);
                        bool split_ok = write_q_bwd_acts_scaled(
                            pls[L].qBwd_in, dq, qkv_bwd_scale);
                        split_ok = split_ok && ane_eval_req(dk.qBwd, plr[L].qBwd);
                        if (split_ok) split_ok = io_read_dyn(dk.qBwd->ioOut, split_q, DIM, SEQ);
                        if (split_ok) split_ok = write_kv_bwd_acts_scaled(
                            pls[L].kvBwd_in, dk_buf, dv, qkv_bwd_scale);
                        split_ok = split_ok && ane_eval_req(dk.kvBwd, plr[L].kvBwd);
                        if (split_ok) split_ok = io_read_dyn(dk.kvBwd->ioOut, split_kv, DIM, SEQ);
                        float max_diff = 0.0f;
                        bool fused_finite =
                            f32_buffer_all_finite(dx_attn, (size_t)SEQ * DIM) &&
                            f32_buffer_all_finite(dx_kv, (size_t)SEQ * DIM);
                        bool split_finite = split_ok &&
                            f32_buffer_all_finite(split_q, (size_t)SEQ * DIM) &&
                            f32_buffer_all_finite(split_kv, (size_t)SEQ * DIM);
                        if (fused_finite && split_finite) {
                            for (size_t i = 0; i < (size_t)SEQ * DIM; i++) {
                                float diff_q = fabsf(dx_attn[i] - split_q[i]);
                                float diff_kv = fabsf(dx_kv[i] - split_kv[i]);
                                // fmaxf(NaN, x) is implementation-sensitive and
                                // can leave a finite accumulator. Treat any
                                // non-finite parity sample as a hard mismatch.
                                if (!isfinite(diff_q) || !isfinite(diff_kv)) {
                                    max_diff = INFINITY;
                                    break;
                                }
                                max_diff = fmaxf(max_diff, diff_q);
                                max_diff = fmaxf(max_diff, diff_kv);
                            }
                        } else {
                            max_diff = INFINITY;
                        }
                        printf("    qkv_shadow L%d max_dx_diff=%.3e\n", L, max_diff);
                        if (!split_finite) {
                            printf("    qkv shadow split reference failed or produced non-finite output; aborting gate\n");
                            free(split_q);
                            free(split_kv);
                            qkv_shadow_failed = true;
                            break;
                        }
                        if (!fused_finite || max_diff > 2.0e-2f) {
                            printf("    qkv fusion mismatch; using finite split reference\n");
                            qkv_fusion_disabled = true;
                            memcpy(dx_attn, split_q, (size_t)SEQ * DIM * 4);
                            memcpy(dx_kv, split_kv, (size_t)SEQ * DIM * 4);
                        }
                        free(split_q);
                        free(split_kv);
                    }
                }
                if (!fused_qkv_ok) {
                    // Established split path: qBwd and kvBwd are separately
                    // evaluated for GQA models and as the default MHA path.
                    t0 = mach_absolute_time();
                    if (!write_q_bwd_acts_scaled(pls[L].qBwd_in, dq,
                                                 qkv_bwd_scale)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval_req(dk.qBwd, plr[L].qBwd)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);

                    t0 = mach_absolute_time();
                    if (!write_kv_bwd_acts_scaled(pls[L].kvBwd_in,
                                                  dk_buf, dv,
                                                  qkv_bwd_scale)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!ane_eval_req(dk.kvBwd, plr[L].kvBwd)) return 1;
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    if (!io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ)) return 1;
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                }

                if (qkv_bwd_scale != 1.0f) {
                    float inv_qkv_bwd_scale = 1.0f / qkv_bwd_scale;
                    vDSP_vsmul(dx_attn, 1, &inv_qkv_bwd_scale, dx_attn, 1,
                               (vDSP_Length)(SEQ * DIM));
                    vDSP_vsmul(dx_kv, 1, &inv_qkv_bwd_scale, dx_kv, 1,
                               (vDSP_Length)(SEQ * DIM));
                }

                // dx_attn = dx_q + dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];
                trace_norm(trace_norms, step, L, "backward", "dx_attn", dx_attn,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "backward", "dx_attn",
                                    dx_attn, (size_t)SEQ * DIM);

                // Pre-attention norm backward
                t0 = mach_absolute_time();
                if (use_metal_norm) {
                    if (!metal_layernorm_backward(dx_norm_scratch, gr->rms_att, gr->rms_att_b,
                                                  dx_attn, ac->layer_in, lw[L].rms_att)) return 1;
                } else {
                    norm_backward(dx_norm_scratch, gr->rms_att, gr->rms_att_b, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                }
                // The next residual gradient is fully materialized in the
                // alternate buffer. Add in place, then swap ownership for the
                // next layer; no [SEQ,DIM] copy is needed.
                vDSP_vadd(dx_norm_scratch, 1, dx2, 1, dx_norm_scratch, 1,
                          (vDSP_Length)(SEQ*DIM));
                gradient_swap = dy;
                dy = dx_norm_scratch;
                dx_norm_scratch = gradient_swap;
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
                trace_norm(trace_norms, step, L, "backward", "dy_out", dy,
                           (size_t)SEQ * DIM);
                if (trace_dir && step == 0 && (L == 0 || L == NLAYERS - 1))
                    trace_layer_f32(trace_dir, step, L, "backward", "dy_out", dy,
                                    (size_t)SEQ * DIM);
            }

            // Do not run embedding backward or update gradients after a
            // shadow oracle failure. Leaving through the normal cleanup path
            // keeps ANE surfaces and asynchronous dW captures reclaimable.
            if (qkv_shadow_failed) {
                training_failed = true;
                break;
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
                vDSP_maxv(final_residual,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(final_residual,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            if (hybrid_mode) {
                hybrid_microstep++;
                if (hybrid_microstep < hybrid_microsteps) {
                    // Keep accumulating into the same fp32 gradient buffers.
                    // The next loop iteration consumes the next shared record
                    // without restaging the unchanged weight generation.
                    continue;
                }
                // MLX owns accumulation, clipping and AdamW. Convert the
                // native loss-scaled gradient back to the unscaled fp16
                // protocol representation, publish it, and wait for the
                // coordinator to provide the next generation's weights.
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accumulated_microsteps * loss_scale);
                for (int L = 0; L < NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for (size_t i = 0; i < WQ_SZ; i++) g->Wq[i] *= gsc;
                    for (size_t i = 0; i < WK_SZ; i++) g->Wk[i] *= gsc;
                    for (size_t i = 0; i < WV_SZ; i++) g->Wv[i] *= gsc;
                    for (size_t i = 0; i < WO_SZ; i++) g->Wo[i] *= gsc;
                    for (size_t i = 0; i < W1_SZ; i++) g->W1[i] *= gsc;
                    for (size_t i = 0; i < W2_SZ; i++) g->W2[i] *= gsc;
                    for (size_t i = 0; i < W3_SZ; i++) g->W3[i] *= gsc;
                    for (int i = 0; i < DIM; i++) {
                        g->rms_att[i] *= gsc; g->rms_att_b[i] *= gsc;
                        g->rms_ffn[i] *= gsc; g->rms_ffn_b[i] *= gsc;
                    }
                }
                for (int i = 0; i < DIM; i++) {
                    grms_final[i] *= gsc;
                    grms_final_b[i] *= gsc;
                }
                if (!identity_vocab) vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for (size_t i = 0; i < (size_t)VOCAB * DIM; i++) gembed[i] *= gsc;

                if (!hybrid_export_gradients(&hybrid_gradients, grads,
                                             grms_final, grms_final_b, gembed)) {
                    printf("Hybrid generation %u gradient export failed\n",
                           hybrid_generation);
                    hybrid_control_set_status(hybrid_control, HYBRID_STATUS_ERROR);
                    training_failed = true;
                    break;
                }
                hybrid_control_set_status(hybrid_control, HYBRID_STATUS_GRAD_READY);
                for (int L = 0; L < NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM * 4);
                memset(grms_final_b, 0, DIM * 4);
                memset(gembed, 0, (size_t)VOCAB * DIM * 4);
                if (gcembed) memset(gcembed, 0, (size_t)CV * DIM * 4);
                accumulated_microsteps = 0;
                continue;
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
                if (!identity_vocab) vocab_scatter_grads(gembed, gcembed, &vm, DIM);
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
                    // Fill each ANE-facing transpose while Adam writes the
                    // fp32 master matrix; this replaces six full post-update
                    // transpose passes over the transformer weights.
                    adam_update_with_transpose(lw[L].Wq, Wqt_buf[L], g->Wq,
                                               &la[L].Wq, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, Q_DIM, DIM);
                    adam_update_with_transpose(lw[L].Wk, Wkt_buf[L], g->Wk,
                                               &la[L].Wk, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, KV_DIM, DIM);
                    adam_update_with_transpose(lw[L].Wv, Wvt_buf[L], g->Wv,
                                               &la[L].Wv, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, KV_DIM, DIM);
                    adam_update_with_transpose(lw[L].Wo, Wot_buf[L], g->Wo,
                                               &la[L].Wo, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, DIM, Q_DIM);
                    adam_update_with_transpose(lw[L].W1, W1t_buf[L], g->W1,
                                               &la[L].W1, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, HIDDEN, DIM);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update_with_transpose(lw[L].W3, W3t_buf[L], g->W3,
                                               &la[L].W3, adam_t, lr, adam_b1,
                                               adam_b2, adam_eps, wd, HIDDEN, DIM);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_att_b, g->rms_att_b, &la[L].rms_att_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn_b, g->rms_ffn_b, &la[L].rms_ffn_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    // Re-stage weights
                    if (!stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]) ||
                        !stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]) ||
                        !stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2) ||
                        !stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2) ||
                        !stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3) ||
                        !stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo) ||
                        !stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq) ||
                        !stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv) ||
                        (pls[L].qkvBwd_in &&
                         !stage_qkv_bwd_weights(pls[L].qkvBwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv))) {
                        printf("Per-layer IOSurface restaging failed at layer %d\n", L);
                        return 1;
                    }
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(rms_final_b, grms_final_b, &arms_final_b, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                if (!identity_vocab) {
                    free(cembed);
                    cembed = vocab_compact_embed(embed, &vm, DIM);
                }

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
                if (gcembed) memset(gcembed, 0, (size_t)CV*DIM*4);
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

        double benchmark_ms = tb_ms(mach_absolute_time() - t_benchmark_start);
        printf("BENCHMARK_JSON {\"worker\":\"ane\",\"microsteps\":%d,"
               "\"tokens\":%lld,\"seconds\":%.9f,\"tokens_per_second\":%.3f}\n",
               total_steps_done, (long long)total_steps_done * SEQ,
               benchmark_ms / 1000.0,
               benchmark_ms > 0.0 ? (double)total_steps_done * SEQ * 1000.0 / benchmark_ms : 0.0);

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
        if (trace_norms && fclose(trace_norms) != 0)
            perror("trace norms close");

        // Cleanup
        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
        if (use_metal_norm || use_metal_silu || use_metal_io) metal_ops_shutdown();
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W3t_buf[L]);
        }
        free_per_layer(pls, plr);
        free_dynamic_kernels(&dk);
        free(da_buf); free(k_tiled); free(v_tiled);
        free(dq_full); free(dk_full); free(dv_full);
        free(dx2_scaled);
        free(dq); free(dk_buf); free(dv);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(dx_attn);
        free(final_residual); free(x_final); free(dx_norm_scratch);
        free(logits);
        free(dh1); free(dh3); free(dsilu); free(silu_tmp); free(silu_tmp2);
        free(rms_final); free(rms_final_b); free(grms_final); free(grms_final_b);
        rope_backward_shutdown();
        adam_free(&arms_final); adam_free(&arms_final_b); adam_free(&aembed);
        free(embed); free(gembed);
        if (!identity_vocab) free(cembed);
        free(gcembed);
        free(vm.full_to_compact); free(vm.compact_to_full);
        if (hybrid_mode) {
            hybrid_tensor_map_close(&hybrid_weights);
            hybrid_tensor_map_close(&hybrid_gradients);
            hybrid_control_close(hybrid_control, hybrid_control_fd);
        }
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
