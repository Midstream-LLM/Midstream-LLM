// io.h — IOSurface helpers, NEON conversion, kernel compile/eval
// Updated for GQA (Qwen3-0.6B): Q_DIM != DIM, separate KV heads
#pragma once
#include "config.h"

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Blob builders for const weights (mask, rms)
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)jishui_calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)jishui_calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}
static void cvt_f32_scaled_f16(_Float16 *dst, const float *src, float scale, int n) {
    int i = 0;
    float32x4_t factor = vdupq_n_f32(scale);
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(
            vcvt_f16_f32(vmulq_f32(vld1q_f32(src+i), factor)),
            vcvt_f16_f32(vmulq_f32(vld1q_f32(src+i+4), factor)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)(src[i] * scale);
}

// IOSurface I/O (channel-first [C,S] layout, fp16 on surface).  Every CPU
// mapping is fail-closed: a lock/base-address/unlock failure must never leave
// the caller reading or writing an unvalidated pointer.  This matters on
// systems where a transient IOSurface mapping failure can otherwise turn into
// a userspace bad access or an invalid ANE DMA request.
static bool io_surface_lock_base(IOSurfaceRef surface, uint32_t options,
                                  void **base, const char *operation) {
    if (!surface || !base) {
        printf("IOSurface %s is NULL\n", operation ? operation : "I/O");
        return false;
    }
    IOReturn result = IOSurfaceLock(surface, options, NULL);
    if (result != kIOReturnSuccess) {
        printf("IOSurface %s lock failed: 0x%x\n",
               operation ? operation : "I/O", result);
        return false;
    }
    *base = IOSurfaceGetBaseAddress(surface);
    if (!*base) {
        printf("IOSurface %s has no CPU mapping\n",
               operation ? operation : "I/O");
        IOReturn unlock_result = IOSurfaceUnlock(surface, options, NULL);
        if (unlock_result != kIOReturnSuccess)
            printf("IOSurface %s unlock after mapping failure failed: 0x%x\n",
                   operation ? operation : "I/O", unlock_result);
        return false;
    }
    return true;
}

static bool io_surface_unlock_checked(IOSurfaceRef surface, uint32_t options,
                                       const char *operation) {
    IOReturn result = IOSurfaceUnlock(surface, options, NULL);
    if (result != kIOReturnSuccess) {
        printf("IOSurface %s unlock failed: 0x%x\n",
               operation ? operation : "I/O", result);
        return false;
    }
    return true;
}

static bool io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    if (!data || channels <= 0 || sp <= 0) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "fp16 write")) return false;
    cvt_f32_f16((_Float16 *)base, data, channels * sp);
    return io_surface_unlock_checked(s, 0, "fp16 write");
}

static bool io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    if (!data || ch_off < 0 || channels <= 0 || sp <= 0) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, kIOSurfaceLockReadOnly, &base, "fp16 read")) return false;
    cvt_f16_f32(data, (_Float16 *)base + (size_t)ch_off * (size_t)sp,
                channels * sp);
    return io_surface_unlock_checked(s, kIOSurfaceLockReadOnly, "fp16 read");
}

static bool io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch,
                    int channels, int sp) {
    if (dst_ch < 0 || src_ch < 0 || channels <= 0 || sp <= 0) return false;
    void *dst_base = NULL, *src_base = NULL;
    if (!io_surface_lock_base(dst, 0, &dst_base, "fp16 copy destination")) return false;
    if (!io_surface_lock_base(src, kIOSurfaceLockReadOnly, &src_base,
                               "fp16 copy source")) {
        (void)io_surface_unlock_checked(dst, 0, "fp16 copy destination");
        return false;
    }
    memcpy((_Float16 *)dst_base + (size_t)dst_ch * (size_t)sp,
           (_Float16 *)src_base + (size_t)src_ch * (size_t)sp,
           (size_t)channels * (size_t)sp * sizeof(_Float16));
    bool src_ok = io_surface_unlock_checked(src, kIOSurfaceLockReadOnly,
                                             "fp16 copy source");
    bool dst_ok = io_surface_unlock_checked(dst, 0, "fp16 copy destination");
    return src_ok && dst_ok;
}

static bool io_write_fp16_at_scaled(IOSurfaceRef s, int ch_off,
                                    const float *data, float scale,
                                    int channels, int sp) {
    if (!data || ch_off < 0 || channels <= 0 || sp <= 0) return false;
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "fp16 offset write")) return false;
    _Float16 *destination =
        (_Float16 *)base + (size_t)ch_off * (size_t)sp;
    if (scale == 1.0f)
        cvt_f32_f16(destination, data, channels * sp);
    else
        cvt_f32_scaled_f16(destination, data, scale, channels * sp);
    return io_surface_unlock_checked(s, 0, "fp16 offset write");
}
static bool io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data,
                             int channels, int sp) {
    return io_write_fp16_at_scaled(s, ch_off, data, 1.0f, channels, sp);
}

// Batch adjacent channel segments while holding one IOSurface mapping. This
// is used at the SDPA/QKV boundaries where several tensors share one surface;
// it preserves each segment's fp16 conversion while removing redundant
// lock/unlock calls.
static bool io_write_fp16_at_many_scaled(IOSurfaceRef s,
                                         const int *channel_offsets,
                                         const float *const *data,
                                         const float *scales, int segments,
                                         int channels, int sp) {
    if (!s || !channel_offsets || !data || segments <= 0 || channels <= 0 || sp <= 0)
        return false;
    size_t capacity = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    size_t span = (size_t)channels * (size_t)sp;
    for (int i = 0; i < segments; i++) {
        if (!data[i] || channel_offsets[i] < 0) return false;
        size_t offset = (size_t)channel_offsets[i] * (size_t)sp;
        if (offset > capacity || span > capacity - offset) return false;
    }
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "batched fp16 offset write")) return false;
    _Float16 *destination = (_Float16 *)base;
    for (int i = 0; i < segments; i++) {
        float scale = scales ? scales[i] : 1.0f;
        if (!isfinite(scale) || scale <= 0.0f) {
            (void)io_surface_unlock_checked(s, 0,
                                            "batched fp16 offset write");
            return false;
        }
        _Float16 *segment =
            destination + (size_t)channel_offsets[i] * (size_t)sp;
        if (scale == 1.0f)
            cvt_f32_f16(segment, data[i], (int)span);
        else
            cvt_f32_scaled_f16(segment, data[i], scale, (int)span);
    }
    return io_surface_unlock_checked(s, 0, "batched fp16 offset write");
}

static bool io_write_fp16_at_many(IOSurfaceRef s, const int *channel_offsets,
                                  const float *const *data, int segments,
                                  int channels, int sp) {
    return io_write_fp16_at_many_scaled(s, channel_offsets, data, NULL,
                                        segments, channels, sp);
}

static bool io_read_fp16_at_many(IOSurfaceRef s, const int *channel_offsets,
                                 float *const *data, int segments,
                                 int channels, int sp) {
    if (!s || !channel_offsets || !data || segments <= 0 || channels <= 0 || sp <= 0)
        return false;
    size_t capacity = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    size_t span = (size_t)channels * (size_t)sp;
    for (int i = 0; i < segments; i++) {
        if (!data[i] || channel_offsets[i] < 0) return false;
        size_t offset = (size_t)channel_offsets[i] * (size_t)sp;
        if (offset > capacity || span > capacity - offset) return false;
    }
    void *base = NULL;
    if (!io_surface_lock_base(s, kIOSurfaceLockReadOnly, &base,
                               "batched fp16 read")) return false;
    const _Float16 *source = (const _Float16 *)base;
    for (int i = 0; i < segments; i++) {
        cvt_f16_f32(data[i],
                    source + (size_t)channel_offsets[i] * (size_t)sp,
                    (int)span);
    }
    return io_surface_unlock_checked(s, kIOSurfaceLockReadOnly,
                                     "batched fp16 read");
}

// fp16 IOSurface I/O (for dynamic matmul kernels with fp16 input/output)
static bool io_write_dyn(IOSurfaceRef s, const float *act, int ic, int seq,
                         const float *W, int oc) {
    if (!act || !W || ic <= 0 || seq <= 0 || oc <= 0) return false;
    int sp = seq + oc;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "dynamic fp16 write")) return false;
    _Float16 *buf = (_Float16 *)base;
    for (int d = 0; d < ic; d++) {
        cvt_f32_f16(buf + (size_t)d * (size_t)sp, act + (size_t)d * (size_t)seq, seq);
        cvt_f32_f16(buf + (size_t)d * (size_t)sp + seq, W + (size_t)d * (size_t)oc, oc);
    }
    return io_surface_unlock_checked(s, 0, "dynamic fp16 write");
}

// Read output from dynamic matmul kernel: [1, OC, 1, SEQ]
static bool io_read_dyn(IOSurfaceRef s, float *out, int oc, int seq) {
    if (!out || oc <= 0 || seq <= 0) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, kIOSurfaceLockReadOnly, &base,
                               "dynamic fp16 read")) return false;
    cvt_f16_f32(out, (_Float16 *)base, oc * seq);
    return io_surface_unlock_checked(s, kIOSurfaceLockReadOnly,
                                     "dynamic fp16 read");
}

// Compile MIL to ANE kernel
static void free_kern(Kern *k);
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)jishui_calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->tmpDir = (void*)CFBridgingRetain(td);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    if (!k->ioIn || !k->ioOut) {
        printf("  [compile] IOSurface allocation failed\n");
        free_kern(k);
        return NULL;
    }
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id request = (wI && wO) ? ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0) : nil;
    if (!request) {
        printf("  [compile] ANE request allocation failed\n");
        free_kern(k);
        return NULL;
    }
    k->request = (void*)CFBridgingRetain(request);
    return k;
    }
}
static void free_kern(Kern *k) {
    if (!k) return;
    if (k->model) {
        id mdl = (__bridge id)k->model; NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    }
    if (k->ioIn) CFRelease(k->ioIn);
    if (k->ioOut) CFRelease(k->ioOut);
    if (k->tmpDir)
        [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    if (k->model) CFRelease(k->model);
    if (k->request) CFRelease(k->request);
    if (k->tmpDir) CFRelease(k->tmpDir);
    free(k);
}
static bool ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) printf("ANE evaluate failed: %s\n", e ? [[e description] UTF8String] : "no error");
    return ok;
}
static bool ane_eval_req(Kern *k, void *request) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) printf("ANE evaluate failed: %s\n", e ? [[e description] UTF8String] : "no error");
    return ok;
}
static void *make_request(Kern *k, IOSurfaceRef ioIn) {
    if (!k || !k->ioOut || !ioIn) return NULL;
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    if (!wI || !wO) return NULL;
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return req ? (void*)CFBridgingRetain(req) : NULL;
}

// ===== Per-layer weight staging for GQA =====
// sdpaFwd: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16 — no Wo (separate kernel)
//   Wq: [DIM, Q_DIM], Wk: [DIM, KV_DIM], Wv: [DIM, KV_DIM]
#define SDPA_FWD_SP (SEQ + Q_DIM + KV_DIM + KV_DIM)
static bool stage_sdpa_fwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "sdpa forward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ,                   Wq + d*Q_DIM, Q_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM,             Wk + d*KV_DIM, KV_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM+KV_DIM,     Wv + d*KV_DIM, KV_DIM);
    }
    return io_surface_unlock_checked(s, 0, "sdpa forward weights");
}
static bool write_sdpa_fwd_acts(IOSurfaceRef s, const float *xnorm) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "sdpa forward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*SDPA_FWD_SP, xnorm + d*SEQ, SEQ);
    return io_surface_unlock_checked(s, 0, "sdpa forward activations");
}

// woFwd: [1, Q_DIM, 1, SEQ + DIM] fp16 — Wo: [Q_DIM, DIM]
#define WO_FWD_SP (SEQ + DIM)
static bool stage_wo_fwd_weights(IOSurfaceRef s, const float *Wo) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Wo forward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wo + d*DIM, DIM);
    return io_surface_unlock_checked(s, 0, "Wo forward weights");
}
static bool write_wo_fwd_acts(IOSurfaceRef s, const float *attn_out) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Wo forward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP, attn_out + d*SEQ, SEQ);
    return io_surface_unlock_checked(s, 0, "Wo forward activations");
}

// ffnFused: [1, DIM, 1, 2*SEQ+3*HIDDEN] fp16
#define FFN_FUSED_SP (2*SEQ + 3*HIDDEN)
static bool stage_ffn_fused_weights(IOSurfaceRef s,
                                     const float *W1t, const float *W3t, const float *W2_orig) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN forward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ,          W1t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+HIDDEN,   W3t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+2*HIDDEN, W2_orig + d*HIDDEN, HIDDEN);
    }
    return io_surface_unlock_checked(s, 0, "FFN forward weights");
}
static bool write_ffn_fused_acts(IOSurfaceRef s, const float *x2norm, const float *x2) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN forward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP,       x2norm + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ, x2 + d*SEQ, SEQ);
    }
    return io_surface_unlock_checked(s, 0, "FFN forward activations");
}

// ffnBwdW2t: [1, DIM, 1, SEQ+HIDDEN] fp16
#define FFN_BWD_W2T_SP (SEQ + HIDDEN)
static bool stage_ffn_bwd_w2t_weights(IOSurfaceRef s, const float *W2) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN W2 backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP + SEQ, W2 + d*HIDDEN, HIDDEN);
    return io_surface_unlock_checked(s, 0, "FFN W2 backward weights");
}
static bool write_ffn_bwd_w2t_acts_scaled(IOSurfaceRef s, const float *dffn,
                                           float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN W2 backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_scaled_f16(buf + d*FFN_BWD_W2T_SP, dffn + d*SEQ, scale, SEQ);
    return io_surface_unlock_checked(s, 0, "FFN W2 backward activations");
}
static bool write_ffn_bwd_w2t_acts(IOSurfaceRef s, const float *dffn) {
    return write_ffn_bwd_w2t_acts_scaled(s, dffn, 1.0f);
}

// ffnBwdW13t: [1, HIDDEN, 1, 2*SEQ+2*DIM] fp16
#define FFN_BWD_W13T_SP (2*SEQ + 2*DIM)
static bool stage_ffn_bwd_w13t_weights(IOSurfaceRef s, const float *W1, const float *W3) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN W1/W3 backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ,       W1 + d*DIM, DIM);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ + DIM, W3 + d*DIM, DIM);
    }
    return io_surface_unlock_checked(s, 0, "FFN W1/W3 backward weights");
}
static bool write_ffn_bwd_w13t_acts_scaled(IOSurfaceRef s, const float *dh1,
                                            const float *dh3, float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "FFN W1/W3 backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_scaled_f16(buf + d*FFN_BWD_W13T_SP,       dh1 + d*SEQ, scale, SEQ);
        cvt_f32_scaled_f16(buf + d*FFN_BWD_W13T_SP + SEQ, dh3 + d*SEQ, scale, SEQ);
    }
    return io_surface_unlock_checked(s, 0, "FFN W1/W3 backward activations");
}
static bool write_ffn_bwd_w13t_acts(IOSurfaceRef s, const float *dh1, const float *dh3) {
    return write_ffn_bwd_w13t_acts_scaled(s, dh1, dh3, 1.0f);
}

// wotBwd: [1, DIM, 1, SEQ+Q_DIM] fp16 — Wo is [DIM, Q_DIM], matmul gives Wo^T @ dy
#define WOT_BWD_SP (SEQ + Q_DIM)
static bool stage_wot_bwd_weights(IOSurfaceRef s, const float *Wo) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Wo backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP + SEQ, Wo + d*Q_DIM, Q_DIM);
    return io_surface_unlock_checked(s, 0, "Wo backward weights");
}
static bool write_wot_bwd_acts(IOSurfaceRef s, const float *dy) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Wo backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP, dy + d*SEQ, SEQ);
    return io_surface_unlock_checked(s, 0, "Wo backward activations");
}

// qBwd: [1, Q_DIM, 1, SEQ+DIM] fp16 — Wq is [Q_DIM, DIM], matmul gives Wq^T @ dq
#define Q_BWD_SP (SEQ + DIM)
static bool stage_q_bwd_weights(IOSurfaceRef s, const float *Wq) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Q backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP + SEQ, Wq + d*DIM, DIM);
    return io_surface_unlock_checked(s, 0, "Q backward weights");
}
static bool write_q_bwd_acts_scaled(IOSurfaceRef s, const float *dq,
                                    float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "Q backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_scaled_f16(buf + d*Q_BWD_SP, dq + d*SEQ, scale, SEQ);
    return io_surface_unlock_checked(s, 0, "Q backward activations");
}
static bool write_q_bwd_acts(IOSurfaceRef s, const float *dq) {
    return write_q_bwd_acts_scaled(s, dq, 1.0f);
}

// kvBwd: [1, KV_DIM, 1, 2*SEQ+2*DIM] fp16 — dk @ Wk + dv @ Wv → dx_kv
#define KV_BWD_SP (2*SEQ + 2*DIM)
static bool stage_kv_bwd_weights(IOSurfaceRef s, const float *Wk, const float *Wv) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "KV backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ,       Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ + DIM, Wv + d*DIM, DIM);
    }
    return io_surface_unlock_checked(s, 0, "KV backward weights");
}
static bool write_kv_bwd_acts_scaled(IOSurfaceRef s, const float *dk,
                                     const float *dv, float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "KV backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_scaled_f16(buf + d*KV_BWD_SP,       dk + d*SEQ, scale, SEQ);
        cvt_f32_scaled_f16(buf + d*KV_BWD_SP + SEQ, dv + d*SEQ, scale, SEQ);
    }
    return io_surface_unlock_checked(s, 0, "KV backward activations");
}
static bool write_kv_bwd_acts(IOSurfaceRef s, const float *dk, const float *dv) {
    return write_kv_bwd_acts_scaled(s, dk, dv, 1.0f);
}
static bool write_wot_bwd_acts_scaled(IOSurfaceRef s, const float *dy, float scale) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "scaled Wo backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < DIM; d++)
        cvt_f32_scaled_f16(buf + d*WOT_BWD_SP, dy + d*SEQ, scale, SEQ);
    return io_surface_unlock_checked(s, 0, "scaled Wo backward activations");
}

// qkvBwd is an MHA-only fusion of qBwd and kvBwd. It returns dx_q and the
// fp16-summed dx_kv used by the established split graph.
#define QKV_BWD_SP (3*SEQ + 3*DIM)
static bool stage_qkv_bwd_weights(IOSurfaceRef s, const float *Wq,
                                  const float *Wk, const float *Wv) {
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "QKV backward weights")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++) {
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ,       Wq + d*DIM, DIM);
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ+DIM,   Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ+2*DIM, Wv + d*DIM, DIM);
    }
    return io_surface_unlock_checked(s, 0, "QKV backward weights");
}
static bool write_qkv_bwd_acts_scaled(IOSurfaceRef s, const float *dq,
                                      const float *dk, const float *dv,
                                      float scale) {
    if (!isfinite(scale) || scale <= 0.0f) return false;
    void *base = NULL;
    if (!io_surface_lock_base(s, 0, &base, "QKV backward activations")) return false;
    _Float16 *buf = (_Float16*)base;
    for (int d = 0; d < Q_DIM; d++) {
        cvt_f32_scaled_f16(buf + d*QKV_BWD_SP,       dq + d*SEQ, scale, SEQ);
        cvt_f32_scaled_f16(buf + d*QKV_BWD_SP + SEQ, dk + d*SEQ, scale, SEQ);
        cvt_f32_scaled_f16(buf + d*QKV_BWD_SP + 2*SEQ, dv + d*SEQ, scale, SEQ);
    }
    return io_surface_unlock_checked(s, 0, "QKV backward activations");
}
static bool write_qkv_bwd_acts(IOSurfaceRef s, const float *dq,
                               const float *dk, const float *dv) {
    return write_qkv_bwd_acts_scaled(s, dq, dk, dv, 1.0f);
}

// Free per-layer surfaces and requests
static void free_per_layer(PerLayerSurfaces *pls, PerLayerRequests *plr) {
    for (int L = 0; L < NLAYERS; L++) {
#define RELEASE_IF_SET(value) do { if (value) CFRelease(value); } while (0)
        RELEASE_IF_SET(pls[L].sdpaFwd_in); RELEASE_IF_SET(pls[L].woFwd_in); RELEASE_IF_SET(pls[L].ffnFused_in);
        RELEASE_IF_SET(pls[L].ffnBwdW2t_in); RELEASE_IF_SET(pls[L].ffnBwdW13t_in);
        RELEASE_IF_SET(pls[L].wotBwd_in); RELEASE_IF_SET(pls[L].qBwd_in);
        RELEASE_IF_SET(pls[L].kvBwd_in); RELEASE_IF_SET(pls[L].qkvBwd_in);
        RELEASE_IF_SET(plr[L].sdpaFwd); RELEASE_IF_SET(plr[L].woFwd); RELEASE_IF_SET(plr[L].ffnFused);
        RELEASE_IF_SET(plr[L].ffnBwdW2t); RELEASE_IF_SET(plr[L].ffnBwdW13t);
        RELEASE_IF_SET(plr[L].wotBwd); RELEASE_IF_SET(plr[L].qBwd);
        RELEASE_IF_SET(plr[L].kvBwd); RELEASE_IF_SET(plr[L].qkvBwd);
#undef RELEASE_IF_SET
    }
}

// GQA helpers: tile KV from KV_HEADS to HEADS, and reduce HEADS to KV_HEADS
// tile_kv: input [KV_DIM, SEQ], output [Q_DIM, SEQ]
// Each KV head is duplicated GQA_RATIO times
static void gqa_tile_kv(float *out, const float *in, int seq) {
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            memcpy(out + q_head * HD * seq, in + kv * HD * seq, HD * seq * sizeof(float));
        }
    }
}
// reduce_kv: input [Q_DIM, SEQ], output [KV_DIM, SEQ]
// Sum contributions from Q heads sharing each KV head
static void gqa_reduce_kv(float *out, const float *in, int seq) {
    memset(out, 0, KV_DIM * seq * sizeof(float));
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            const float *src = in + q_head * HD * seq;
            float *dst = out + kv * HD * seq;
            for (int i = 0; i < HD * seq; i++)
                dst[i] += src[i];
        }
    }
}
