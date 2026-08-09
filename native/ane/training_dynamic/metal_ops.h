// metal_ops.h -- optional Metal elementwise/reduction backend for the native
// ANE trainer.  All calls are synchronous at ANE dependency boundaries.
#pragma once

#import <IOKit/IOReturn.h>
#import <Metal/Metal.h>

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_metal_norm_fwd = nil;
static id<MTLComputePipelineState> g_metal_norm_dx = nil;
static id<MTLComputePipelineState> g_metal_norm_params = nil;
static id<MTLComputePipelineState> g_metal_silu_bwd = nil;
static id<MTLComputePipelineState> g_metal_sdpa_io = nil;
static id<MTLComputePipelineState> g_metal_ffn_unpack = nil;
static id<MTLBuffer> g_metal_means = nil;
static id<MTLBuffer> g_metal_invstd = nil;
static id<MTLBuffer> g_metal_dgamma = nil;
static id<MTLBuffer> g_metal_dbeta = nil;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *g_metal_shared_buffers = nil;
static bool g_metal_shadow = false;
static int g_metal_forward_calls = 0, g_metal_backward_calls = 0;
static int g_metal_silu_calls = 0;
static int g_metal_sdpa_io_calls = 0, g_metal_ffn_io_calls = 0;

static void metal_ops_set_shadow(bool enabled) { g_metal_shadow = enabled; }

static float metal_max_abs_difference(const float *expected, const float *actual,
                                      size_t count, float *actual_max) {
    float difference = 0.0f, magnitude = 0.0f;
    for (size_t i=0; i<count; i++) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) return INFINITY;
        difference = fmaxf(difference, fabsf(expected[i] - actual[i]));
        magnitude = fmaxf(magnitude, fabsf(actual[i]));
    }
    *actual_max = magnitude;
    return difference;
}

static void metal_silu_cpu_reference(float *dh1, float *dh3,
                                     const float *h1, const float *h3,
                                     const float *dsilu, size_t count) {
    int n = (int)count;
    float minus1 = -1.0f, one = 1.0f;
    float *sigmoid = (float *)jishui_malloc(count * sizeof(float));
    float *derivative = (float *)jishui_malloc(count * sizeof(float));
    vDSP_vsmul(h1, 1, &minus1, sigmoid, 1, (vDSP_Length)count);
    vvexpf(sigmoid, sigmoid, &n);
    vDSP_vsadd(sigmoid, 1, &one, sigmoid, 1, (vDSP_Length)count);
    vvrecf(sigmoid, sigmoid, &n);
    vDSP_vmul(h1, 1, sigmoid, 1, dh3, 1, (vDSP_Length)count);
    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)count);
    vDSP_vsadd(sigmoid, 1, &minus1, derivative, 1, (vDSP_Length)count);
    vDSP_vneg(derivative, 1, derivative, 1, (vDSP_Length)count);
    vDSP_vmul(h1, 1, derivative, 1, derivative, 1, (vDSP_Length)count);
    vDSP_vsadd(derivative, 1, &one, derivative, 1, (vDSP_Length)count);
    vDSP_vmul(sigmoid, 1, derivative, 1, derivative, 1, (vDSP_Length)count);
    vDSP_vmul(dsilu, 1, h3, 1, dh1, 1, (vDSP_Length)count);
    vDSP_vmul(dh1, 1, derivative, 1, dh1, 1, (vDSP_Length)count);
    free(derivative);
    free(sigmoid);
}

static id<MTLBuffer> metal_wrap_shared(const void *pointer, size_t bytes) {
    if (!pointer || ((uintptr_t)pointer % JISHUI_SHARED_ALIGNMENT) != 0) {
        printf("Metal shared buffer is not 16 KiB aligned: %p\n", pointer);
        return nil;
    }
    NSValue *key = [NSValue valueWithPointer:pointer];
    id<MTLBuffer> buffer = [g_metal_shared_buffers objectForKey:key];
    if (!buffer) {
        buffer = [g_metal_device newBufferWithBytesNoCopy:(void *)pointer
                                                   length:jishui_shared_bytes(bytes)
                                                  options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (buffer) [g_metal_shared_buffers setObject:buffer forKey:key];
    } else if (buffer.length < jishui_shared_bytes(bytes)) {
        printf("Metal shared buffer length mismatch for %p\n", pointer);
        return nil;
    }
    return buffer;
}

static id<MTLBuffer> metal_wrap_iosurface(IOSurfaceRef surface, const char *name) {
    if (!surface) {
        printf("Metal IOSurface %s is NULL\n", name);
        return nil;
    }
    IOReturn result = IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    if (result != kIOReturnSuccess) {
        printf("Metal IOSurface %s lock failed: 0x%x\n", name, result);
        return nil;
    }
    void *base = IOSurfaceGetBaseAddress(surface);
    size_t bytes = IOSurfaceGetAllocSize(surface);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    if (!base || bytes == 0) {
        printf("Metal IOSurface %s has no CPU mapping\n", name);
        return nil;
    }
    return metal_wrap_shared(base, bytes);
}

static bool metal_iosurface_fence(IOSurfaceRef surface, IOSurfaceLockOptions options,
                                  const char *name) {
    IOReturn result = IOSurfaceLock(surface, options, NULL);
    if (result != kIOReturnSuccess) {
        printf("Metal IOSurface %s fence lock failed: 0x%x\n", name, result);
        return false;
    }
    if (!IOSurfaceGetBaseAddress(surface)) {
        printf("Metal IOSurface %s lost its CPU mapping\n", name);
        IOSurfaceUnlock(surface, options, NULL);
        return false;
    }
    result = IOSurfaceUnlock(surface, options, NULL);
    if (result != kIOReturnSuccess) {
        printf("Metal IOSurface %s fence unlock failed: 0x%x\n", name, result);
        return false;
    }
    return true;
}

static bool metal_command_succeeded(id<MTLCommandBuffer> command, const char *operation) {
    if (!command) {
        printf("Metal %s failed: unable to allocate command buffer\n", operation);
        return false;
    }
    if (command.status == MTLCommandBufferStatusCompleted) return true;
    NSString *message = command.error ? command.error.localizedDescription : @"unknown Metal error";
    printf("Metal %s failed: %s\n", operation, message.UTF8String);
    return false;
}

static bool metal_ops_init(void) {
    g_metal_device = MTLCreateSystemDefaultDevice();
    if (!g_metal_device) {
        printf("Metal backend requested but no device is available\n");
        return false;
    }
    g_metal_queue = [g_metal_device newCommandQueue];
    g_metal_shared_buffers = [NSMutableDictionary dictionary];
    NSError *error = nil;
    const char *source_c =
         "#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void norm_fwd(device const float *x [[buffer(0)]], device const float *g [[buffer(1)]], device const float *b [[buffer(2)]], device float *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant uint &seq [[buffer(5)]], uint lane [[thread_index_in_threadgroup]], uint token [[threadgroup_position_in_grid]]) {\n"
         "  threadgroup float r[256]; float part=0.0f;\n"
         "  for(uint f=lane;f<dim;f+=256) part+=x[f*seq+token]; r[lane]=part; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s)r[lane]+=r[lane+s];threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  float mean=r[0]/float(dim);threadgroup_barrier(mem_flags::mem_threadgroup);part=0.0f;\n"
         "  for(uint f=lane;f<dim;f+=256){float z=x[f*seq+token]-mean;part+=z*z;} r[lane]=part;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s)r[lane]+=r[lane+s];threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  float inv=rsqrt(r[0]/float(dim)+1.0e-5f);\n"
         "  for(uint f=lane;f<dim;f+=256){uint i=f*seq+token;out[i]=(x[i]-mean)*inv*g[f]+b[f];}\n"
         "}\n"
         "kernel void norm_dx(device const float *dy [[buffer(0)]], device const float *x [[buffer(1)]], device const float *g [[buffer(2)]], device float *dx [[buffer(3)]], device float *means [[buffer(4)]], device float *invs [[buffer(5)]], constant uint &dim [[buffer(6)]], constant uint &seq [[buffer(7)]], uint lane [[thread_index_in_threadgroup]], uint token [[threadgroup_position_in_grid]]) {\n"
         "  threadgroup float a[256]; threadgroup float c[256]; float part=0.0f;\n"
         "  for(uint f=lane;f<dim;f+=256)part+=x[f*seq+token];a[lane]=part;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s)a[lane]+=a[lane+s];threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  float mean=a[0]/float(dim);threadgroup_barrier(mem_flags::mem_threadgroup);part=0.0f;for(uint f=lane;f<dim;f+=256){float z=x[f*seq+token]-mean;part+=z*z;}a[lane]=part;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s)a[lane]+=a[lane+s];threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  float inv=rsqrt(a[0]/float(dim)+1.0e-5f);if(lane==0){means[token]=mean;invs[token]=inv;}threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  float s1=0.0f,s2=0.0f;for(uint f=lane;f<dim;f+=256){uint i=f*seq+token;float v=dy[i]*g[f];float h=(x[i]-mean)*inv;s1+=v;s2+=v*h;}a[lane]=s1;c[lane]=s2;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s){a[lane]+=a[lane+s];c[lane]+=c[lane+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  s1=a[0];s2=c[0];for(uint f=lane;f<dim;f+=256){uint i=f*seq+token;float h=(x[i]-mean)*inv;float v=dy[i]*g[f];dx[i]=inv/float(dim)*(float(dim)*v-s1-h*s2);}\n"
         "}\n"
         "kernel void norm_params(device const float *dy [[buffer(0)]], device const float *x [[buffer(1)]], device const float *means [[buffer(2)]], device const float *invs [[buffer(3)]], device float *dg [[buffer(4)]], device float *db [[buffer(5)]], constant uint &dim [[buffer(6)]], constant uint &seq [[buffer(7)]], uint lane [[thread_index_in_threadgroup]], uint feature [[threadgroup_position_in_grid]]) {\n"
         "  threadgroup float a[256];threadgroup float c[256];float sg=0.0f,sb=0.0f;for(uint t=lane;t<seq;t+=256){uint i=feature*seq+t;float v=dy[i];sg+=v*(x[i]-means[t])*invs[t];sb+=v;}a[lane]=sg;c[lane]=sb;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for(uint s=128;s>0;s>>=1){if(lane<s){a[lane]+=a[lane+s];c[lane]+=c[lane+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         "  if(lane==0){dg[feature]=a[0];db[feature]=c[0];}\n"
         "}\n"
         "kernel void silu_bwd(device const float *h1 [[buffer(0)]], device const float *h3 [[buffer(1)]], device const float *ds [[buffer(2)]], device float *dh1 [[buffer(3)]], device float *dh3 [[buffer(4)]], constant uint &count [[buffer(5)]], uint i [[thread_position_in_grid]]) {\n"
         "  if(i>=count)return;float x=h1[i];float sig=1.0f/(1.0f+exp(-x));float u=ds[i];dh3[i]=u*x*sig;dh1[i]=u*h3[i]*sig*(1.0f+x*(1.0f-sig));\n"
         "}\n"
         "kernel void sdpa_unpack_pack(device const half *src [[buffer(0)]], device half *wo [[buffer(1)]], device float *attn [[buffer(2)]], device float *q [[buffer(3)]], device float *k [[buffer(4)]], device float *v [[buffer(5)]], constant uint &seq [[buffer(6)]], constant uint &qdim [[buffer(7)]], constant uint &kvdim [[buffer(8)]], constant uint &wo_stride [[buffer(9)]], uint i [[thread_position_in_grid]]) {\n"
         "  uint qc=qdim*seq,kc=kvdim*seq;\n"
         "  if(i<qc){half a=src[i];attn[i]=float(a);q[i]=float(src[qc+i]);uint row=i/seq;wo[row*wo_stride+i-row*seq]=a;}\n"
         "  if(i<kc){k[i]=float(src[2*qc+i]);v[i]=float(src[2*qc+kc+i]);}\n"
         "}\n"
         "kernel void ffn_unpack(device const half *src [[buffer(0)]], device float *x [[buffer(1)]], device float *h1 [[buffer(2)]], device float *h3 [[buffer(3)]], device float *silu [[buffer(4)]], constant uint &dim_count [[buffer(5)]], constant uint &hidden_count [[buffer(6)]], uint i [[thread_position_in_grid]]) {\n"
         "  if(i<dim_count)x[i]=float(src[i]);\n"
         "  if(i<hidden_count){h1[i]=float(src[dim_count+i]);h3[i]=float(src[dim_count+hidden_count+i]);silu[i]=float(src[dim_count+2*hidden_count+i]);}\n"
         "}\n";
    NSString *source = [NSString stringWithUTF8String:source_c];
    id<MTLLibrary> library = [g_metal_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        printf("Metal kernel compilation failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }
#define BUILD_PIPELINE(target, name) do { \
    id<MTLFunction> fn = [library newFunctionWithName:@name]; \
    target = [g_metal_device newComputePipelineStateWithFunction:fn error:&error]; \
    if (!target) { printf("Metal pipeline " name " failed: %s\n", error.localizedDescription.UTF8String); return false; } \
} while (0)
    BUILD_PIPELINE(g_metal_norm_fwd, "norm_fwd");
    BUILD_PIPELINE(g_metal_norm_dx, "norm_dx");
    BUILD_PIPELINE(g_metal_norm_params, "norm_params");
    BUILD_PIPELINE(g_metal_silu_bwd, "silu_bwd");
    BUILD_PIPELINE(g_metal_sdpa_io, "sdpa_unpack_pack");
    BUILD_PIPELINE(g_metal_ffn_unpack, "ffn_unpack");
#undef BUILD_PIPELINE
    g_metal_means = [g_metal_device newBufferWithLength:jishui_shared_bytes(SEQ*4)
                                                 options:MTLResourceStorageModeShared];
    g_metal_invstd = [g_metal_device newBufferWithLength:jishui_shared_bytes(SEQ*4)
                                                  options:MTLResourceStorageModeShared];
    g_metal_dgamma = [g_metal_device newBufferWithLength:jishui_shared_bytes(DIM*4)
                                                  options:MTLResourceStorageModeShared];
    g_metal_dbeta = [g_metal_device newBufferWithLength:jishui_shared_bytes(DIM*4)
                                                 options:MTLResourceStorageModeShared];
    if (!g_metal_queue || !g_metal_means || !g_metal_invstd || !g_metal_dgamma || !g_metal_dbeta)
        return false;
    printf("Metal operator backend: %s\n", g_metal_device.name.UTF8String);
    return true;
}

static void metal_ops_shutdown(void) {
    [g_metal_shared_buffers removeAllObjects]; g_metal_shared_buffers = nil;
    g_metal_dbeta = nil; g_metal_dgamma = nil; g_metal_invstd = nil; g_metal_means = nil;
    g_metal_ffn_unpack = nil; g_metal_sdpa_io = nil; g_metal_silu_bwd = nil;
    g_metal_norm_params = nil; g_metal_norm_dx = nil;
    g_metal_norm_fwd = nil; g_metal_queue = nil; g_metal_device = nil;
}

static bool metal_layernorm_forward(float *out, const float *x,
                                    const float *gamma, const float *beta) {
    @autoreleasepool {
        id<MTLBuffer> bx = metal_wrap_shared(x, DIM*SEQ*4);
        id<MTLBuffer> bg = metal_wrap_shared(gamma, DIM*4);
        id<MTLBuffer> bb = metal_wrap_shared(beta, DIM*4);
        id<MTLBuffer> bo = metal_wrap_shared(out, DIM*SEQ*4);
        if (!bx || !bg || !bb || !bo) return false;
        id<MTLCommandBuffer> command = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        [encoder setComputePipelineState:g_metal_norm_fwd];
        [encoder setBuffer:bx offset:0 atIndex:0]; [encoder setBuffer:bg offset:0 atIndex:1];
        [encoder setBuffer:bb offset:0 atIndex:2]; [encoder setBuffer:bo offset:0 atIndex:3];
        uint32_t dim=DIM, seq=SEQ;
        [encoder setBytes:&dim length:4 atIndex:4]; [encoder setBytes:&seq length:4 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(SEQ,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (!metal_command_succeeded(command, "LayerNorm forward")) return false;
        if (g_metal_shadow) {
            float *expected=(float *)jishui_malloc(DIM*SEQ*4), actual_max=0.0f;
            norm_forward(expected,x,gamma,beta,DIM,SEQ);
            float difference=metal_max_abs_difference(expected,out,(size_t)DIM*SEQ,&actual_max);
            printf("  metal_shadow fwd[%d] max_diff=%.3e actual_max=%.3e\n",
                   g_metal_forward_calls++, difference, actual_max);
            free(expected);
            if (!isfinite(difference) || difference > fmaxf(1.0e-4f, actual_max*1.0e-5f))
                return false;
        }
        return true;
    }
}

static bool metal_layernorm_backward(float *dx, float *dg, float *db,
                                     const float *dy, const float *x,
                                     const float *gamma) {
    @autoreleasepool {
        id<MTLBuffer> bdy = metal_wrap_shared(dy, DIM*SEQ*4);
        id<MTLBuffer> bx = metal_wrap_shared(x, DIM*SEQ*4);
        id<MTLBuffer> bg = metal_wrap_shared(gamma, DIM*4);
        id<MTLBuffer> bdx = metal_wrap_shared(dx, DIM*SEQ*4);
        if (!bdy || !bx || !bg || !bdx) return false;
        uint32_t dim=DIM, seq=SEQ;
        id<MTLCommandBuffer> command = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        [encoder setComputePipelineState:g_metal_norm_dx];
        [encoder setBuffer:bdy offset:0 atIndex:0]; [encoder setBuffer:bx offset:0 atIndex:1];
        [encoder setBuffer:bg offset:0 atIndex:2]; [encoder setBuffer:bdx offset:0 atIndex:3];
        [encoder setBuffer:g_metal_means offset:0 atIndex:4]; [encoder setBuffer:g_metal_invstd offset:0 atIndex:5];
        [encoder setBytes:&dim length:4 atIndex:6]; [encoder setBytes:&seq length:4 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(SEQ,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:g_metal_norm_params];
        [encoder setBuffer:bdy offset:0 atIndex:0]; [encoder setBuffer:bx offset:0 atIndex:1];
        [encoder setBuffer:g_metal_means offset:0 atIndex:2]; [encoder setBuffer:g_metal_invstd offset:0 atIndex:3];
        [encoder setBuffer:g_metal_dgamma offset:0 atIndex:4]; [encoder setBuffer:g_metal_dbeta offset:0 atIndex:5];
        [encoder setBytes:&dim length:4 atIndex:6]; [encoder setBytes:&seq length:4 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (!metal_command_succeeded(command, "LayerNorm backward")) return false;
        if (g_metal_shadow) {
            float *expected_dx=(float *)jishui_calloc((size_t)DIM*SEQ,4);
            float *expected_dg=(float *)jishui_calloc(DIM,4), *expected_db=(float *)jishui_calloc(DIM,4);
            float dx_max=0.0f,dg_max=0.0f,db_max=0.0f;
            norm_backward(expected_dx,expected_dg,expected_db,dy,x,gamma,DIM,SEQ);
            float dx_diff=metal_max_abs_difference(expected_dx,dx,(size_t)DIM*SEQ,&dx_max);
            float dg_diff=metal_max_abs_difference(expected_dg,(float *)g_metal_dgamma.contents,DIM,&dg_max);
            float db_diff=metal_max_abs_difference(expected_db,(float *)g_metal_dbeta.contents,DIM,&db_max);
            printf("  metal_shadow bwd[%d] dx=%.3e/%.3e dg=%.3e/%.3e db=%.3e/%.3e\n",
                   g_metal_backward_calls++,dx_diff,dx_max,dg_diff,dg_max,db_diff,db_max);
            free(expected_dx);free(expected_dg);free(expected_db);
            if (!isfinite(dx_diff) || !isfinite(dg_diff) || !isfinite(db_diff) ||
                dx_diff > fmaxf(1.0e-4f,dx_max*1.0e-5f) ||
                dg_diff > fmaxf(1.0e-3f,dg_max*1.0e-5f) ||
                db_diff > fmaxf(1.0e-3f,db_max*1.0e-5f)) return false;
        }
        vDSP_vadd(dg,1,(float *)g_metal_dgamma.contents,1,dg,1,(vDSP_Length)DIM);
        vDSP_vadd(db,1,(float *)g_metal_dbeta.contents,1,db,1,(vDSP_Length)DIM);
        return true;
    }
}

static bool metal_silu_backward(float *dh1, float *dh3, const float *h1,
                                const float *h3, const float *dsilu) {
    @autoreleasepool {
        size_t count=(size_t)HIDDEN*SEQ, bytes=count*4;
        id<MTLBuffer> bh1=metal_wrap_shared(h1,bytes), bh3=metal_wrap_shared(h3,bytes);
        id<MTLBuffer> bds=metal_wrap_shared(dsilu,bytes), bdh1=metal_wrap_shared(dh1,bytes);
        id<MTLBuffer> bdh3=metal_wrap_shared(dh3,bytes);
        if(!bh1||!bh3||!bds||!bdh1||!bdh3)return false;
        id<MTLCommandBuffer> command=[g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (!command || !encoder) return false;
        [encoder setComputePipelineState:g_metal_silu_bwd];
        [encoder setBuffer:bh1 offset:0 atIndex:0]; [encoder setBuffer:bh3 offset:0 atIndex:1];
        [encoder setBuffer:bds offset:0 atIndex:2]; [encoder setBuffer:bdh1 offset:0 atIndex:3];
        [encoder setBuffer:bdh3 offset:0 atIndex:4]; uint32_t n=(uint32_t)count;
        [encoder setBytes:&n length:4 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (!metal_command_succeeded(command, "SiLU backward")) return false;
        if (g_metal_shadow) {
            float *expected_dh1=(float *)jishui_malloc(bytes);
            float *expected_dh3=(float *)jishui_malloc(bytes);
            float dh1_max=0.0f, dh3_max=0.0f;
            metal_silu_cpu_reference(expected_dh1,expected_dh3,h1,h3,dsilu,count);
            float dh1_diff=metal_max_abs_difference(expected_dh1,dh1,count,&dh1_max);
            float dh3_diff=metal_max_abs_difference(expected_dh3,dh3,count,&dh3_max);
            printf("  metal_shadow silu[%d] dh1=%.3e/%.3e dh3=%.3e/%.3e\n",
                   g_metal_silu_calls++,dh1_diff,dh1_max,dh3_diff,dh3_max);
            free(expected_dh1);free(expected_dh3);
            if (!isfinite(dh1_diff) || !isfinite(dh3_diff) ||
                dh1_diff > fmaxf(1.0e-5f,dh1_max*2.0e-5f) ||
                dh3_diff > fmaxf(1.0e-5f,dh3_max*2.0e-5f)) return false;
        }
        return true;
    }
}

static bool metal_sdpa_unpack_and_pack(IOSurfaceRef source, IOSurfaceRef wo_input,
                                       float *attn, float *q, float *k, float *v) {
    @autoreleasepool {
        if (!metal_iosurface_fence(source, kIOSurfaceLockReadOnly, "sdpa output"))
            return false;
        id<MTLBuffer> bsrc=metal_wrap_iosurface(source,"sdpa output");
        id<MTLBuffer> bwo=metal_wrap_iosurface(wo_input,"wo input");
        id<MTLBuffer> battn=metal_wrap_shared(attn,(size_t)Q_DIM*SEQ*4);
        id<MTLBuffer> bq=metal_wrap_shared(q,(size_t)Q_DIM*SEQ*4);
        id<MTLBuffer> bk=metal_wrap_shared(k,(size_t)KV_DIM*SEQ*4);
        id<MTLBuffer> bv=metal_wrap_shared(v,(size_t)KV_DIM*SEQ*4);
        if(!bsrc||!bwo||!battn||!bq||!bk||!bv)return false;
        id<MTLCommandBuffer> command=[g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(!command||!encoder)return false;
        [encoder setComputePipelineState:g_metal_sdpa_io];
        [encoder setBuffer:bsrc offset:0 atIndex:0]; [encoder setBuffer:bwo offset:0 atIndex:1];
        [encoder setBuffer:battn offset:0 atIndex:2]; [encoder setBuffer:bq offset:0 atIndex:3];
        [encoder setBuffer:bk offset:0 atIndex:4]; [encoder setBuffer:bv offset:0 atIndex:5];
        uint32_t seq=SEQ,qdim=Q_DIM,kvdim=KV_DIM,wo_stride=WO_FWD_SP;
        [encoder setBytes:&seq length:4 atIndex:6]; [encoder setBytes:&qdim length:4 atIndex:7];
        [encoder setBytes:&kvdim length:4 atIndex:8]; [encoder setBytes:&wo_stride length:4 atIndex:9];
        size_t count=(size_t)(Q_DIM>KV_DIM?Q_DIM:KV_DIM)*SEQ;
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if(!metal_command_succeeded(command,"SDPA IOSurface unpack/pack"))return false;
        if(!metal_iosurface_fence(wo_input,kIOSurfaceLockReadOnly,"wo input"))return false;
        if(g_metal_shadow){
            if(IOSurfaceLock(source,kIOSurfaceLockReadOnly,NULL)!=kIOReturnSuccess)return false;
            if(IOSurfaceLock(wo_input,kIOSurfaceLockReadOnly,NULL)!=kIOReturnSuccess){
                IOSurfaceUnlock(source,kIOSurfaceLockReadOnly,NULL);return false;
            }
            const _Float16 *src=(_Float16*)IOSurfaceGetBaseAddress(source);
            const _Float16 *dst=(_Float16*)IOSurfaceGetBaseAddress(wo_input);
            float max_diff=0.0f; bool packed=true;
            size_t qc=(size_t)Q_DIM*SEQ,kc=(size_t)KV_DIM*SEQ;
            for(size_t i=0;i<qc;i++){
                max_diff=fmaxf(max_diff,fabsf(attn[i]-(float)src[i]));
                max_diff=fmaxf(max_diff,fabsf(q[i]-(float)src[qc+i]));
                size_t row=i/SEQ,col=i-row*SEQ;
                if(dst[row*WO_FWD_SP+col]!=src[i])packed=false;
            }
            for(size_t i=0;i<kc;i++){
                max_diff=fmaxf(max_diff,fabsf(k[i]-(float)src[2*qc+i]));
                max_diff=fmaxf(max_diff,fabsf(v[i]-(float)src[2*qc+kc+i]));
            }
            IOSurfaceUnlock(wo_input,kIOSurfaceLockReadOnly,NULL);
            IOSurfaceUnlock(source,kIOSurfaceLockReadOnly,NULL);
            printf("  metal_shadow sdpa_io[%d] unpack=%.3e pack=%s\n",
                   g_metal_sdpa_io_calls++,max_diff,packed?"exact":"FAIL");
            if(max_diff!=0.0f||!packed)return false;
        }
        return true;
    }
}

static bool metal_ffn_unpack_output(IOSurfaceRef source, float *x, float *h1,
                                    float *h3, float *silu) {
    @autoreleasepool {
        if(!metal_iosurface_fence(source,kIOSurfaceLockReadOnly,"ffn output"))return false;
        id<MTLBuffer> bsrc=metal_wrap_iosurface(source,"ffn output");
        id<MTLBuffer> bx=metal_wrap_shared(x,(size_t)DIM*SEQ*4);
        id<MTLBuffer> bh1=metal_wrap_shared(h1,(size_t)HIDDEN*SEQ*4);
        id<MTLBuffer> bh3=metal_wrap_shared(h3,(size_t)HIDDEN*SEQ*4);
        id<MTLBuffer> bsilu=metal_wrap_shared(silu,(size_t)HIDDEN*SEQ*4);
        if(!bsrc||!bx||!bh1||!bh3||!bsilu)return false;
        id<MTLCommandBuffer> command=[g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(!command||!encoder)return false;
        [encoder setComputePipelineState:g_metal_ffn_unpack];
        [encoder setBuffer:bsrc offset:0 atIndex:0]; [encoder setBuffer:bx offset:0 atIndex:1];
        [encoder setBuffer:bh1 offset:0 atIndex:2]; [encoder setBuffer:bh3 offset:0 atIndex:3];
        [encoder setBuffer:bsilu offset:0 atIndex:4];
        uint32_t dim_count=DIM*SEQ,hidden_count=HIDDEN*SEQ;
        [encoder setBytes:&dim_count length:4 atIndex:5];
        [encoder setBytes:&hidden_count length:4 atIndex:6];
        size_t count=dim_count>hidden_count?dim_count:hidden_count;
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if(!metal_command_succeeded(command,"FFN IOSurface unpack"))return false;
        if(g_metal_shadow){
            if(IOSurfaceLock(source,kIOSurfaceLockReadOnly,NULL)!=kIOReturnSuccess)return false;
            const _Float16 *src=(_Float16*)IOSurfaceGetBaseAddress(source);
            float max_diff=0.0f;
            size_t dc=(size_t)DIM*SEQ,hc=(size_t)HIDDEN*SEQ;
            for(size_t i=0;i<dc;i++)max_diff=fmaxf(max_diff,fabsf(x[i]-(float)src[i]));
            for(size_t i=0;i<hc;i++){
                max_diff=fmaxf(max_diff,fabsf(h1[i]-(float)src[dc+i]));
                max_diff=fmaxf(max_diff,fabsf(h3[i]-(float)src[dc+hc+i]));
                max_diff=fmaxf(max_diff,fabsf(silu[i]-(float)src[dc+2*hc+i]));
            }
            IOSurfaceUnlock(source,kIOSurfaceLockReadOnly,NULL);
            printf("  metal_shadow ffn_io[%d] unpack=%.3e\n",g_metal_ffn_io_calls++,max_diff);
            if(max_diff!=0.0f)return false;
        }
        return true;
    }
}
