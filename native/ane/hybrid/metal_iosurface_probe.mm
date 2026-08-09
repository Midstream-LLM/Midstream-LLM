// metal_iosurface_probe.mm -- verify that an IOSurface allocation can be
// pre-bound as a shared Metal buffer without copying. This probe never loads
// or calls AppleNeuralEngine.framework.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <IOKit/IOReturn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr size_t kElementCount = 8192;

struct SurfaceBuffer {
  IOSurfaceRef surface = nullptr;
  void *base = nullptr;
  size_t allocation_bytes = 0;
  id<MTLBuffer> metal = nil;
};

static size_t align_up(size_t bytes, size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

static void release_surface_buffer(SurfaceBuffer *buffer) {
  buffer->metal = nil;
  if (buffer->surface) CFRelease(buffer->surface);
  buffer->surface = nullptr;
  buffer->base = nullptr;
  buffer->allocation_bytes = 0;
}

static bool make_surface_buffer(id<MTLDevice> device, size_t requested_bytes,
                                size_t page_bytes, SurfaceBuffer *result) {
  const size_t padded_bytes = align_up(requested_bytes, page_bytes);
  IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(padded_bytes),
    (id)kIOSurfaceHeight: @1,
    (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(padded_bytes),
    (id)kIOSurfaceAllocSize: @(padded_bytes),
    (id)kIOSurfacePixelFormat: @0,
  });
  if (!surface) {
    std::fprintf(stderr, "IOSurfaceCreate failed for %zu bytes\n", padded_bytes);
    return false;
  }

  const IOReturn lock_result = IOSurfaceLock(surface, 0, nullptr);
  if (lock_result != kIOReturnSuccess) {
    std::fprintf(stderr, "IOSurfaceLock failed: 0x%x\n", lock_result);
    CFRelease(surface);
    return false;
  }
  void *base = IOSurfaceGetBaseAddress(surface);
  const size_t allocation_bytes = IOSurfaceGetAllocSize(surface);
  bool valid = base && allocation_bytes >= padded_bytes &&
               reinterpret_cast<uintptr_t>(base) % page_bytes == 0 &&
               allocation_bytes % page_bytes == 0;
  if (valid) std::memset(base, 0, allocation_bytes);
  IOSurfaceUnlock(surface, 0, nullptr);
  if (!valid) {
    std::fprintf(stderr,
                 "IOSurface is not Metal-compatible: base=%p alloc=%zu page=%zu\n",
                 base, allocation_bytes, page_bytes);
    CFRelease(surface);
    return false;
  }

  id<MTLBuffer> metal =
      [device newBufferWithBytesNoCopy:base
                                length:allocation_bytes
                               options:MTLResourceStorageModeShared
                            deallocator:nil];
  if (!metal || metal.contents != base) {
    std::fprintf(stderr, "Metal no-copy binding failed: base=%p contents=%p\n",
                 base, metal ? metal.contents : nullptr);
    CFRelease(surface);
    return false;
  }

  result->surface = surface;
  result->base = base;
  result->allocation_bytes = allocation_bytes;
  result->metal = metal;
  return true;
}

static bool write_input(IOSurfaceRef surface, size_t count, int round) {
  const IOReturn lock_result = IOSurfaceLock(surface, 0, nullptr);
  if (lock_result != kIOReturnSuccess) return false;
  _Float16 *values = static_cast<_Float16 *>(IOSurfaceGetBaseAddress(surface));
  if (!values) {
    IOSurfaceUnlock(surface, 0, nullptr);
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    const int centered = static_cast<int>((i + static_cast<size_t>(round)) % 31) - 15;
    values[i] = static_cast<_Float16>(static_cast<float>(centered) * 0.125f);
  }
  IOSurfaceUnlock(surface, 0, nullptr);
  return true;
}

static bool verify_output(IOSurfaceRef input, IOSurfaceRef output, size_t count,
                          float *maximum_error) {
  const IOReturn input_lock = IOSurfaceLock(input, kIOSurfaceLockReadOnly, nullptr);
  if (input_lock != kIOReturnSuccess) return false;
  const IOReturn output_lock = IOSurfaceLock(output, kIOSurfaceLockReadOnly, nullptr);
  if (output_lock != kIOReturnSuccess) {
    IOSurfaceUnlock(input, kIOSurfaceLockReadOnly, nullptr);
    return false;
  }
  const _Float16 *input_values =
      static_cast<const _Float16 *>(IOSurfaceGetBaseAddress(input));
  const _Float16 *output_values =
      static_cast<const _Float16 *>(IOSurfaceGetBaseAddress(output));
  bool valid = input_values && output_values;
  float error = 0.0f;
  if (valid) {
    for (size_t i = 0; i < count; ++i) {
      const _Float16 expected_half =
          static_cast<_Float16>(static_cast<float>(input_values[i]) * 2.0f + 1.0f);
      error = std::fmax(error, std::fabs(static_cast<float>(output_values[i]) -
                                        static_cast<float>(expected_half)));
    }
  }
  IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, nullptr);
  IOSurfaceUnlock(input, kIOSurfaceLockReadOnly, nullptr);
  *maximum_error = error;
  return valid && error == 0.0f;
}

}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    int rounds = 8;
    if (argc == 3 && std::strcmp(argv[1], "--rounds") == 0)
      rounds = std::atoi(argv[2]);
    else if (argc != 1)
      return 2;
    if (rounds < 1 || rounds > 1000) return 2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 3;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *error = nil;
    NSString *source = @R"metal(
      #include <metal_stdlib>
      using namespace metal;
      kernel void affine_fp16(device const half *input [[buffer(0)]],
                              device half *output [[buffer(1)]],
                              constant uint &count [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
        if (index < count) output[index] = input[index] * half(2.0h) + half(1.0h);
      }
    )metal";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"affine_fp16"];
    id<MTLComputePipelineState> pipeline =
        function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!queue || !library || !pipeline) {
      std::fprintf(stderr, "Metal setup failed: %s\n",
                   error ? error.localizedDescription.UTF8String : "unknown error");
      return 4;
    }

    const long configured_page = sysconf(_SC_PAGESIZE);
    if (configured_page <= 0) return 5;
    const size_t page_bytes = static_cast<size_t>(configured_page);
    const size_t data_bytes = kElementCount * sizeof(_Float16);
    SurfaceBuffer input, output;
    if (!make_surface_buffer(device, data_bytes, page_bytes, &input) ||
        !make_surface_buffer(device, data_bytes, page_bytes, &output)) {
      release_surface_buffer(&output);
      release_surface_buffer(&input);
      return 6;
    }

    float maximum_error = 0.0f;
    bool passed = true;
    for (int round = 0; round < rounds && passed; ++round) {
      if (!write_input(input.surface, kElementCount, round)) {
        passed = false;
        break;
      }
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!command || !encoder) {
        passed = false;
        break;
      }
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input.metal offset:0 atIndex:0];
      [encoder setBuffer:output.metal offset:0 atIndex:1];
      const uint32_t count = static_cast<uint32_t>(kElementCount);
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(kElementCount, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "Metal command failed: %s\n",
                     command.error ? command.error.localizedDescription.UTF8String
                                   : "unknown error");
        passed = false;
        break;
      }
      float round_error = 0.0f;
      passed = verify_output(input.surface, output.surface, kElementCount,
                             &round_error);
      maximum_error = std::fmax(maximum_error, round_error);
    }

    std::printf("Metal IOSurface no-copy: device=%s page=%zu alloc=%zu rounds=%d\n",
                device.name.UTF8String, page_bytes, input.allocation_bytes, rounds);
    std::printf("  input base=%p metal=%p; output base=%p metal=%p\n",
                input.base, input.metal.contents, output.base, output.metal.contents);
    std::printf("  fp16 CPU->Metal->CPU max_error=%.3g %s\n", maximum_error,
                passed ? "PASS" : "FAIL");

    release_surface_buffer(&output);
    release_surface_buffer(&input);
    return passed ? 0 : 1;
  }
}
