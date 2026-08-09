// metal_silu_probe.mm -- compare the trainer's CPU SiLU backward with one
// fused Metal kernel over the real Jishui hidden-state shapes.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kHidden = 1856;

struct Allocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  id<MTLBuffer> buffer = nil;
};

static double now_ms() {
  using Clock = std::chrono::steady_clock;
  static const auto origin = Clock::now();
  return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() & 1 ? values[middle]
                           : 0.5 * (values[middle - 1] + values[middle]);
}

static void report(const char *label, const std::vector<double> &samples) {
  const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
  double sum = 0.0;
  for (double sample : samples) sum += sample;
  std::printf("  %-12s mean=%8.3f median=%8.3f min=%8.3f max=%8.3f ms\n",
              label, sum / samples.size(), median(samples), *minimum, *maximum);
}

static Allocation allocate_shared(id<MTLDevice> device, size_t bytes) {
  Allocation allocation;
  constexpr size_t page = 16384;
  const size_t padded = (bytes + page - 1) & ~(page - 1);
  allocation.bytes = padded;
  if (posix_memalign(&allocation.ptr, page, padded) != 0) return allocation;
  std::memset(allocation.ptr, 0, padded);
  allocation.buffer = [device newBufferWithBytesNoCopy:allocation.ptr
                                                length:padded
                                               options:MTLResourceStorageModeShared
                                            deallocator:nil];
  if (!allocation.buffer || allocation.buffer.contents != allocation.ptr) {
    allocation.buffer = nil;
    std::free(allocation.ptr);
    allocation.ptr = nullptr;
    allocation.bytes = 0;
  }
  return allocation;
}

static void release_allocation(Allocation *allocation) {
  allocation->buffer = nil;
  std::free(allocation->ptr);
  allocation->ptr = nullptr;
  allocation->bytes = 0;
}

static uint32_t next_random(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static void fill_random(float *values, size_t count, uint32_t *state) {
  for (size_t i = 0; i < count; ++i) {
    const float unit = static_cast<float>(next_random(state) >> 8) /
                       static_cast<float>(0x00ffffffu);
    values[i] = (2.0f * unit - 1.0f) * 2.0f;
  }
}

static void cpu_silu_backward(float *dh1, float *dh3, const float *h1,
                              const float *h3, const float *dsilu,
                              float *sigmoid, float *scratch, int count) {
  const float minus_one = -1.0f;
  const float one = 1.0f;
  vDSP_vsmul(h1, 1, &minus_one, sigmoid, 1, count);
  vvexpf(sigmoid, sigmoid, &count);
  vDSP_vsadd(sigmoid, 1, &one, sigmoid, 1, count);
  vvrecf(sigmoid, sigmoid, &count);

  vDSP_vmul(h1, 1, sigmoid, 1, dh3, 1, count);
  vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, count);

  vDSP_vsadd(sigmoid, 1, &minus_one, scratch, 1, count);
  vDSP_vneg(scratch, 1, scratch, 1, count);
  vDSP_vmul(h1, 1, scratch, 1, scratch, 1, count);
  vDSP_vsadd(scratch, 1, &one, scratch, 1, count);
  vDSP_vmul(sigmoid, 1, scratch, 1, scratch, 1, count);
  vDSP_vmul(dsilu, 1, h3, 1, dh1, 1, count);
  vDSP_vmul(dh1, 1, scratch, 1, dh1, 1, count);
}

static float max_error(const float *expected, const float *actual, size_t count,
                       double *mean_error) {
  float maximum = 0.0f;
  double total = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const float error = std::fabs(expected[i] - actual[i]);
    maximum = std::max(maximum, error);
    total += error;
  }
  *mean_error = total / count;
  return maximum;
}

}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    int seq = 256;
    int iterations = 20;
    int warmup = 3;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = std::atoi(argv[++i]);
      else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
      else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
      else {
        std::fprintf(stderr, "usage: %s [--seq 256|2048] [--iters N] [--warmup N]\n", argv[0]);
        return 2;
      }
    }
    if ((seq != 256 && seq != 2048) || iterations < 1 || warmup < 0) return 2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "No Metal device is available.\n");
      return 3;
    }
    NSError *error = nil;
    NSString *source = @R"metal(
      #include <metal_stdlib>
      using namespace metal;
      kernel void silu_backward(
          device const float *h1 [[buffer(0)]],
          device const float *h3 [[buffer(1)]],
          device const float *dsilu [[buffer(2)]],
          device float *dh1 [[buffer(3)]],
          device float *dh3 [[buffer(4)]],
          constant uint &count [[buffer(5)]],
          uint index [[thread_position_in_grid]]) {
        if (index >= count) return;
        const float x = h1[index];
        const float sigmoid = 1.0f / (1.0f + exp(-x));
        const float upstream = dsilu[index];
        dh3[index] = upstream * x * sigmoid;
        dh1[index] = upstream * h3[index] * sigmoid *
                     (1.0f + x * (1.0f - sigmoid));
      }
    )metal";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
      std::fprintf(stderr, "Metal compile failed: %s\n", error.localizedDescription.UTF8String);
      return 4;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"silu_backward"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!pipeline || !queue) {
      std::fprintf(stderr, "Metal pipeline failed: %s\n", error.localizedDescription.UTF8String);
      return 4;
    }

    const size_t count = static_cast<size_t>(kHidden) * seq;
    const size_t bytes = count * sizeof(float);
    Allocation h1 = allocate_shared(device, bytes);
    Allocation h3 = allocate_shared(device, bytes);
    Allocation dsilu = allocate_shared(device, bytes);
    Allocation gpu_dh1 = allocate_shared(device, bytes);
    Allocation gpu_dh3 = allocate_shared(device, bytes);
    if (!h1.buffer || !h3.buffer || !dsilu.buffer || !gpu_dh1.buffer || !gpu_dh3.buffer) {
      std::fprintf(stderr, "Unable to allocate shared buffers.\n");
      return 5;
    }
    uint32_t random_state = 0x31415926u;
    fill_random(static_cast<float *>(h1.ptr), count, &random_state);
    fill_random(static_cast<float *>(h3.ptr), count, &random_state);
    fill_random(static_cast<float *>(dsilu.ptr), count, &random_state);

    float *cpu_dh1 = static_cast<float *>(std::malloc(bytes));
    float *cpu_dh3 = static_cast<float *>(std::malloc(bytes));
    float *sigmoid = static_cast<float *>(std::malloc(bytes));
    float *scratch = static_cast<float *>(std::malloc(bytes));
    const int vector_count = static_cast<int>(count);

    auto run_cpu = [&]() {
      cpu_silu_backward(cpu_dh1, cpu_dh3, static_cast<const float *>(h1.ptr),
                        static_cast<const float *>(h3.ptr),
                        static_cast<const float *>(dsilu.ptr), sigmoid, scratch,
                        vector_count);
    };
    auto run_gpu = [&](double *gpu_ms) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:h1.buffer offset:0 atIndex:0];
      [encoder setBuffer:h3.buffer offset:0 atIndex:1];
      [encoder setBuffer:dsilu.buffer offset:0 atIndex:2];
      [encoder setBuffer:gpu_dh1.buffer offset:0 atIndex:3];
      [encoder setBuffer:gpu_dh3.buffer offset:0 atIndex:4];
      const uint32_t element_count = static_cast<uint32_t>(count);
      [encoder setBytes:&element_count length:sizeof(element_count) atIndex:5];
      const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return false;
      if (gpu_ms) *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
      return true;
    };

    for (int i = 0; i < warmup; ++i) run_cpu();
    std::vector<double> cpu_samples;
    for (int i = 0; i < iterations; ++i) {
      const double start = now_ms();
      run_cpu();
      cpu_samples.push_back(now_ms() - start);
    }
    for (int i = 0; i < warmup; ++i) {
      if (!run_gpu(nullptr)) return 6;
    }
    std::vector<double> gpu_wall_samples;
    std::vector<double> gpu_device_samples;
    for (int i = 0; i < iterations; ++i) {
      double device_ms = 0.0;
      const double start = now_ms();
      if (!run_gpu(&device_ms)) return 6;
      gpu_wall_samples.push_back(now_ms() - start);
      gpu_device_samples.push_back(device_ms);
    }

    double dh1_mean = 0.0, dh3_mean = 0.0;
    const float dh1_max = max_error(cpu_dh1, static_cast<const float *>(gpu_dh1.ptr),
                                    count, &dh1_mean);
    const float dh3_max = max_error(cpu_dh3, static_cast<const float *>(gpu_dh3.ptr),
                                    count, &dh3_mean);
    const bool correct = dh1_max <= 2.0e-5f && dh3_max <= 2.0e-5f;
    std::printf("Metal SiLU backward: device=%s seq=%d elements=%zu (%.1f MiB/buffer)\n",
                device.name.UTF8String, seq, count, bytes / (1024.0 * 1024.0));
    report("CPU vDSP", cpu_samples);
    report("Metal wall", gpu_wall_samples);
    report("Metal GPU", gpu_device_samples);
    std::printf("  speed ratio CPU/Metal wall: %.2fx\n",
                median(cpu_samples) / median(gpu_wall_samples));
    std::printf("  correctness: dh1 max=%.3g mean=%.3g; dh3 max=%.3g mean=%.3g %s\n",
                dh1_max, dh1_mean, dh3_max, dh3_mean, correct ? "PASS" : "FAIL");

    std::free(cpu_dh1); std::free(cpu_dh3); std::free(sigmoid); std::free(scratch);
    release_allocation(&gpu_dh3); release_allocation(&gpu_dh1);
    release_allocation(&dsilu); release_allocation(&h3); release_allocation(&h1);
    return correct ? 0 : 1;
  }
}
