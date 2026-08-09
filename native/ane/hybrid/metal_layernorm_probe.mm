// metal_layernorm_probe.mm -- benchmark channel-major LayerNorm forward at
// the exact Jishui hidden size and sequence lengths.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kDim = 704;
constexpr float kEpsilon = 1.0e-5f;

struct SharedBuffer {
  void *ptr = nullptr;
  size_t bytes = 0;
  id<MTLBuffer> metal = nil;
};

static SharedBuffer allocate_shared(id<MTLDevice> device, size_t bytes) {
  SharedBuffer buffer;
  constexpr size_t page = 16384;
  const size_t padded = (bytes + page - 1) & ~(page - 1);
  buffer.bytes = padded;
  if (posix_memalign(&buffer.ptr, page, padded) != 0) return buffer;
  std::memset(buffer.ptr, 0, padded);
  buffer.metal = [device newBufferWithBytesNoCopy:buffer.ptr
                                           length:padded
                                          options:MTLResourceStorageModeShared
                                       deallocator:nil];
  if (!buffer.metal || buffer.metal.contents != buffer.ptr) {
    buffer.metal = nil;
    std::free(buffer.ptr);
    buffer.ptr = nullptr;
    buffer.bytes = 0;
  }
  return buffer;
}

static void release_shared(SharedBuffer *buffer) {
  buffer->metal = nil;
  std::free(buffer->ptr);
  buffer->ptr = nullptr;
}

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

static void report(const char *name, const std::vector<double> &samples) {
  const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
  double total = 0.0;
  for (double sample : samples) total += sample;
  std::printf("  %-12s mean=%8.3f median=%8.3f min=%8.3f max=%8.3f ms\n",
              name, total / samples.size(), median(samples), *minimum, *maximum);
}

static uint32_t next_random(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static void fill(float *values, size_t count, uint32_t *state, float center,
                 float radius) {
  for (size_t i = 0; i < count; ++i) {
    const float unit = static_cast<float>(next_random(state) >> 8) /
                       static_cast<float>(0x00ffffffu);
    values[i] = center + (2.0f * unit - 1.0f) * radius;
  }
}

static void layernorm_cpu(float *output, const float *input, const float *gamma,
                          const float *beta, int seq) {
  const float inverse_dim = 1.0f / kDim;
  for (int token = 0; token < seq; ++token) {
    float mean = 0.0f;
    for (int feature = 0; feature < kDim; ++feature)
      mean += input[static_cast<size_t>(feature) * seq + token];
    mean *= inverse_dim;
    float variance = 0.0f;
    for (int feature = 0; feature < kDim; ++feature) {
      const float centered = input[static_cast<size_t>(feature) * seq + token] - mean;
      variance += centered * centered;
    }
    const float inverse_stddev = 1.0f / std::sqrt(variance * inverse_dim + kEpsilon);
    for (int feature = 0; feature < kDim; ++feature) {
      const size_t index = static_cast<size_t>(feature) * seq + token;
      output[index] = (input[index] - mean) * inverse_stddev * gamma[feature] + beta[feature];
    }
  }
}

static float compare(const float *expected, const float *actual, size_t count,
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
      else return 2;
    }
    if ((seq != 256 && seq != 2048) || iterations < 1 || warmup < 0) return 2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 3;
    NSError *error = nil;
    NSString *source = @R"metal(
      #include <metal_stdlib>
      using namespace metal;
      constant uint kDim = 704;
      constant float kEpsilon = 1.0e-5f;
      kernel void layernorm_forward(
          device const float *input [[buffer(0)]],
          device const float *gamma [[buffer(1)]],
          device const float *beta [[buffer(2)]],
          device float *output [[buffer(3)]],
          constant uint &seq [[buffer(4)]],
          uint lane [[thread_index_in_threadgroup]],
          uint token [[threadgroup_position_in_grid]]) {
        threadgroup float reduction[256];
        float partial = 0.0f;
        for (uint feature = lane; feature < kDim; feature += 256)
          partial += input[feature * seq + token];
        reduction[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) reduction[lane] += reduction[lane + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float mean = reduction[0] / float(kDim);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        partial = 0.0f;
        for (uint feature = lane; feature < kDim; feature += 256) {
          const float centered = input[feature * seq + token] - mean;
          partial += centered * centered;
        }
        reduction[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) reduction[lane] += reduction[lane + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inverse_stddev = rsqrt(reduction[0] / float(kDim) + kEpsilon);
        for (uint feature = lane; feature < kDim; feature += 256) {
          const uint index = feature * seq + token;
          output[index] = (input[index] - mean) * inverse_stddev * gamma[feature] + beta[feature];
        }
      }
    )metal";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"layernorm_forward"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!library || !pipeline || !queue) {
      std::fprintf(stderr, "Metal setup failed: %s\n", error.localizedDescription.UTF8String);
      return 4;
    }

    const size_t count = static_cast<size_t>(kDim) * seq;
    SharedBuffer input = allocate_shared(device, count * sizeof(float));
    SharedBuffer gamma = allocate_shared(device, kDim * sizeof(float));
    SharedBuffer beta = allocate_shared(device, kDim * sizeof(float));
    SharedBuffer output = allocate_shared(device, count * sizeof(float));
    if (!input.metal || !gamma.metal || !beta.metal || !output.metal) return 5;
    float *cpu_output = static_cast<float *>(std::malloc(count * sizeof(float)));
    uint32_t random_state = 0x27182818u;
    fill(static_cast<float *>(input.ptr), count, &random_state, 0.0f, 3.0f);
    fill(static_cast<float *>(gamma.ptr), kDim, &random_state, 1.0f, 0.2f);
    fill(static_cast<float *>(beta.ptr), kDim, &random_state, 0.0f, 0.1f);

    auto run_cpu = [&]() {
      layernorm_cpu(cpu_output, static_cast<const float *>(input.ptr),
                    static_cast<const float *>(gamma.ptr),
                    static_cast<const float *>(beta.ptr), seq);
    };
    auto run_gpu = [&](double *device_ms) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input.metal offset:0 atIndex:0];
      [encoder setBuffer:gamma.metal offset:0 atIndex:1];
      [encoder setBuffer:beta.metal offset:0 atIndex:2];
      [encoder setBuffer:output.metal offset:0 atIndex:3];
      const uint32_t sequence_length = static_cast<uint32_t>(seq);
      [encoder setBytes:&sequence_length length:sizeof(sequence_length) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(seq, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return false;
      if (device_ms) *device_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
      return true;
    };

    for (int i = 0; i < warmup; ++i) run_cpu();
    std::vector<double> cpu_samples;
    for (int i = 0; i < iterations; ++i) {
      const double start = now_ms(); run_cpu(); cpu_samples.push_back(now_ms() - start);
    }
    for (int i = 0; i < warmup; ++i) if (!run_gpu(nullptr)) return 6;
    std::vector<double> gpu_wall_samples, gpu_device_samples;
    for (int i = 0; i < iterations; ++i) {
      double device_ms = 0.0;
      const double start = now_ms();
      if (!run_gpu(&device_ms)) return 6;
      gpu_wall_samples.push_back(now_ms() - start);
      gpu_device_samples.push_back(device_ms);
    }

    double mean_error = 0.0;
    const float max_error = compare(cpu_output, static_cast<const float *>(output.ptr),
                                    count, &mean_error);
    const bool correct = max_error <= 2.0e-4f && mean_error <= 2.0e-5;
    std::printf("Metal LayerNorm forward: device=%s seq=%d dim=%d\n",
                device.name.UTF8String, seq, kDim);
    report("CPU scalar", cpu_samples);
    report("Metal wall", gpu_wall_samples);
    report("Metal GPU", gpu_device_samples);
    std::printf("  speed ratio CPU/Metal wall: %.2fx\n",
                median(cpu_samples) / median(gpu_wall_samples));
    std::printf("  correctness: max=%.3g mean=%.3g %s\n", max_error, mean_error,
                correct ? "PASS" : "FAIL");

    std::free(cpu_output);
    release_shared(&output); release_shared(&beta); release_shared(&gamma);
    release_shared(&input);
    return correct ? 0 : 1;
  }
}
