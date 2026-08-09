// metal_layernorm_bwd_probe.mm -- validate and time a two-kernel Metal
// LayerNorm backward for Jishui's channel-major [DIM, SEQ] activations.

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
  id<MTLBuffer> metal = nil;
};

static SharedBuffer allocate_shared(id<MTLDevice> device, size_t bytes) {
  SharedBuffer buffer;
  const size_t page = 16384;
  const size_t padded = (bytes + page - 1) & ~(page - 1);
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

static void layernorm_backward_cpu(float *dx, float *dg, float *db,
                                   const float *dy, const float *x,
                                   const float *gamma, int seq) {
  std::memset(dg, 0, kDim * sizeof(float));
  std::memset(db, 0, kDim * sizeof(float));
  const float inverse_dim = 1.0f / kDim;
  for (int token = 0; token < seq; ++token) {
    float mean = 0.0f;
    for (int feature = 0; feature < kDim; ++feature)
      mean += x[static_cast<size_t>(feature) * seq + token];
    mean *= inverse_dim;
    float variance = 0.0f;
    for (int feature = 0; feature < kDim; ++feature) {
      const float centered = x[static_cast<size_t>(feature) * seq + token] - mean;
      variance += centered * centered;
    }
    const float inverse_stddev = 1.0f / std::sqrt(variance * inverse_dim + kEpsilon);
    float sum_dyg = 0.0f;
    float sum_dygx = 0.0f;
    for (int feature = 0; feature < kDim; ++feature) {
      const size_t index = static_cast<size_t>(feature) * seq + token;
      const float xhat = (x[index] - mean) * inverse_stddev;
      const float dyg = dy[index] * gamma[feature];
      sum_dyg += dyg;
      sum_dygx += dyg * xhat;
      dg[feature] += dy[index] * xhat;
      db[feature] += dy[index];
    }
    for (int feature = 0; feature < kDim; ++feature) {
      const size_t index = static_cast<size_t>(feature) * seq + token;
      const float xhat = (x[index] - mean) * inverse_stddev;
      const float dyg = dy[index] * gamma[feature];
      dx[index] = inverse_stddev * inverse_dim *
                  (kDim * dyg - sum_dyg - xhat * sum_dygx);
    }
  }
}

static float compare(const float *expected, const float *actual, size_t count,
                     double *mean_error) {
  float maximum = 0.0f;
  double total = 0.0;
  size_t nonfinite = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
      if (nonfinite == 0)
        std::fprintf(stderr, "nonfinite at %zu: expected=%g actual=%g\n",
                     i, expected[i], actual[i]);
      ++nonfinite;
      continue;
    }
    const float error = std::fabs(expected[i] - actual[i]);
    maximum = std::max(maximum, error);
    total += error;
  }
  if (nonfinite) {
    std::fprintf(stderr, "nonfinite values: %zu/%zu\n", nonfinite, count);
    *mean_error = INFINITY;
    return INFINITY;
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

      kernel void layernorm_dx(
          device const float *dy [[buffer(0)]],
          device const float *x [[buffer(1)]],
          device const float *gamma [[buffer(2)]],
          device float *dx [[buffer(3)]],
          device float *means [[buffer(4)]],
          device float *inverse_stddevs [[buffer(5)]],
          constant uint &seq [[buffer(6)]],
          uint lane [[thread_index_in_threadgroup]],
          uint token [[threadgroup_position_in_grid]]) {
        threadgroup float first[256];
        threadgroup float second[256];
        float partial = 0.0f;
        for (uint feature = lane; feature < kDim; feature += 256)
          partial += x[feature * seq + token];
        first[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) first[lane] += first[lane + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float mean = first[0] / float(kDim);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        partial = 0.0f;
        for (uint feature = lane; feature < kDim; feature += 256) {
          const float centered = x[feature * seq + token] - mean;
          partial += centered * centered;
        }
        first[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) first[lane] += first[lane + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inverse_stddev = rsqrt(first[0] / float(kDim) + kEpsilon);
        if (lane == 0) {
          means[token] = mean;
          inverse_stddevs[token] = inverse_stddev;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sum_dyg = 0.0f;
        float sum_dygx = 0.0f;
        for (uint feature = lane; feature < kDim; feature += 256) {
          const uint index = feature * seq + token;
          const float dyg = dy[index] * gamma[feature];
          const float xhat = (x[index] - mean) * inverse_stddev;
          sum_dyg += dyg;
          sum_dygx += dyg * xhat;
        }
        first[lane] = sum_dyg;
        second[lane] = sum_dygx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) {
            first[lane] += first[lane + stride];
            second[lane] += second[lane + stride];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum_dyg = first[0];
        sum_dygx = second[0];
        for (uint feature = lane; feature < kDim; feature += 256) {
          const uint index = feature * seq + token;
          const float xhat = (x[index] - mean) * inverse_stddev;
          const float dyg = dy[index] * gamma[feature];
          dx[index] = inverse_stddev / float(kDim) *
                      (float(kDim) * dyg - sum_dyg - xhat * sum_dygx);
        }
      }

      kernel void layernorm_params(
          device const float *dy [[buffer(0)]],
          device const float *x [[buffer(1)]],
          device const float *means [[buffer(2)]],
          device const float *inverse_stddevs [[buffer(3)]],
          device float *dg [[buffer(4)]],
          device float *db [[buffer(5)]],
          constant uint &seq [[buffer(6)]],
          uint lane [[thread_index_in_threadgroup]],
          uint feature [[threadgroup_position_in_grid]]) {
        threadgroup float gamma_sum[256];
        threadgroup float beta_sum[256];
        float partial_gamma = 0.0f;
        float partial_beta = 0.0f;
        for (uint token = lane; token < seq; token += 256) {
          const uint index = feature * seq + token;
          const float gradient = dy[index];
          partial_gamma += gradient * (x[index] - means[token]) * inverse_stddevs[token];
          partial_beta += gradient;
        }
        gamma_sum[lane] = partial_gamma;
        beta_sum[lane] = partial_beta;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
          if (lane < stride) {
            gamma_sum[lane] += gamma_sum[lane + stride];
            beta_sum[lane] += beta_sum[lane + stride];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
          dg[feature] = gamma_sum[0];
          db[feature] = beta_sum[0];
        }
      }
    )metal";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> dx_function = [library newFunctionWithName:@"layernorm_dx"];
    id<MTLFunction> params_function = [library newFunctionWithName:@"layernorm_params"];
    id<MTLComputePipelineState> dx_pipeline =
        [device newComputePipelineStateWithFunction:dx_function error:&error];
    id<MTLComputePipelineState> params_pipeline =
        [device newComputePipelineStateWithFunction:params_function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!library || !dx_pipeline || !params_pipeline || !queue) {
      std::fprintf(stderr, "Metal setup failed: %s\n", error.localizedDescription.UTF8String);
      return 4;
    }

    const size_t count = static_cast<size_t>(kDim) * seq;
    SharedBuffer x = allocate_shared(device, count * sizeof(float));
    SharedBuffer dy = allocate_shared(device, count * sizeof(float));
    SharedBuffer gamma = allocate_shared(device, kDim * sizeof(float));
    SharedBuffer dx = allocate_shared(device, count * sizeof(float));
    SharedBuffer dg = allocate_shared(device, kDim * sizeof(float));
    SharedBuffer db = allocate_shared(device, kDim * sizeof(float));
    SharedBuffer means = allocate_shared(device, seq * sizeof(float));
    SharedBuffer inverse_stddevs = allocate_shared(device, seq * sizeof(float));
    if (!x.metal || !dy.metal || !gamma.metal || !dx.metal || !dg.metal ||
        !db.metal || !means.metal || !inverse_stddevs.metal) return 5;
    float *cpu_dx = static_cast<float *>(std::malloc(count * sizeof(float)));
    float *cpu_dg = static_cast<float *>(std::malloc(kDim * sizeof(float)));
    float *cpu_db = static_cast<float *>(std::malloc(kDim * sizeof(float)));
    uint32_t random_state = 0x16180339u;
    fill(static_cast<float *>(x.ptr), count, &random_state, 0.0f, 3.0f);
    fill(static_cast<float *>(dy.ptr), count, &random_state, 0.0f, 0.1f);
    fill(static_cast<float *>(gamma.ptr), kDim, &random_state, 1.0f, 0.2f);

    auto run_cpu = [&]() {
      layernorm_backward_cpu(cpu_dx, cpu_dg, cpu_db,
                             static_cast<const float *>(dy.ptr),
                             static_cast<const float *>(x.ptr),
                             static_cast<const float *>(gamma.ptr), seq);
    };
    auto run_gpu = [&](double *device_ms) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      const uint32_t sequence_length = static_cast<uint32_t>(seq);
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:dx_pipeline];
      [encoder setBuffer:dy.metal offset:0 atIndex:0];
      [encoder setBuffer:x.metal offset:0 atIndex:1];
      [encoder setBuffer:gamma.metal offset:0 atIndex:2];
      [encoder setBuffer:dx.metal offset:0 atIndex:3];
      [encoder setBuffer:means.metal offset:0 atIndex:4];
      [encoder setBuffer:inverse_stddevs.metal offset:0 atIndex:5];
      [encoder setBytes:&sequence_length length:sizeof(sequence_length) atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(seq, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:params_pipeline];
      [encoder setBuffer:dy.metal offset:0 atIndex:0];
      [encoder setBuffer:x.metal offset:0 atIndex:1];
      [encoder setBuffer:means.metal offset:0 atIndex:2];
      [encoder setBuffer:inverse_stddevs.metal offset:0 atIndex:3];
      [encoder setBuffer:dg.metal offset:0 atIndex:4];
      [encoder setBuffer:db.metal offset:0 atIndex:5];
      [encoder setBytes:&sequence_length length:sizeof(sequence_length) atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(kDim, 1, 1)
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

    double dx_mean = 0.0, dg_mean = 0.0, db_mean = 0.0;
    size_t bad_tokens = 0;
    for (int token = 0; token < seq; ++token) {
      bool bad = false;
      for (int feature = 0; feature < kDim; ++feature) {
        const size_t index = static_cast<size_t>(feature) * seq + token;
        if (!std::isfinite(static_cast<const float *>(dx.ptr)[index])) {
          bad = true;
          break;
        }
      }
      if (bad && bad_tokens++ < 8)
        std::fprintf(stderr, "bad token %d: mean=%g invstd=%g\n", token,
                     static_cast<const float *>(means.ptr)[token],
                     static_cast<const float *>(inverse_stddevs.ptr)[token]);
    }
    if (bad_tokens) std::fprintf(stderr, "bad dx tokens: %zu/%d\n", bad_tokens, seq);
    const float dx_max = compare(cpu_dx, static_cast<const float *>(dx.ptr), count, &dx_mean);
    const float dg_max = compare(cpu_dg, static_cast<const float *>(dg.ptr), kDim, &dg_mean);
    const float db_max = compare(cpu_db, static_cast<const float *>(db.ptr), kDim, &db_mean);
    const bool correct = dx_max <= 3.0e-5f && dg_max <= 3.0e-3f && db_max <= 3.0e-4f;
    std::printf("Metal LayerNorm backward: device=%s seq=%d dim=%d\n",
                device.name.UTF8String, seq, kDim);
    report("CPU scalar", cpu_samples);
    report("Metal wall", gpu_wall_samples);
    report("Metal GPU", gpu_device_samples);
    std::printf("  speed ratio CPU/Metal wall: %.2fx\n",
                median(cpu_samples) / median(gpu_wall_samples));
    std::printf("  correctness: dx %.3g/%.3g dg %.3g/%.3g db %.3g/%.3g %s\n",
                dx_max, dx_mean, dg_max, dg_mean, db_max, db_mean,
                correct ? "PASS" : "FAIL");

    std::free(cpu_dx); std::free(cpu_dg); std::free(cpu_db);
    release_shared(&inverse_stddevs); release_shared(&means);
    release_shared(&db); release_shared(&dg); release_shared(&dx);
    release_shared(&gamma); release_shared(&dy); release_shared(&x);
    return correct ? 0 : 1;
  }
}
