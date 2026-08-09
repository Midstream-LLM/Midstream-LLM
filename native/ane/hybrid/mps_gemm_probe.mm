// mps_gemm_probe.mm -- bounded Metal/MPS probe for the ANE trainer.
//
// This deliberately has no dependency on train.m.  It answers two narrow
// questions before a hybrid integration is attempted:
//   1. Can MPSMatrixMultiplication consume the trainer's row-major shapes?
//   2. Can a shared host allocation be wrapped without an extra upload copy?
//
// The probe uses float32 so its result can be compared directly with the
// Accelerate cblas_sgemm path used by native/ane/training_dynamic/train.m.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MPSMatrix/MPSMatrix.h>
#import <MPSMatrix/MPSMatrixMultiplication.h>
#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 704;
constexpr int kHidden = 1856;
constexpr int kVocab = 32768;

struct Options {
  int seq = 256;
  int iters = 10;
  int warmup = 2;
  bool run_dw = true;
  bool run_classifier = true;
  bool run_classifier_token = false;
  bool cpu_reference = true;
};

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  id<MTLBuffer> buffer = nil;
};

static void usage(const char *argv0) {
  std::fprintf(stderr,
      "usage: %s [--seq 256|2048] [--op all|dw|classifier|classifier-token] "
      "[--iters N] [--warmup N] [--no-cpu]\n", argv0);
}

static bool parse_options(int argc, char **argv, Options *out) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (std::strcmp(arg, "--seq") == 0 && i + 1 < argc) {
      out->seq = std::atoi(argv[++i]);
    } else if (std::strcmp(arg, "--iters") == 0 && i + 1 < argc) {
      out->iters = std::atoi(argv[++i]);
    } else if (std::strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
      out->warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(arg, "--op") == 0 && i + 1 < argc) {
      const char *op = argv[++i];
      if (std::strcmp(op, "all") == 0) {
        out->run_dw = out->run_classifier = out->run_classifier_token = true;
      } else if (std::strcmp(op, "dw") == 0) {
        out->run_dw = true; out->run_classifier = out->run_classifier_token = false;
      } else if (std::strcmp(op, "classifier") == 0) {
        out->run_dw = out->run_classifier_token = false; out->run_classifier = true;
      } else if (std::strcmp(op, "classifier-token") == 0) {
        out->run_dw = out->run_classifier = false; out->run_classifier_token = true;
      } else {
        usage(argv[0]); return false;
      }
    } else if (std::strcmp(arg, "--no-cpu") == 0) {
      out->cpu_reference = false;
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      usage(argv[0]); return false;
    } else {
      usage(argv[0]); return false;
    }
  }
  if ((out->seq != 256 && out->seq != 2048) || out->iters < 1 || out->warmup < 0)
    return false;
  return true;
}

static double wall_ms() {
  using clock = std::chrono::steady_clock;
  static const auto start = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

static double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() & 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

static double mean(const std::vector<double> &values) {
  double total = 0.0;
  for (double value : values) total += value;
  return values.empty() ? 0.0 : total / values.size();
}

static void print_timings(const char *label, const std::vector<double> &values) {
  if (values.empty()) return;
  const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
  std::printf("  %-14s mean=%.3f median=%.3f min=%.3f max=%.3f ms\n",
              label, mean(values), median(values), *minimum, *maximum);
}

// The no-copy Metal API requires the caller to keep the allocation alive until
// the MTLBuffer is released.  A page-aligned allocation is also friendlier to
// the unified-memory implementation than a small malloc block.
static HostAllocation make_shared_buffer(id<MTLDevice> device, size_t bytes) {
  HostAllocation a;
  constexpr size_t page = 16384;
  const size_t padded = (bytes + page - 1) & ~(page - 1);
  a.bytes = padded;
  if (posix_memalign(&a.ptr, page, padded) != 0) {
    std::fprintf(stderr, "posix_memalign failed for %zu bytes\n", bytes);
    return a;
  }
  std::memset(a.ptr, 0, padded);
  a.buffer = [device newBufferWithBytesNoCopy:a.ptr
                                      length:padded
                                     options:MTLResourceStorageModeShared
                                  deallocator:nil];
  if (!a.buffer) {
    std::fprintf(stderr, "newBufferWithBytesNoCopy returned nil (%zu bytes)\n", bytes);
    std::free(a.ptr); a.ptr = nullptr; a.bytes = 0;
    return a;
  }
  if ([a.buffer contents] != a.ptr) {
    std::fprintf(stderr, "shared-buffer pointer mismatch: host=%p metal=%p\n",
                 a.ptr, [a.buffer contents]);
    a.buffer = nil;
    std::free(a.ptr); a.ptr = nullptr; a.bytes = 0;
    return a;
  }
  return a;
}

static void release_shared_buffer(HostAllocation *a) {
  // ARC releases the Objective-C object at the end of this scope, but make the
  // ordering explicit: the host allocation must outlive every command buffer.
  a->buffer = nil;
  std::free(a->ptr);
  a->ptr = nullptr;
  a->bytes = 0;
}

static uint32_t next_random(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static void fill_random(float *ptr, size_t n, uint32_t *state, float scale) {
  for (size_t i = 0; i < n; ++i) {
    const uint32_t bits = next_random(state) >> 8;
    const float unit = static_cast<float>(bits) / static_cast<float>(0x00ffffffu);
    ptr[i] = (2.0f * unit - 1.0f) * scale;
  }
}

static bool check_status(id<MTLCommandBuffer> command_buffer, const char *label) {
  if (command_buffer.status == MTLCommandBufferStatusCompleted) return true;
  NSError *error = command_buffer.error;
  std::fprintf(stderr, "%s command failed: status=%ld error=%s\n", label,
               static_cast<long>(command_buffer.status),
               error ? error.localizedDescription.UTF8String : "(none)");
  return false;
}

struct MPSGemm {
  id<MTLCommandQueue> queue = nil;
  MPSMatrixMultiplication *kernel = nil;
  MPSMatrix *left = nil;
  MPSMatrix *right = nil;
  MPSMatrix *result = nil;
  HostAllocation left_storage;
  HostAllocation right_storage;
  HostAllocation result_storage;
  int m = 0, n = 0, k = 0;
  bool left_transposed = false;
  bool right_transposed = false;
};

static void release_gemm(MPSGemm *g) {
  // Release matrix/kernel objects before their backing buffers.
  g->result = nil; g->right = nil; g->left = nil; g->kernel = nil; g->queue = nil;
  release_shared_buffer(&g->result_storage);
  release_shared_buffer(&g->right_storage);
  release_shared_buffer(&g->left_storage);
}

static bool init_gemm(id<MTLDevice> device, MPSGemm *g, int m, int n, int k,
                      bool left_transposed, bool right_transposed) {
  g->m = m; g->n = n; g->k = k;
  g->left_transposed = left_transposed; g->right_transposed = right_transposed;
  g->queue = [device newCommandQueue];
  if (!g->queue) { std::fprintf(stderr, "newCommandQueue failed\n"); return false; }

  // op(A) is [m,k] and op(B) is [k,n].  The backing matrices retain the
  // trainer's physical layout and MPS applies either transpose at dispatch.
  const size_t left_rows = left_transposed ? k : m;
  const size_t left_cols = left_transposed ? m : k;
  const size_t left_bytes = left_rows * left_cols * sizeof(float);
  const size_t right_rows = right_transposed ? n : k;
  const size_t right_cols = right_transposed ? k : n;
  const size_t right_bytes = right_rows * right_cols * sizeof(float);
  const size_t result_bytes = static_cast<size_t>(m) * n * sizeof(float);
  g->left_storage = make_shared_buffer(device, left_bytes);
  g->right_storage = make_shared_buffer(device, right_bytes);
  g->result_storage = make_shared_buffer(device, result_bytes);
  if (!g->left_storage.buffer || !g->right_storage.buffer || !g->result_storage.buffer)
    return false;

  MPSMatrixDescriptor *left_desc = [MPSMatrixDescriptor
      matrixDescriptorWithRows:left_rows columns:left_cols
      rowBytes:left_cols * sizeof(float)
      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *right_desc = [MPSMatrixDescriptor
      matrixDescriptorWithRows:right_rows columns:right_cols
      rowBytes:right_cols * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *result_desc = [MPSMatrixDescriptor
      matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
      dataType:MPSDataTypeFloat32];
  g->left = [[MPSMatrix alloc] initWithBuffer:g->left_storage.buffer descriptor:left_desc];
  g->right = [[MPSMatrix alloc] initWithBuffer:g->right_storage.buffer descriptor:right_desc];
  g->result = [[MPSMatrix alloc] initWithBuffer:g->result_storage.buffer descriptor:result_desc];
  g->kernel = [[MPSMatrixMultiplication alloc]
      initWithDevice:device transposeLeft:left_transposed transposeRight:right_transposed
      resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
  if (!g->left || !g->right || !g->result || !g->kernel) {
    std::fprintf(stderr, "MPS matrix/kernel initialization failed\n");
    return false;
  }
  return true;
}

static bool encode_once(MPSGemm *g, const char *label, double *gpu_ms = nullptr) {
  id<MTLCommandBuffer> command_buffer = [g->queue commandBuffer];
  if (!command_buffer) return false;
  [g->kernel encodeToCommandBuffer:command_buffer
                        leftMatrix:g->left rightMatrix:g->right
                       resultMatrix:g->result];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (!check_status(command_buffer, label)) return false;
  if (gpu_ms) {
    const CFTimeInterval begin = command_buffer.GPUStartTime;
    const CFTimeInterval end = command_buffer.GPUEndTime;
    *gpu_ms = (begin > 0.0 && end >= begin) ? (end - begin) * 1000.0 : 0.0;
  }
  return true;
}

static float max_abs_error(const float *a, const float *b, size_t n,
                           double *mean_abs) {
  double total = 0.0;
  float maximum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float error = std::fabs(a[i] - b[i]);
    maximum = std::max(maximum, error);
    total += error;
  }
  *mean_abs = n ? total / static_cast<double>(n) : 0.0;
  return maximum;
}

static double sum_abs(const float *values, size_t n) {
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) total += std::fabs(values[i]);
  return total;
}

static bool run_case(id<MTLDevice> device, const char *label, int m, int n, int k,
                     bool left_transposed, bool right_transposed, const Options &options,
                     float input_scale = 0.02f) {
  std::printf("\n[%s] M=%d N=%d K=%d left_transpose=%s right_transpose=%s\n",
              label, m, n, k, left_transposed ? "yes" : "no",
              right_transposed ? "yes" : "no");
  std::printf("  storage: left=%.1f MiB right=%.1f MiB result=%.1f MiB\n",
              static_cast<double>(static_cast<size_t>(left_transposed ? k : m) *
                                  (left_transposed ? m : k) * 4) / (1024.0 * 1024.0),
              static_cast<double>(static_cast<size_t>(right_transposed ? n : k) *
                                  (right_transposed ? k : n) * 4) / (1024.0 * 1024.0),
              static_cast<double>(static_cast<size_t>(m) * n * 4) / (1024.0 * 1024.0));

  MPSGemm gemm;
  if (!init_gemm(device, &gemm, m, n, k, left_transposed, right_transposed)) {
    release_gemm(&gemm);
    return false;
  }
  uint32_t seed = 0x13579bdfu ^ static_cast<uint32_t>(m + 31 * n + 131 * k);
  const size_t left_count = static_cast<size_t>(left_transposed ? k : m) *
                            (left_transposed ? m : k);
  fill_random(static_cast<float *>(gemm.left_storage.ptr), left_count, &seed, input_scale);
  const size_t right_count = static_cast<size_t>(right_transposed ? n : k) *
                             (right_transposed ? k : n);
  fill_random(static_cast<float *>(gemm.right_storage.ptr), right_count, &seed, input_scale);
  std::memset(gemm.result_storage.ptr, 0, gemm.result_storage.bytes);

  // Verify the no-copy contract before dispatch.  This is the key interop
  // property needed to wrap an IOSurface/host staging allocation later.
  std::printf("  shared_ptr: host=%p metal=%p %s\n", gemm.result_storage.ptr,
              [gemm.result_storage.buffer contents],
              [gemm.result_storage.buffer contents] == gemm.result_storage.ptr ? "OK" : "MISMATCH");

  float *cpu_result = nullptr;
  double cpu_ms = 0.0;
  std::vector<double> cpu_samples;
  if (options.cpu_reference) {
    cpu_result = static_cast<float *>(std::calloc(static_cast<size_t>(m) * n, sizeof(float)));
    if (!cpu_result) {
      std::fprintf(stderr, "CPU result allocation failed\n");
      release_gemm(&gemm); return false;
    }
    auto run_cpu = [&]() {
      cblas_sgemm(CblasRowMajor, left_transposed ? CblasTrans : CblasNoTrans,
                  right_transposed ? CblasTrans : CblasNoTrans,
                  m, n, k, 1.0f,
                  static_cast<const float *>(gemm.left_storage.ptr),
                  left_transposed ? m : k,
                  static_cast<const float *>(gemm.right_storage.ptr),
                  right_transposed ? k : n,
                  0.0f, cpu_result, n);
    };
    for (int i = 0; i < options.warmup; ++i) run_cpu();
    for (int i = 0; i < options.iters; ++i) {
      const double start = wall_ms();
      run_cpu();
      cpu_samples.push_back(wall_ms() - start);
    }
    cpu_ms = median(cpu_samples);
    print_timings("CPU cblas", cpu_samples);
  }

  for (int i = 0; i < options.warmup; ++i) {
    if (!encode_once(&gemm, label)) {
      std::free(cpu_result); release_gemm(&gemm); return false;
    }
  }
  std::vector<double> mps_wall_samples;
  std::vector<double> mps_gpu_samples;
  for (int i = 0; i < options.iters; ++i) {
    double gpu_ms = 0.0;
    const double start = wall_ms();
    if (!encode_once(&gemm, label, &gpu_ms)) {
      std::free(cpu_result); release_gemm(&gemm); return false;
    }
    mps_wall_samples.push_back(wall_ms() - start);
    if (gpu_ms > 0.0) mps_gpu_samples.push_back(gpu_ms);
  }
  const double mps_ms = median(mps_wall_samples);
  print_timings("MPS wall", mps_wall_samples);
  print_timings("MPS GPU", mps_gpu_samples);
  std::printf("  samples: %d measured, %d warmup\n", options.iters, options.warmup);
  if (cpu_ms > 0.0) std::printf("  speed ratio (CPU/MPS): %.2fx\n", cpu_ms / mps_ms);

  bool ok = true;
  if (cpu_result) {
    double mean_error = 0.0;
    const float maximum = max_abs_error(cpu_result,
        static_cast<const float *>(gemm.result_storage.ptr), static_cast<size_t>(m) * n,
        &mean_error);
    // MPS float32 should be close to the Accelerate result.  The tolerance is
    // intentionally loose enough for different reduction orders, while still
    // catching transposition/layout mistakes.
    ok = maximum <= 2.0e-3f || mean_error <= 2.0e-4;
    std::printf("  result_abs_sum: cpu=%.6g mps=%.6g\n",
                sum_abs(cpu_result, static_cast<size_t>(m) * n),
                sum_abs(static_cast<const float *>(gemm.result_storage.ptr),
                        static_cast<size_t>(m) * n));
    std::printf("  correctness: max_abs=%.6g mean_abs=%.6g %s\n", maximum, mean_error,
                ok ? "PASS" : "FAIL");
  }
  std::free(cpu_result);
  release_gemm(&gemm);
  return ok;
}

}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    Options options;
    if (!parse_options(argc, argv, &options)) {
      usage(argv[0]);
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "No Metal device is available (run outside a headless/sandboxed process).\n");
      return 3;
    }
    std::printf("MPS GEMM probe: device=%s seq=%d iters=%d warmup=%d\n",
                device.name.UTF8String, options.seq, options.iters, options.warmup);
    std::printf("  recommended rowBytes: dim=%zu hidden=%zu vocab=%zu\n",
                [MPSMatrixDescriptor rowBytesForColumns:kDim dataType:MPSDataTypeFloat32],
                [MPSMatrixDescriptor rowBytesForColumns:kHidden dataType:MPSDataTypeFloat32],
                [MPSMatrixDescriptor rowBytesForColumns:kVocab dataType:MPSDataTypeFloat32]);

    bool ok = true;
    if (options.run_dw)
      ok &= run_case(device, "ffn_dW2", kDim, kHidden, options.seq,
                     false, true, options);
    if (options.run_classifier)
      ok &= run_case(device, "classifier_fwd_channel_major", kVocab, options.seq, kDim,
                     false, false, options);
    if (options.run_classifier_token)
      ok &= run_case(device, "classifier_fwd_token_major", options.seq, kVocab, kDim,
                     true, true, options);
    std::printf("\nprobe result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
}
