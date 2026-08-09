# ANE + Metal probes

This directory contains bounded benchmarks used to choose the GPU boundary for
`training_dynamic/train.m`.  They use Jishui-200M's real dimensions, page-aligned
host allocations, `MTLResourceStorageModeShared`, and CPU reference checks.

## Build and run

```bash
make
./mps_gemm_probe --seq 2048 --op all --iters 20 --warmup 3
./metal_silu_probe --seq 2048 --iters 50 --warmup 5
./metal_layernorm_probe --seq 2048 --iters 30 --warmup 4
./metal_layernorm_bwd_probe --seq 2048 --iters 100 --warmup 5
./metal_iosurface_probe --rounds 8
```

The probes allocate only process memory and small binaries.  They do not write
training checkpoints. `metal_iosurface_probe` loads only Metal and IOSurface;
it does not load or call AppleNeuralEngine.framework. It verifies that an
IOSurface's page-aligned allocation can be pre-bound with
`newBufferWithBytesNoCopy`, then checks repeated fp16 CPU -> Metal -> CPU
round trips without a staging copy.

## Measured decision

Measurements below are wall-time medians on the local Apple M4.  Concurrent
MLX activity and the fanless thermal envelope cause run-to-run variance, so the
end-to-end microstep is the deciding measurement.

| Operation | Seq | CPU | Metal/MPS | CPU / GPU |
|---|---:|---:|---:|---:|
| FFN dW2 GEMM | 256 | 0.449 ms | 0.983 ms | 0.46x |
| FFN dW2 GEMM | 2048 | 4.118 ms | 4.539 ms | 0.91x |
| classifier GEMM | 256 | 9.029 ms | 11.147 ms | 0.81x |
| classifier GEMM | 2048 | 62.628 ms | 130.501 ms | 0.48x |
| fused SiLU backward | 256 | 0.368 ms | 0.619 ms | 0.59x |
| fused SiLU backward | 2048 | 8.312 ms | 5.431 ms | 1.53x |
| LayerNorm forward | 256 | 0.791 ms | 0.453 ms | 1.75x |
| LayerNorm forward | 2048 | 11.917 ms | 1.418 ms | 8.41x |
| LayerNorm backward | 2048 | 21.980 ms | 4.734 ms | 4.64x |

`MPSMatrixMultiplication` is therefore not used in the trainer: Accelerate's
fp32 GEMM is faster at these shapes, even though the MPS results are bitwise
equal and the buffers are genuinely no-copy.  The retained GPU boundary is the
custom Metal LayerNorm and SiLU kernels.

## Integrated backend

Build the fixed-sequence trainer, then select the experimental GPU operations:

```bash
cd ../training_dynamic
make MODEL=jishui_200m_2048
./train --scratch \
  --index /private/tmp/jishui_ane_train.index \
  --data-dir /Volumes/PS2000/LLM/Jishui/dataset/processed \
  --steps 2 --accum 2 --no-checkpoint \
  --metal-norm --metal-silu
```

The resulting heterogeneous split is:

- ANE: transformer forward, attention backward, and activation-gradient GEMMs;
- Metal GPU: LayerNorm forward/backward and fused SiLU backward;
- CPU/Accelerate: classifier, softmax, weight-gradient GEMMs, AdamW, sampling,
  and orchestration.

This is operator-level heterogeneous training on one SoC, not replicated-model
data parallelism.  `--metal-shadow` enables the Metal LayerNorm path and checks
every real invocation against the CPU implementation; it aborts on a failed
tolerance gate. `--parity-report FILE` writes compact JSONL probes for comparing
loss, gradients, weights and Adam state without a full checkpoint.

The production-length `--metal-shadow` run checked every integrated LayerNorm
and SiLU invocation against the CPU reference and completed a real-index Adam
update with `accum=2`. Metal and CPU are tolerance-equivalent rather than
bitwise equal: on the first index-sampled step the losses were `10.4381294` and
`10.4381037`. This is the expected effect of a different floating-point
reduction order; the shadow gate rejects larger local errors.

After the token-major classifier and bounded dW queue changes, a cool-state
paired bounded-record run measured **5.6265 s** for ANE+CPU and **4.0101 s** for
ANE+Metal, or **1.40x** end to end. `cblas_wait=0` in the Metal run means the
serial Accelerate dW work was hidden behind ANE execution, so moving those GEMMs
to MPS has no exposed wait to remove. Absolute throughput still varies with
temperature and concurrent GPU or swap pressure. At seq=256, GPU submission
overhead generally makes the integrated path slower, so Metal is only useful
with the production sequence length.

A separate same-seed, three-update real-index A/B measured 5.555 s/step on
ANE+CPU and 4.680 s/step on ANE+Metal (15.8% faster). The third-update loss
differed by `7.4e-4` relative and the sampled weights by at most `1.41e-5`
absolute. Under live MLX plus heavy swap, a previous corrected production-shape
gate was effectively tied at 20.654 s versus 20.467 s. Treat all of these as
short correctness/performance gates, not stable overnight throughput.

Running a complete model replica on each accelerator is deliberately out of
scope for this host. It would duplicate training state in 24 GB unified memory,
add a custom gradient all-reduce boundary, and make asymmetric ANE/GPU workers
compete for the same memory and thermal budget.

The reduction kernels require a barrier after every thread has read the final
shared reduction value and before the same threadgroup array is reused.  The
shadow gate caught this race during integration; removing that barrier causes
rare local errors that amplify into invalid gradients across 30 layers.
