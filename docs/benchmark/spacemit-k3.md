# SpacemiT K3 benchmark

With the fastest configuration, FP32 median latency is 805.0 ms for TurboVLA and 5,207 ms for SmolVLA
on the eight X100 cores, about 2× the Raspberry Pi 5 for both. The opt-in BF16 path
(`TCPU_BF16_DEQ=0`) brings SmolVLA to 4,195 ms. Only these two models were run. In closed-loop
LIBERO, a SmolVLA checkpoint on the K3 succeeds as often as vla.cpp on a GPU (16/20 FP32 and 18/20
BF16, against 18/20); see [Task success](#task-success-libero).

## Test setup

| Item | Value |
| --- | --- |
| Device | SpacemiT K3 Pico ITX, 16 GB |
| CPU | SpacemiT K3: 8 × X100 (RVV 1.0, VLEN 256, 2.4 GHz; two clusters of 4 sharing 4 MB L2) used; the 8 × A100 cores (CPUs 8-15, 2.0 GHz) were not used, see Notes |
| SIMD used | RVV 1.0 (`zvl256b`), Zvfh; Zvfbfwma for the opt-in BF16 row only |
| Memory | 16 GB (15.6 GiB visible to Linux); type not reported by the OS |
| OS | openKylin 3.0, kernel 6.18.3-generic-pcie-perst-test |
| Compiler | Cross-compiled on x86-64 with the SpacemiT glibc toolchain v1.2.4, `riscv64-unknown-linux-gnu-g++` 15.2.0; `-march=rv64gcv_zba_zbb_zbs_zfh_zvfh_zvfbfwma_zvl256b` |
| Python, NumPy | 3.12.2, 2.3.5 |
| Power | Mains power, `ondemand` governor; cooling as installed, fan state not recorded |
| Commit | `46a62933fb1c51962914560afdf8cc3962557467` (`rvv-backend`) |
| Date | 2026-11-16 00:43 to 01:14 UTC-8 (sweep), 01:26 to 01:28 (BF16 row) |

## Method

- Harness: `vla-simd-serve --model <model> --model-dir <gguf> --bench <N> --json`, one fresh process
  per configuration, driven by `tools/bench_sweep.py`.
- Warmup: 5 untimed predictions after loading, then 50 or 20 timed predictions for the reported
  configuration (20 when a query takes more than 3 s). Search runs use 10 timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction, seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing, every encoder,
  all solver steps, and action unnormalization. Network and gRPC time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's share is peak RSS minus
  the RSS of the same interpreter after importing the server but before loading a checkpoint (25 MiB
  on this device).
- VRAM: not applicable. vla.simd runs on the CPU only.
- Tuning: for every model, the thread count is swept over 8, 6, 4, 2 and 1, highest first. At the
  fastest count, each runtime setting that leaves results unchanged is tried in turn and kept only
  when it is at least 2% faster twice, the second time against a fresh baseline. The results table
  reports the final measurement of the fastest configuration.

## Model configurations

| Model | Checkpoint | Weights | Cameras and frame size | Network input | Action chunk | State | Language | Action decoding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SmolVLA | `smolvla-so101-multi-task.gguf` | 1008 MB, 75% BF16 / 25% FP32 (402.7M values) | 2 × any size | 512×512, padded | 50 × 6 | 6 | SmolLM2, every query | flow matching, 10 steps |
| TurboVLA | `turbovla-libero-f32.gguf` | 862 MB, FP32 (215.5M values) | 2 × 256×256 | 256×256 | 12 × 7 | 8 | BERT-base, every query | action head, single pass |

Checkpoints: `khanhnd61/smolvla-so101-multi-task-gguf` and `vrfai/turbovla-libero-gguf` on the
Hugging Face Hub, the same files as in the other reports.

## Engine configuration

| Item | Value |
| --- | --- |
| Backend | `rvv` |
| GEMM register tile (mr × nr) | 14×16 (RVV packed micro-kernel, LMUL 2, `VLA_RVV_MR=14`) |
| INT8 (W8A8) path | not available (no RVV int8 kernel) |
| BF16 path | opt-in, `TCPU_BF16_DEQ=0`: BF16 weight panels and `vfwmaccbf16` with BF16-rounded activations |
| Build | CMake Release (`-O3 -funroll-loops -ffp-contract=fast`), cross-compiled with a CMake toolchain file for the SpacemiT toolchain |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| SmolVLA | FP32 | 8 | none | defaults |
| SmolVLA | BF16 (opt-in) | 8 | none | `TCPU_BF16_DEQ=0`, otherwise defaults; not searched |
| TurboVLA | FP32 | 8 | none | defaults |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SmolVLA | FP32 | 5,207 | 5,177 | 5,275 | 5,302 | 9.6 | 2,247 | 2,222 |
| SmolVLA | BF16 (opt-in) | 4,195 | 4,172 | 4,212 | 4,217 | 11.9 | 1,955 | 1,930 |
| TurboVLA | FP32 | 805.0 | 792.0 | 819.2 | 820.9 | 14.9 | 1,768 | 1,743 |

TurboVLA has no INT8 path, and the RVV backend has no INT8 kernel, so there are no INT8 rows. The
BF16 row is not part of the standard protocol: it is one 20-query measurement at the FP32 winner's
configuration with `TCPU_BF16_DEQ=0` added, and it changes the numerics (see Notes).

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| SmolVLA | FP32 | 27,894 | 18,454 | 8,881 | 6,097 | **5,196** |
| TurboVLA | FP32 | 4,411 | 2,924 | 1,285 | 963.7 | **802.7** |

Runtime settings tried at 8 threads, none kept (median ms; FP32 baseline 5,196 and 802.7):

| Model | `OMP_PROC_BIND=close`, `OMP_PLACES=cores` | `TCPU_VIEW_THREADS=4` | `TCPU_EXPERT_THREADS=4` | `TCPU_OMP_MIN=2048` | `TCPU_OMP_MIN=32768` |
| --- | --- | --- | --- | --- | --- |
| SmolVLA | 5,279 | 5,508 | 6,463 | 5,212 | 5,189 |
| TurboVLA | 799.0 | n/a | n/a | 794.1 | 816.3 |

## Register-tile selection

Uses the RVV 14×16 tile (`VLA_RVV_MR=14`), measured fastest on the X100 with `tools/bench_tile`
when the backend was written; the search was not repeated for this report.

## Task success (LIBERO)

The latency rows above use the SO-101 checkpoint, which has no simulator. To check that the K3
produces usable actions, a LIBERO SmolVLA checkpoint ran closed loop in LIBERO-Object task 0 ("pick up
the alphabet soup and place it in the basket"), against vla.cpp on a GPU as the reference.

| Item | Value |
| --- | --- |
| vla.simd checkpoint | `HuggingFaceVLA/smolvla_libero`, converted with `tools/convert_hf_safetensors.py --pos-ids identity` |
| vla.cpp checkpoint | `vrfai/smolvla-libero-gguf` (`smolvla-libero.gguf`), the one in vla.cpp's `docs/benchmark/libero.md` |
| Client | vla.cpp `eval/client/run_sim_client_direct.py` at vla.cpp `71cdb09`, `--arch smolvla --task libero_object --task-id 0 --n-episodes 20 --n-action-steps 10 --seed 1` |
| Simulator | LIBERO, 500-step limit; an episode that reaches it counts as a failure |
| vla.simd serving | a ZeroMQ adapter that maps vla.cpp's request onto `SmolvlaEngine`; it rounds the client's float images to `uint8` |
| Replay | 10 of each 50-action chunk, vla.cpp's `run_libero.sh` default for SmolVLA |
| Noise | drawn fresh for every chunk, as in normal use |
| Date | 2026-11-16, 17:00 to 19:10 UTC-8 |

| Engine | Device | Successes | Inference per env step |
| --- | --- | --- | --- |
| vla.simd FP32 | K3, 8 × X100 | 16/20 (80%) | 1,133 ms |
| vla.simd BF16 (`TCPU_BF16_DEQ=0`) | K3, 8 × X100 | 18/20 (90%) | 1,108 ms |
| vla.simd FP32 | Intel Core i7-12700, 8 threads (AVX2) | 13/20 (65%) | 211 ms |
| vla.cpp (CUDA, default flags) | NVIDIA GeForce RTX 4060 Ti | 18/20 (90%) | 14 ms |

Episode *i* starts from LIBERO's *i*-th fixed initial state in every run. Failures fall on different
episodes in each run, and none fails in all four. With 20 episodes per run the gaps are not
significant: vla.cpp against the i7-12700, the largest, is p ≈ 0.13 (two-sided Fisher exact test),
and vla.simd pooled (47/60) against vla.cpp is p ≈ 0.3. vla.cpp's 18/20 on this task matches its
89.5% over all ten LIBERO-Object tasks.

The two K3 runs shared the eight X100 cores, so their per-step inference times are about twice a
single run's. "Inference per env step" is the client's average and includes the steps replayed
without a query.

Run-to-run variation is the main source of the spread. The FP32 runs on the K3 and on the i7-12700
compute the same actions (below) and still scored 16 and 13. Each chunk draws new noise, so any
change in the numerics leads to a different episode.

Open-loop agreement, on four recorded LIBERO requests with the noise pinned in the request:

| Comparison | Max abs. difference | Mean abs. difference | Cosine |
| --- | --- | --- | --- |
| K3 X100 FP32 vs i7-12700 FP32 (vla.simd) | 3.0e-6 | ≤ 3.4e-7 | 1.00000 |
| K3 X100 BF16 vs i7-12700 FP32 (vla.simd) | 0.0058 | ≤ 7e-4 | 1.00000 |
| vla.cpp CUDA vs vla.simd i7-12700 FP32 | 0.0044 | ≤ 4e-4 | 1.00000 |

Actions lie in about ±1. The vla.cpp gap is the same size as vla.simd's BF16 gap; vla.cpp keeps the
vision tower's weights in BF16 by default.

Speed is where the devices differ. One SmolVLA chunk takes about 0.1 s on the RTX 4060 Ti and 4.1 to
5.2 s on the K3's X100 cores. With 10 actions per chunk at LIBERO's 20 Hz, the K3 is about 8 to 10×
too slow to keep up with the control loop. In simulation that only lengthens each episode.

## Notes

- Only the eight X100 cores ran the benchmark. The board also brings up eight A100 cores (CPUs 8-15,
  2.0 GHz), but login sessions start with CPU affinity 0-7, which every benchmark process inherited.
- Against the Raspberry Pi 5 report, the K3 is 2.0× faster on both models at its best thread count
  (8 vs 4). Per core it is even on SmolVLA (one thread: 27,894 ms vs 27,725 ms) and 14% slower on
  TurboVLA (4,411 ms vs 3,880 ms). Two threads scale worse than on the Pi (1.5× vs 1.8× for
  SmolVLA); four and more scale well.
- The commit differs from the other reports (`7636baa`): it adds the RVV backend, including the
  K^T and score-row padding that made the 1024-token ViT attention about 3× faster on this CPU.
- INT8: `vla-simd-serve --int8` runs without error when no INT8 kernel is available and silently
  measures FP32, so the SmolVLA INT8 entry was disabled in the copy of `tools/bench_sweep.py` used
  here.
- BF16 opt-in: rounding activations to BF16 changes the outputs. On the LIBERO SmolVLA checkpoint
  with pinned noise, actions stayed within 0.006 of FP32 (cosine 1.00000), and the BF16 run succeeded
  in 18 of 20 LIBERO-Object task-0 episodes, against 16 for FP32 (see Task success).
- Never load the X100 and the A100 cores at the same time. The board reset twice under that load
  (temperature logger: load 16.8, cluster 0 at 80 to 81 °C, then nothing), including once when a single
  A100 request arrived while the X100 cores were busy.
- Temperatures during the sweep, from a 2 s logger: hottest cluster median 71 °C, peak 78 °C. The
  CPU clock was not sampled.
- Search runs used 10 timed queries instead of 20, as on the Pi, to bound the total time.
- Three idle inference servers stayed resident during the run (about 5 GB of RAM in total, no CPU
  load between requests), plus the 2 s temperature logger.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query.
