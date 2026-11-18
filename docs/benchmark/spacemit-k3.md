# SpacemiT K3 benchmark

With the fastest configuration, FP32 median latency is 805.0 ms for TurboVLA and 5,207 ms for SmolVLA
on the eight X100 cores, about 2× the Raspberry Pi 5 for both. The opt-in BF16 path
(`TCPU_BF16_DEQ=0`) brings SmolVLA to 4,195 ms. Only these two models were run. The opt-in IME2
path on the eight A100 cores (`TCPU_IME=1`) brings a chunk to about 2.1 s for SmolVLA and 346 ms for
TurboVLA; see [A100 cores with IME2](#a100-cores-with-ime2-opt-in). In closed-loop LIBERO, TurboVLA
succeeds in every episode on the X100 cores, the A100 cores and vla.cpp on a GPU (40/40 each), and
SmolVLA on the K3 stays within run-to-run variation of vla.cpp and of the PyTorch reference; see
[Task success](#task-success-libero).

## Test setup

| Item | Value |
| --- | --- |
| Device | SpacemiT K3 Pico ITX, 16 GB |
| CPU | SpacemiT K3: 8 × X100 (RVV 1.0, VLEN 256, 2.4 GHz; two clusters of 4 sharing 4 MB L2) used for the sweep; the 8 × A100 cores (CPUs 8-15, 2.0 GHz, VLEN 1024, IME2) only in the opt-in IME2 section |
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

## A100 cores with IME2 (opt-in)

The K3's eight A100 cores (CPUs 8-15) have 1024-bit vectors and SpacemiT's IME2 matrix extension.
`vfwmadot` multiplies two 8×8 fp16 tiles into an 8×8 fp32 tile. Builds with `VLA_RVV_IME=ON` run
GEMMs (`TCPU_IME=1`) and attention (on with `TCPU_IME=1`; `TCPU_IME_ATTN=0` keeps it on RVV) on it.
Weights, activations, Q, K, V and the softmax weights are rounded to fp16; accumulation and the
softmax sums stay fp32. Values past fp16's range never become inf: such weights keep their
fp32/bf16 path, such activation rows are scaled by a power of two, and such attention runs on the
RVV kernel.

A process has to start on the A100 cores; a library cannot move a running process there:

```sh
TCPU_IME=1 OMP_NUM_THREADS=8 bash -c 'echo $$ > /proc/set_ai_thread; exec vla-simd-serve ...'
```

`/proc/set_ai_thread` moves the writing thread onto the A100 cores; children inherit the 1024-bit
vectors. A thread that is already running when it moves keeps 256-bit vectors. IME is used only
when every thread reports 1024-bit vectors and `vfwmadot` executes. Otherwise the engine falls back
to the X100 kernels and says so on stderr.

| Model | X100, FP32 | A100, IME GEMM only | A100, IME GEMM and attention | Max action difference vs FP32 |
| --- | --- | --- | --- | --- |
| SmolVLA (LIBERO checkpoint) | 5.2 s | 3.2 to 4.0 s | 2.1 s | 6e-4 |
| TurboVLA | 827 ms | 459 ms | 346 ms | 2.0e-3 (4.4e-4 GEMM only) |

Chunk times at 8 threads. TurboVLA: `vla-simd-serve --bench 6`, median. SmolVLA: mean per request
of the LIBERO runs below (2,080 ms over 572 requests); the X100 entry is the FP32 result above, and
the GEMM-only entry comes from separate runs on recorded requests.
In the LIBERO runs, TurboVLA averaged 376 ms on the A100 cores and 837 ms on the X100 cores.
Action differences: SmolVLA on five recorded LIBERO requests with pinned noise against vla.simd
FP32 on x86; TurboVLA on the eight benchmark queries against the X100 cores. Actions lie in about ±1.

IME2 fp16 GEMM throughput at SmolVLA's shapes is 305-745 GFLOP/s, against 140-189 for FP32 and
205-248 for BF16 on the X100 cores. Plain RVV kernels are slower on the A100 cores than on the X100
cores, so everything outside the GEMMs and attention costs more there. In the IME2 attention, a
ViT layer (1024 tokens, 12 heads) takes 13 ms, against 64 ms for the RVV kernel on the same cores.

Commits: `d99ab7b` and `4b06998` on `ime-backend`. `4b06998` adds the fp16 range handling and gives
bit-identical actions to `d99ab7b` on the recorded requests.

## ACT INT8 with the IME integer kernel

SpacemiT's IME also has an integer matrix instruction, `vmadot`: an M×M int32 tile accumulates
the product of M rows of int8 activations and M rows of int8 weights. Both clusters execute it,
with the tile set by the vector length: 4×4 over 8 values of k on the X100 cores (128
multiply-adds per instruction) and 8×8 over 16 on the A100 cores (1,024). `src/hal/rvv/gemm_i8.cpp`
is the W8A8 GEMM on it, so `--int8` now works on RISC-V; before, the RVV backend had no int8
kernel and reported INT8 as unavailable. It is on by default where `vmadot` executes
(`TCPU_IME_I8=0` turns it off) and matches the scalar int8 reference bit for bit.

| ACT (`act-so101-multi-task.gguf`), 8 threads | Median (ms) | p10 | p90 | Max / mean action difference vs FP32 |
| --- | --- | --- | --- | --- |
| X100, FP32 | 620.1 | 598.1 | 641.1 | reference |
| X100, INT8 (`--int8 63`) | 212.4 | 205.4 | 237.0 | 4.06 / 0.16 |
| A100, FP32 | 1,713 | 1,708 | 1,717 | 2.3e-5 / 2.7e-6 |
| A100, INT8 (`--int8 63`) | 308.6 | 308.1 | 309.3 | 3.97 / 0.16 |
| A100, INT8 with `TCPU_IME=1` | 220.0 | 219.5 | 220.5 | 4.08 / 0.16 |

2 warmup and 50 timed queries (10 for A100 FP32) of `policy_server --bench` on 2026-11-17. Actions
are un-normalized joint targets spanning -59 to 157 in these queries; cosine similarity to FP32 is
0.99999 for every INT8 row. The INT8 rows differ from each other because the FP32 layers between
the int8 GEMMs round differently on the two clusters, which moves a few quantized values. The
last row runs the attention scores on the IME2 fp16 kernel as well; without it the A100 cores
spend 108 ms of the 309 in FP32 attention.

The tile loop is bound by loading its operands, not by `vmadot` (about one cycle on the X100
cores, 0.7 ns on the A100 cores): in L1 a core reaches 185 G multiply-adds/s on the X100 and 480
on the A100. The X100 cores load one register at a time; the A100 cores need wide loads, as in
the fp16 kernel. The kernel reads a byte of operands per five multiply-adds, so the driver is
blocked like a BLAS GEMM (64-row groups, K in 512-byte chunks) to keep the inner loop in the 64
KB L1; unblocked, ACT took 271 ms on the X100 cores. What is left on the X100 cores is mostly
outside the int8 GEMM: FP32 attention (about 70 ms), quantization, im2col and the float epilogue.

## Task success (LIBERO)

The latency rows above use the SO-101 checkpoint, which has no simulator. To check that the K3
produces usable actions, LIBERO checkpoints of both models ran closed loop in LIBERO-Object task 0
("pick up the alphabet soup and place it in the basket") and task 1 ("pick up the cream cheese and
place it in the basket"). References: vla.cpp on a GPU, and for SmolVLA the PyTorch implementation.

| Item | Value |
| --- | --- |
| SmolVLA, vla.simd | `HuggingFaceVLA/smolvla_libero`, converted with `tools/convert_hf_safetensors.py --pos-ids identity` |
| SmolVLA, vla.cpp | `vrfai/smolvla-libero-gguf` (`smolvla-libero.gguf`), the one in vla.cpp's `docs/benchmark/libero.md` |
| TurboVLA, both engines | `vrfai/turbovla-libero-gguf` (`turbovla-libero-f32.gguf`), the file in the latency rows |
| SmolVLA, PyTorch | `HuggingFaceVLA/smolvla_libero` in LeRobot 0.6.1 (torch 2.11.0+cu126), FP32 on the GPU, TF32 off |
| Client | vla.cpp `eval/client/run_sim_client_direct.py` at vla.cpp `71cdb09`, `--task libero_object --n-episodes 20 --seed 1`; `--arch smolvla --n-action-steps 10` or `--arch turbovla --n-action-steps 12` (vla.cpp's `run_libero.sh` defaults) |
| Simulator | LIBERO, 500-step limit; an episode that reaches it counts as a failure |
| vla.simd serving | `tools/vla_cpp_server.py` (below) |
| vla.cpp serving | `vla-server`, CUDA, default flags; `--warmup 0` for TurboVLA, whose synthetic warmup input is 224×224 and fails |
| PyTorch serving | the same request mapping in a ZeroMQ server around `SmolVLAPolicy.predict_action_chunk`, using the client's float images as sent |
| Noise (SmolVLA) | drawn fresh for every chunk, as in normal use |
| Date | 2026-11-16, 17:00 to 23:46 UTC-8 |

Successes out of 20; S marks a successful episode, `.` a failure:

| Model | Task | Engine | Device | Episodes 1-20 | Successes | Inference per env step |
| --- | --- | --- | --- | --- | --- | --- |
| SmolVLA | 0 | PyTorch (LeRobot 0.6.1) | RTX 4060 Ti | `SSSSS.SSSSSS.SSS.SSS` | 17 | 61 ms |
| SmolVLA | 0 | vla.cpp | RTX 4060 Ti | `SSSSSSSSSSSSSS.SSSS.` | 18 | 14 ms |
| SmolVLA | 0 | vla.simd FP32 | K3, 8 × X100 | `SSSS.SS..SSSSSSSSSS.` | 16 | 1,133 ms |
| SmolVLA | 0 | vla.simd BF16 (`TCPU_BF16_DEQ=0`) | K3, 8 × X100 | `SSSSSSSSS.SSSSSS.SSS` | 18 | 1,108 ms |
| SmolVLA | 0 | vla.simd IME2, run 1 (`d99ab7b`) | K3, 8 × A100 | `SSSSS.S.SS..S.SS.S.S` | 13 | 246 ms |
| SmolVLA | 0 | vla.simd IME2, run 2 (`4b06998`) | K3, 8 × A100 | `SSSSS.SS.SSS.SS.S.S.` | 14 | 245 ms |
| SmolVLA | 0 | vla.simd FP32 | Intel Core i7-12700, 8 threads | `.S.SSSSSSS.SSSS...S.` | 13 | 211 ms |
| SmolVLA | 1 | vla.cpp | RTX 4060 Ti | all S | 20 | 14 ms |
| SmolVLA | 1 | vla.simd IME2 | K3, 8 × A100 | all S | 20 | 270 ms |
| TurboVLA | 0 and 1 | vla.cpp | RTX 4060 Ti | all S | 20 and 20 | 8 to 9 ms |
| TurboVLA | 0 and 1 | vla.simd FP32 | K3, 8 × X100 | all S | 20 and 20 | 86 to 89 ms |
| TurboVLA | 0 and 1 | vla.simd IME2 | K3, 8 × A100 | all S | 20 and 20 | 49 to 51 ms |

SmolVLA task 1 was not run on the X100 cores. Episode *i* starts from LIBERO's *i*-th fixed initial
state in every run. "Inference per env step" is the client's average and includes the steps replayed
without a query. The two SmolVLA FP32 and BF16 runs on the X100 cores shared the cores, so their
per-step times are about twice a single run's.

SmolVLA misses some task-0 episodes on every engine, the PyTorch reference included, and the
misses fall on different episodes in each run. No episode fails in more than four of the seven
runs, and the two IME2 runs share only one failed episode (episode 6, which PyTorch also failed).
With 20 episodes per run none of the gaps is significant (two-sided Fisher exact test): the A100
runs pooled (27/40) against the X100 runs pooled (34/40), p ≈ 0.11; against the PyTorch reference,
p ≈ 0.22; vla.cpp against the i7-12700, p ≈ 0.13. vla.cpp's 18/20 on task 0 matches its 89.5% over all ten LIBERO-Object tasks.
TurboVLA, which draws no noise, succeeded in all 120 of its episodes on the three engines.

Run-to-run variation is the main source of the spread. Each SmolVLA chunk draws new noise, so any
change in the numerics leads to a different episode. The client also restarts `request_id` at 0
in every episode, and vla.simd's server then seeds that episode's first chunk from the clock, so no
SmolVLA run repeats exactly, not even on the same engine.

Open-loop agreement on recorded LIBERO requests with the noise pinned in the request (SmolVLA):

| Comparison | Max abs. difference | Mean abs. difference | Cosine |
| --- | --- | --- | --- |
| K3 X100 FP32 vs i7-12700 FP32 (vla.simd) | 3.0e-6 | ≤ 3.4e-7 | 1.00000 |
| PyTorch (GPU) vs vla.simd i7-12700 FP32, same `uint8` images | 4.7e-6 | — | — |
| PyTorch (GPU) vs vla.simd i7-12700 FP32, client's float images | 0.0093 | ≤ 5e-4 | 1.00000 |
| K3 A100 IME2 vs i7-12700 FP32 (vla.simd) | 6e-4 | — | — |
| K3 X100 BF16 vs i7-12700 FP32 (vla.simd) | 0.0058 | ≤ 7e-4 | 1.00000 |
| vla.cpp CUDA vs vla.simd i7-12700 FP32 | 0.0044 | ≤ 4e-4 | 1.00000 |

Actions lie in about ±1. The PyTorch reference and vla.simd agree to 5e-6 when both see the same
pixels. With the client's float images the gap is 0.009, all of it from `tools/vla_cpp_server.py`
rounding them to `uint8`, the engine's input format. The vla.cpp gap is the same size as vla.simd's
BF16 gap; vla.cpp keeps the vision tower's weights in BF16 by default. The PyTorch reference must be
LeRobot 0.6.1 or later: LeRobot 0.4.3 computes actions up to 1.3 away from the same checkpoint, and
`tools/check_model_parity.py --model smolvla` fails against it. With 0.6.1 the check passes at
1.3e-5 on this checkpoint.

Speed is where the devices differ. One SmolVLA chunk takes about 0.1 s on the RTX 4060 Ti, 2.1 s on
the A100 cores and 4.1 to 5.2 s on the X100 cores. With 10 actions per chunk at LIBERO's 20 Hz
(0.5 s), the K3 is 4 to 10× too slow to keep up with a real control loop. TurboVLA's 12-action chunk
lasts 0.6 s, which the A100 cores (376 ms) keep up with and the X100 cores (837 ms) do not. In
simulation the client waits for every chunk, so speed only lengthens each episode.

### Serving vla.simd to vla.cpp's clients

`tools/vla_cpp_server.py` serves a SmolVLA or TurboVLA checkpoint over vla.cpp's ZeroMQ + protobuf
protocol, so vla.cpp's simulator clients run against vla.simd unchanged:

```sh
python tools/vla_cpp_server.py --model turbovla --model-dir turbovla-libero-f32.gguf \
    --proto <vla.cpp>/src/serving/vla.proto --bind tcp://127.0.0.1:5556
# in the vla.cpp checkout:
python eval/client/run_sim_client_direct.py --arch turbovla --vla-addr tcp://127.0.0.1:5556 \
    --task libero_object --task-id 0 --n-episodes 20 --n-action-steps 12 --seed 1
```

For TurboVLA the client sends BERT token ids, a normalized state, and expects normalized actions,
while the engine takes the instruction text and raw state. The server decodes the ids and
un-normalizes the state with the checkpoint's statistics. On every new instruction it checks both
against the engine's trace: the re-tokenized ids must equal the client's, and the engine's
normalized state must match the client's. In every run above both matched exactly. The SmolVLA runs on
vla.simd used an earlier copy of the same SmolVLA mapping on the K3, which gives bit-identical actions
to this server on recorded requests.

## Notes

- Only the eight X100 cores ran the latency sweep. The board also brings up eight A100 cores (CPUs
  8-15, 2.0 GHz), but login sessions start with CPU affinity 0-7, which every benchmark process
  inherited. The IME2 section moves a process onto them explicitly.
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
- An idle server on the A100 cores with OpenMP's default wait policy spins at about 150% CPU, so
  it counts as load for the rule above. Stop it, or start it with `OMP_WAIT_POLICY=passive`, before
  using the X100 cores.
- Temperatures during the sweep, from a 2 s logger: hottest cluster median 71 °C, peak 78 °C. The
  CPU clock was not sampled.
- Search runs used 10 timed queries instead of 20, as on the Pi, to bound the total time.
- Three idle inference servers stayed resident during the run (about 5 GB of RAM in total, no CPU
  load between requests), plus the 2 s temperature logger.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query.
