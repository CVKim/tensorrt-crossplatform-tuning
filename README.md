# TensorRT Cross-Platform Tuning

Production tuning matrix and validated `trtexec` recipes for cross-platform
TensorRT engines: built on a **Linux** server, deployed to a **Windows AMD64**
runtime. Targets transformer-based detection models (RF-DETR, D-FINE) where
the default cross-platform FP16 path produces engines that are several times
larger than necessary and load 20–50× slower than equivalent local-native
builds.

The repo carries the full empirical matrix, an English explanation of each
contributing flag, and a per-environment production cmd that has been
validated against bbox-coordinate outputs from a local Windows-native build.

## TL;DR

If your build dispatcher routes Linux build jobs to a host whose GPU SM
matches the Windows deploy target (e.g. 4-series Windows → 4-series Linux
build host), use this:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --directIO \
  --runtimePlatform=WindowsAMD64 \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

If your pipeline ships engines to mixed-SM Windows hosts (no dispatcher
guarantee), add `--hardwareCompatibilityLevel=ampere+` back in. Both recipes
are documented and benchmarked below.

## Why this exists

The company pipeline converts ONNX → TRT on Linux servers and deploys to
Windows AMD64 runtimes. The cross-platform mode required for this flow
(`--runtimePlatform=WindowsAMD64`) interacts badly with `--fp16` on
transformer models. A baseline RF-DETR engine that compiles to **66 MiB**
natively on Windows balloons to **213 MiB** with a **26-second**
`IExecutionContext` creation pause when built cross-platform. FP32 builds of
the same model do not exhibit this; the bloat is specific to the
cross-platform + FP16 + transformer combination.

This repo:
1. Pins down the root cause with a 17-configuration FP16 matrix and a
   5-configuration FP32 matrix.
2. Identifies three `trtexec` flags that recover near-native size and load
   time without changing deploy-side code.
3. Validates the resulting engine against the original on real production
   inputs, including a six-defect bbox-coordinate comparison against a
   local-native build.
4. Provides an updated dispatcher-aware variant that closes the residual
   accuracy gap to within FP16's natural numerical noise.

## Results at a glance

Measured with TensorRT 10.8.0.43 / CUDA 12.8 on Docker-on-Windows (NGC
`tensorrt:25.01-py3`, WSL2 backend, RTX 3080 build host). Engine load
benchmarked on a Windows 11 + RTX 3080 deploy host using the TensorRT Python
API (`deserialize_cuda_engine` + `create_execution_context`).

| Model | FP16 baseline | **FP16 tuned (+3 flags)** | FP32 baseline (reference) |
|---|---:|---:|---:|
| **RF-DETR** (Flash Attention) | 213 MiB / 26.1 s | **91 MiB / 0.50 s** | 160 MiB / 0.61 s |
| **D-FINE** (transformer, no Flash) | 128 MiB / 4.69 s | **88 MiB / 0.58 s** | 155 MiB / 0.52 s |
| **YOLOv7** (CNN) | 142 MiB / 4.08 s | **140 MiB / 0.20 s** | 278 MiB / 0.24 s |

Relative reductions:
- **RF-DETR**: −57% disk, **52× faster** load
- **D-FINE**: −31% disk, **8× faster** load
- **YOLOv7**: −1.5% disk (CNNs have no FP32 fallback cubins to remove),
  **20× faster** load

For the dispatcher-aware variant (`ampere+` removed when SM is guaranteed),
expect additional disk savings from removing four of five SM cubins; load
time is unchanged.

## What each flag does

Three flags carry the entire effect. The rest of the matrix is documentation
of what *doesn't* matter.

### `--precisionConstraints=obey` — disk savings

Forces strict FP16: build fails if any layer cannot run in FP16. The implicit
default (`none`) is more conservative — TensorRT embeds FP32 fallback cubins
for FP16-risky ops (`Softmax`, `LayerNorm`, reductions) alongside the FP16
cubins. With `--hardwareCompatibilityLevel=ampere+` adding five SM variants on
top, this doubles cubin count per layer per SM. `obey` removes the fallback
track.

- RF-DETR disk: 213 → 101 MiB (−53%)
- D-FINE disk: 128 → 87 MiB (−32%)
- YOLOv7 disk: ≈ 0 (CNN ops are FP16-safe, no fallback was generated)
- Load time effect: none (separate concern)
- Flash Attention compatibility: confirmed. Strict FP16 actually *enables*
  more aggressive MHA fusion (RF-DETR layer count 264 → 241 post-fusion).

### `--versionCompatible --excludeLeanRuntime` — load-time savings

Used as a pair. Swaps the engine's runtime header from the conservative
full-runtime variant to a lean-runtime header. Cross-platform default uses
the full header to support plugin-library matching and multi-SM cubin
verification at deploy time; on transformer models with many plugin
references, walking this metadata during `IExecutionContext` creation costs
20+ seconds.

- RF-DETR load: 26.1 s → 0.50 s (52×)
- D-FINE load: 4.69 s → 0.58 s (8×)
- YOLOv7 load: 4.08 s → 0.20 s (20×)
- Disk effect: small reduction (−10 MiB on RF-DETR, lean wrapper omitted)
- Numerical effect: not free — see Validation section below

`--excludeLeanRuntime` omits the embedded lean wrapper; the deploy machine's
TRT installation provides the lean runtime at load time. Safe when build and
deploy TRT major versions match (e.g. both 10.8).

### `--hardwareCompatibilityLevel=ampere+` — keep or remove?

Embeds cubins for `sm_80 / 86 / 87 / 89 / 90`. The right answer depends on
your dispatch model:

- **No SM-aware dispatch** → keep `ampere+`. Engine portable across all
  Ampere / Ada / Hopper GPUs.
- **Dispatcher routes builds to SM-matched hosts** → remove. Engine targets
  build-host SM only; fewer cubins, narrower tactic pool (matches what a
  local-native build would pick), better accuracy match to native.

Removing `ampere+` was the single change that brought our cross-platform
output to within FP16-noise of a local Windows native build. See
[`docs/12-five-option-validation.md`](docs/12-five-option-validation.md) for
the bbox-coordinate measurement.

### Flags that don't matter

These showed zero or sub-noise effect across the matrix. Don't bother adding
them:

| Flag | Effect | Reason |
|---|---|---|
| `--directIO` | 0 MiB, 0 ms | I/O tensor format only; harmless if kept |
| `--maxAuxStreams=0` | <1 MiB | Stream-metadata savings only |
| `--profilingVerbosity=none` | <1 MiB | String-table savings only |
| `--builderOptimizationLevel=2/3` | 0 MiB | Inference-tactic search depth, not engine layout |
| `--noTF32` | 0 MiB | FP16 path has no FP32 tactics for TF32 to influence |
| `--layerOutputTypes=output:fp32` | 0 px shift | Detection output is post-NMS; FP32 guard at output does not propagate |
| `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` | ±1 MiB | Marginal; the included `+CUBLAS_LT` is the only piece worth keeping |
| `onnx-simplifier` / `polygraphy fold` | <1 MiB | Transformer constant-folding scope is small |

## Validation against local-native builds

After the size/load fix landed in production, the residual question was
whether the cross-platform engine produces the same numerical output as a
local Windows-native FP16 build. We rendered detection bboxes from five
engine variants on the same six-defect industrial input image and compared
per-bbox corner coordinates.

| Option | Build host | Cmd | Mean &#124;Δ&#124; vs native | Max &#124;Δ&#124; |
|---|---|---|---:|---:|
| #1 | Linux | `_123` + `ampere+` (prior production) | 2.33 px | **14 px** |
| #2 | Linux | `_123` + `ampere+` + `layerOutputTypes=output:fp32` | 2.33 px | **14 px** (identical to #1) |
| #3 | Linux | `_123` only (no `ampere+`) | **0.33 px** | **2 px** |
| #4 | Windows | `--fp16` only (reference) | 0 | 0 |
| #5 | Windows | `--fp16` + `--precisionConstraints=obey` | 0.33 px | 2 px |

Where `_123` = `--precisionConstraints=obey --versionCompatible
--excludeLeanRuntime`.

Findings:

1. **The 14 px shift came from `--hardwareCompatibilityLevel=ampere+`, not
   from `--versionCompatible`**. We had attributed the drift to lean-runtime
   tactic re-selection; the data shows it is actually the multi-SM cubin
   intersection narrowing the tactic pool. Removing `ampere+` recovers the
   single-SM tactic pool a native build would use.
2. **Option #3's 0.33 px / 2 px delta equals the natural variance between
   two native builds** of the same ONNX with one option toggled (#4 vs #5).
   The cross-platform penalty for Option #3 is effectively zero.
3. **`--layerOutputTypes=output:fp32` is a no-op for detection**. Options #1
   and #2 are bit-identical across all six bboxes. The output FP32 guard does
   not propagate backwards into the bbox-coordinate computation. Remove it
   from any production cmd that has it.

Raw measurements:
[`benchmarks/rfdetr_five_option_bbox_diff.csv`](benchmarks/rfdetr_five_option_bbox_diff.csv),
[`benchmarks/rfdetr_five_option_summary.csv`](benchmarks/rfdetr_five_option_summary.csv).
Reproduction script:
[`scripts/analyze_defect_bboxes.py`](scripts/analyze_defect_bboxes.py).
Full writeup: [`docs/12-five-option-validation.md`](docs/12-five-option-validation.md).

## Choosing a production cmd

### Recipe A — SM-aware dispatcher (recommended where applicable)

For pipelines where the build dispatcher guarantees the Linux build host's
GPU SM matches the Windows deploy target's GPU SM:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --directIO \
  --runtimePlatform=WindowsAMD64 \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

Trade-off: engines fail to load on GPUs with a different SM. With dispatcher
guarantees this is desirable (mis-routed engines fail fast).

Full discussion:
[`docs/13-dispatcher-aware-cmd.md`](docs/13-dispatcher-aware-cmd.md).

### Recipe B — portable engines (no dispatcher guarantee)

For pipelines that build once and ship to a mixed-SM fleet:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --directIO \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

Same disk/load gains; accepts the 1–14 px FP16-noise-level accuracy drift
across the SM portability. Full discussion:
[`docs/07-recommended-cmd.md`](docs/07-recommended-cmd.md).

### Precision-split wrapper

Don't apply the FP16 flags to FP32 builds. FP32 builds have no FP32 fallback
to strip and no IExecutionContext bloat to fix; the extra flags are no-op
at best and slightly harmful at worst.

```bash
if [[ "$PRECISION" == "fp16" ]]; then
  trtexec ... --fp16 \
    --precisionConstraints=obey \
    --versionCompatible \
    --excludeLeanRuntime
else
  trtexec ...
fi
```

### Accuracy tiers

If `--precisionConstraints=obey` causes unacceptable accuracy regression on
your model, drop it; load-time benefits from
`--versionCompatible --excludeLeanRuntime` are preserved. Tiered mitigations
(layer-specific FP32 guards, output-type pinning, etc.) are documented in
[`docs/11-accuracy-guard.md`](docs/11-accuracy-guard.md).

## Repository layout

```
.
├── README.md                                  ← this file
├── CONTRIBUTING.md                            ← branch / commit / IP-protection policy
├── benchmarks/
│   ├── README.md
│   ├── RF-DETR_fp16_build_summary.csv         ← 17-config FP16 build matrix
│   ├── RF-DETR_fp32_build_summary.csv         ← 5-config FP32 build matrix
│   ├── D-FINE_fp16_build_summary.csv
│   ├── D-FINE_fp32_build_summary.csv
│   ├── YOLOv7_fp16_build_summary.csv
│   ├── YOLOv7_fp32_build_summary.csv
│   ├── RF-DETR_variants_summary.csv           ← 4-variant cmd progression
│   ├── D-FINE_variants_summary.csv
│   ├── YOLOv7_variants_summary.csv
│   ├── load_bench_combined.csv                ← Python TRT load times
│   ├── rfdetr_five_option_bbox_diff.csv       ← per-defect bbox delta
│   └── rfdetr_five_option_summary.csv         ← aggregate bbox delta
├── docs/
│   ├── 01-problem-statement.md                ← cross-platform FP16 bloat root cause
│   ├── 02-options-explained.md                ← every flag, what it does, what it doesn't
│   ├── 03-benchmark-methodology.md            ← Docker-on-Windows + NGC TRT setup
│   ├── 04-results-rf-detr.md                  ← full RF-DETR matrix
│   ├── 05-results-d-fine.md                   ← full D-FINE matrix
│   ├── 06-fp32-comparison.md                  ← why FP32 doesn't need the fix
│   ├── 07-recommended-cmd.md                  ← Recipe B (portable, with ampere+)
│   ├── 08-results-yolov7.md                   ← full YOLOv7 matrix
│   ├── 09-final-verification.md               ← 3-model × 4-variant cross-check
│   ├── 10-model-overview.md                   ← RF-DETR / D-FINE / YOLOv7 architecture notes
│   ├── 11-accuracy-guard.md                   ← layerPrecisions / layerOutputTypes tiers
│   ├── 12-five-option-validation.md           ← 5-option bbox-coordinate validation
│   ├── 13-dispatcher-aware-cmd.md             ← Recipe A (SM-aware, ampere+ removed)
│   └── 14-timing-cache.md                     ← reproducible builds via timing cache
└── scripts/
    ├── run_in_docker.sh                       ← top-level Docker runner per model
    ├── run_matrix_generic.sh                  ← 17-config FP16 matrix
    ├── run_matrix_fp32.sh                     ← 5-config FP32 matrix
    ├── run_ampere_isolation.sh                ← ampere+ on/off, side-by-side
    ├── bench_load_python.py                   ← Windows-side load benchmark
    ├── analyze_defect_bboxes.py               ← bbox-coordinate diff across engine variants
    ├── inspect_engine_layers.py
    └── inspect_onnx.py
```

Note: existing Korean docs (`01`–`11`) are kept as-is and remain the
authoritative source for the original matrix work. New docs (`12`–`14`) and
this README are in English for cross-team consumption.
[`CONTRIBUTING.md`](CONTRIBUTING.md) holds the branch / commit policy
unchanged.

## Reproducing the matrix

The full builder matrix runs inside the NGC TensorRT container so it matches
the production Linux build server.

```bash
# 1. Pull the container (TRT 10.8.0.43)
docker pull nvcr.io/nvidia/tensorrt:25.01-py3

# 2. Run the 17-config FP16 + 5-config FP32 matrix for a model
bash scripts/run_in_docker.sh RF-DETR /absolute/path/to/model.onnx

# 3. Measure load times from the Windows side using a TRT 10.8 Python venv
python scripts/bench_load_python.py
```

Outputs land in `engines/<MODEL>/*.trt`, `logs/<MODEL>/_summary.csv`, and
`logs/_load_bench_python_all.csv`. The `engines/`, `onnx/`, `logs/`
directories are gitignored — `benchmarks/*.csv` is the committed,
sanitized subset.

Environment notes:
[`docs/03-benchmark-methodology.md`](docs/03-benchmark-methodology.md).

## Validation against the production pipeline

Builder outputs from this matrix were cross-checked against the company's
own Linux build server:

- Baseline engine: company server 211 MiB ↔ this matrix 212.88 MiB
  (0.4% match)
- Tuned engine (`_123 + ampere+`, RF-DETR): company server 89 MiB / 1.0 s
  ↔ this matrix 91 MiB / 0.50 s
- Bbox-coordinate validation (Option #3 vs local native): 0.33 px mean
  delta, 2 px max — within FP16's natural tactic-variance noise floor.

## Branch policy

- `main` — production-ready, validated. Direct commits prohibited.
- `dev` — experiments and follow-ups land here first. Merge to `main` via
  PR.

Detailed contribution policy and IP-protection rules (no ONNX, no engines,
no recipe IDs in commits): [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Environment

- **TensorRT** 10.8.0.43 (NGC container `nvcr.io/nvidia/tensorrt:25.01-py3`)
- **CUDA** 12.8
- **Build host** Docker on Windows (WSL2 backend), RTX 3080 (sm_86) or
  RTX 4080 (sm_89) per dispatcher
- **Deploy target** Windows AMD64, Ampere or Ada GPU (RTX 30xx / 40xx)
- **Driver** NVIDIA 591.86+

## License

Internal use within the company product. Refer to the company policy for
external sharing.
