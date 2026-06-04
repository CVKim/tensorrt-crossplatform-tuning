# Five-Option Validation (Cross-platform vs Local-native)

Empirical comparison of five engine variants on a real defect-detection image
(RF-DETR, 6 ground-truth defects, 36600×3968 px input).

The goal: identify which cross-platform configuration produces results closest to
a **local Windows native** FP16 build, given a build dispatcher that already
guarantees SM matching between Linux build host and Windows deploy target.

## Configurations

All builds use TensorRT 10.8.0.43, CUDA 12.8, RTX 4080 (sm_89).

| ID | Build host | Cmd suffix |
|---|---|---|
| **Option #1** | Linux server | `--fp16 --runtimePlatform=WindowsAMD64 --hardwareCompatibilityLevel=ampere+ --precisionConstraints=obey --versionCompatible --excludeLeanRuntime --tacticSources=+CUBLAS_LT --directIO` |
| **Option #2** | Linux server | Option #1 + `--layerOutputTypes=output:fp32` |
| **Option #3** | Linux server | Option #1 with `--hardwareCompatibilityLevel=ampere+` **removed** |
| **Option #4** | Local Windows | `--fp16` only (reference / native baseline) |
| **Option #5** | Local Windows | `--fp16 --precisionConstraints=obey` |

Option #1 is the current production cross-platform recipe (`_123` in shorthand:
obey + versionCompatible + excludeLeanRuntime).

## Method

1. Build all five engines from the same ONNX (RF-DETR Flash Attention model).
2. Run the same inference on the same 6-defect input image; render predicted
   bboxes in red on the original image.
3. Detect red bbox connected components per option and pair them across
   options by maximum IoU.
4. Report mean & max per-corner pixel delta vs Option #4 (local native FP16).

Detection / pairing script: [`scripts/analyze_defect_bboxes.py`](../scripts/analyze_defect_bboxes.py).

## Results

### Bbox coordinate delta vs Option #4 (local fp16 reference)

| Option | Mean &#124;Δcoord&#124; | Max &#124;Δcoord&#124; | Defects with non-zero shift |
|---|---:|---:|---:|
| **Option #1** (`_123` + `ampere+`) | 2.33 px | **14 px** | 1 / 6 |
| **Option #2** (`_123` + `ampere+` + `outputTypes=fp32`) | 2.33 px | **14 px** | 1 / 6 |
| **Option #3** (`_123` only, no `ampere+`) | **0.33 px** | **2 px** | 1 / 6 |
| **Option #5** (local + obey) | 0.33 px | 2 px | 1 / 6 |

### Per-defect breakdown (Δx0, Δy0, Δx1, Δy1 vs Option #4)

| Defect # | Opt #1 | Opt #2 | Opt #3 | Opt #5 |
|---:|---|---|---|---|
| 0 | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) |
| 1 | (0,0,0,0) | (0,0,0,0) | (0,0,**−2**,0) | (0,0,0,0) |
| 2 | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) |
| 3 | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) |
| 4 | (0,0,**−14**,0) | (0,0,**−14**,0) | (0,0,0,0) | (0,0,0,0) |
| 5 | (0,0,0,0) | (0,0,0,0) | (0,0,0,0) | (0,0,**+2**,0) |

## Findings

### 1. `--hardwareCompatibilityLevel=ampere+` is the dominant accuracy
   perturbation, not `--versionCompatible`

Removing `ampere+` (Option #3) drops the max coordinate delta from **14 px to
2 px**. Previously we attributed the 1–2 px shift between `_123` and the
baseline to tactic re-selection from `--versionCompatible`; the data shows the
larger shift is actually from multi-SM cubin embedding narrowing the per-layer
tactic pool to the intersection across sm_80/86/87/89/90.

When the build host SM matches the target SM (dispatcher guarantee), the
single-SM tactic pool — same as a native build — is restored, and the shift
collapses to FP16's natural numerical variance.

### 2. Option #3 reaches the same accuracy floor as native build variance

The 0.33 px mean / 2 px max delta of Option #3 vs Option #4 is **identical in
magnitude** to Option #5 vs Option #4 — two native builds of the same ONNX
with only one option (`obey`) added still differ by the same amount.

Conclusion: **the cross-platform penalty for Option #3 is zero, within the
limits of FP16 tactic noise.** This is the closest a cross-platform build can
get to a local-native build short of bit-identical reproduction (which is not
achievable across cross-platform vs native paths).

### 3. `--layerOutputTypes=output:fp32` has zero effect

Option #1 and Option #2 are **bit-identical** across all 6 bboxes (mean 2.33,
max 14, identical per-defect pattern). Forcing the output tensor to FP32 does
not propagate backwards into the post-NMS coordinate computation, so the guard
is a no-op for detection workloads. Safe to remove.

## Recommendation

For environments where the build dispatcher guarantees SM matching between
Linux build host and Windows deploy target (e.g. a 4-series build server only
ever builds engines for a 4-series deploy machine), drop
`--hardwareCompatibilityLevel=ampere+`:

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

Trade-off: an engine built without `ampere+` will fail to load on a GPU with a
different SM (e.g. a 4-series engine cannot run on a 3-series machine). This is
the correct behavior when the dispatcher already enforces SM-correct routing —
mis-routed engines fail fast rather than running silently with degraded
tactics.

See [`13-dispatcher-aware-cmd.md`](13-dispatcher-aware-cmd.md) for the
full updated recommendation, including the fallback path for environments
without an SM-aware dispatcher.

## Followup tests considered

- **`--noTF32` added to Option #1 / Option #3** — predicted no effect because
  `--precisionConstraints=obey` already eliminates the FP32 paths that TF32
  would influence. Confirmed in prior measurements (`--fp16 --noTF32 obey`
  yielded identical bbox count to `--fp16 obey`).
- **Build-time timing cache shared across builds** — see
  [`14-timing-cache.md`](14-timing-cache.md) for the proposal to lock tactic
  selection across rebuilds and (potentially) across native/cross-platform
  builds.
