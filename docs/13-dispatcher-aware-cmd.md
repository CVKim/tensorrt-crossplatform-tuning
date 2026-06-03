# Dispatcher-Aware Production cmd

Updated production recommendation that supersedes
[`07-recommended-cmd.md`](07-recommended-cmd.md) for environments with an
SM-aware build dispatcher.

## When this applies

Use this recipe when the build pipeline guarantees that the **Linux build
host's GPU SM matches the Windows deploy target's GPU SM**. Typical setup:

```
Build dispatcher
 ├── target = RTX 30-series  →  routes to Linux build host with RTX 3080  (sm_86)
 └── target = RTX 40-series  →  routes to Linux build host with RTX 4080  (sm_89)
```

Under this guarantee, `--hardwareCompatibilityLevel=ampere+` (which embeds
cubins for sm_80 / 86 / 87 / 89 / 90) is dead weight — only one of the five
SM cubins is ever loaded by the deploy machine, and the multi-SM tactic
intersection introduces unnecessary numerical drift versus a native build.

For environments without dispatcher-enforced SM routing, keep
[`07-recommended-cmd.md`](07-recommended-cmd.md) (with `ampere+`).

## Recommended cmd (FP16)

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

Diff vs the prior recommendation (`07-recommended-cmd.md`):

```diff
- --hardwareCompatibilityLevel=ampere+
```

Single line removed. All other flags unchanged.

## Recommended cmd (FP32)

FP32 builds do not exhibit the cross-platform bloat issue. Keep the existing
cmd; only drop `ampere+` for the same dispatcher-aware reason:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --runtimePlatform=WindowsAMD64 \
  --onnx=$ONNX \
  --saveEngine=$ENGINE
```

Do **not** add `--precisionConstraints=obey --versionCompatible
--excludeLeanRuntime` to FP32 builds — they are no-ops or slightly harmful in
the FP32 path. See [`06-fp32-comparison.md`](06-fp32-comparison.md).

## Pipeline wrapper

```bash
if [[ "$PRECISION" == "fp16" ]]; then
  trtexec \
    --tacticSources=+CUBLAS_LT \
    --directIO \
    --runtimePlatform=WindowsAMD64 \
    --precisionConstraints=obey \
    --versionCompatible \
    --excludeLeanRuntime \
    --onnx="$ONNX" --saveEngine="$ENGINE" --fp16
else
  trtexec \
    --tacticSources=+CUBLAS_LT \
    --runtimePlatform=WindowsAMD64 \
    --onnx="$ONNX" --saveEngine="$ENGINE"
fi
```

## Expected effects vs prior recommendation

Measured on a single RF-DETR test case (4-series build host & deploy target,
6-defect input image). Bbox deltas computed vs a local Windows native FP16
build (see [`12-five-option-validation.md`](12-five-option-validation.md)).

| Metric | Prior cmd (`_123` + `ampere+`) | New cmd (`_123` no `ampere+`) |
|---|---:|---:|
| Engine size (RF-DETR) | 91 MiB | smaller (4/5 SM cubins removed)* |
| Loading time (RF-DETR) | ~0.5 s | ~0.5 s (unchanged) |
| Mean bbox delta vs native | 2.33 px | **0.33 px** |
| Max bbox delta vs native | 14 px | **2 px** |
| Cross-GPU portability | sm_80/86/87/89/90 | build-host SM only |

*Exact disk delta depends on model. Multi-SM cubin removal is largest for
attention-heavy models (RF-DETR, D-FINE).

## Fallback behavior

If a build with this cmd is mis-routed to a Windows machine with a different
SM (dispatcher bug), the engine will fail to load with a clear
"hardware compatibility" error from the TRT runtime — preferable to silent
correctness degradation that would occur if the engine ran with the wrong
tactic set.

## When NOT to drop `ampere+`

Keep `--hardwareCompatibilityLevel=ampere+` if any of the following hold:

- The build pipeline does not enforce per-SM dispatch (e.g. a single Linux
  build host produces engines for a mixed-SM Windows fleet).
- Engines are shipped to external customers with unknown GPU hardware.
- An engine must be portable across Ampere / Ada / Hopper hardware in the
  field.

In those cases, accept the 1–14 px accuracy drift (within FP16 noise for
most detection tasks) in exchange for portability. See
[`11-accuracy-guard.md`](11-accuracy-guard.md) for tiered mitigation when
both portability and accuracy are required.

## Verification checklist

After deploying this cmd on the Linux build server:

- [ ] Engine builds successfully (no "no tactic available" failures from
      `--precisionConstraints=obey`).
- [ ] Engine disk size is smaller than the prior `_123 + ampere+` baseline.
- [ ] Engine loads on the target Windows machine; load time matches prior
      `_123 + ampere+` (≈ 0.5 s for RF-DETR).
- [ ] Sample inference produces bbox coordinates within 2 px of a local
      native FP16 build on the same input.
- [ ] Engine **fails to load** if manually copied to a Windows machine with a
      different SM — confirms the dispatcher-aware behavior is intentional.
