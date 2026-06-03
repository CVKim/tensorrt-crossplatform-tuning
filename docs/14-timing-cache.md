# Timing Cache for Reproducible Builds

## What it is

TensorRT's builder measures each candidate kernel ("tactic") on the build host
GPU and picks the fastest one per layer. These measurements are stored in a
**timing cache** file. With a cache in hand, subsequent builds skip the
measurement step and pick the cached tactic directly.

```
Layer "/encoder/attn/QKV/MatMul":
  tactic_A: 0.12 ms   ← fastest, selected
  tactic_B: 0.18 ms
  tactic_C: 0.15 ms

Layer "/encoder/norm1/LayerNormalization":
  tactic_X: 0.05 ms   ← selected
  tactic_Y: 0.08 ms
```

## Why it matters for cross-platform builds

The residual 1–2 px bbox shift that remains after applying the
[dispatcher-aware cmd](13-dispatcher-aware-cmd.md) comes from **per-build
timing noise**:

```
Build run 1: MatMul tactic_A measured 0.122 ms  → selected
Build run 2: MatMul tactic_C measured 0.119 ms  → selected (rank flipped by µs-level noise)
```

Different tactic → different FP16 accumulation order → different bbox
coordinate by 1–2 px.

A persisted timing cache locks tactic selection across rebuilds:

```
Build run 1: measure, write cache, select tactic_A
Build run 2: read cache, select tactic_A (no measurement)
Build run 3: read cache, select tactic_A
```

Result: bit-identical engine across rebuilds on the same host.

## Usage

```bash
# First build: cache is created and populated
trtexec \
  --onnx=$ONNX --saveEngine=$ENGINE --fp16 \
  --runtimePlatform=WindowsAMD64 \
  --precisionConstraints=obey --versionCompatible --excludeLeanRuntime \
  --tacticSources=+CUBLAS_LT --directIO \
  --timingCacheFile=/share/cache/RFDETR_sm89.cache

# Subsequent builds: cache is read, no re-measurement
trtexec \
  --onnx=$ONNX --saveEngine=$ENGINE --fp16 \
  --runtimePlatform=WindowsAMD64 \
  --precisionConstraints=obey --versionCompatible --excludeLeanRuntime \
  --tacticSources=+CUBLAS_LT --directIO \
  --timingCacheFile=/share/cache/RFDETR_sm89.cache
```

## Cache validity matrix

| Scenario | Cache reusable? |
|---|---|
| Same build server, same model, rebuild | Yes — full reuse |
| Same SM but different build server | Partial — tactic IDs are valid, but some measurements may be re-triggered |
| Different SM (sm_89 cache → sm_86 server) | No — TRT rejects |
| Different TRT version | No — TRT rejects |
| Same model, different precision (FP16 ↔ FP32) | Partial — only common-precision layers reused |
| Windows native cache → Linux cross-platform build | **Unverified** — see below |

## Open hypothesis: Windows-native cache → Linux cross-platform build

If a timing cache produced by a Windows-native FP16 build is fed to a Linux
cross-platform build, the cross-platform builder *may* pick the same tactics
the Windows builder picked, potentially closing the residual 0.33 px / 2 px
gap.

Caveats:
- Cross-platform mode (`--runtimePlatform=WindowsAMD64`) may exclude some
  tactic IDs from the candidate pool; missing tactics fall back to whatever
  alternative is available.
- This is **not validated** in the current matrix. Adding it would require:
  1. Build the same ONNX natively on a Windows machine (sm matching the deploy
     target) with cache export.
  2. Copy the cache to the Linux build host.
  3. Re-build cross-platform with `--timingCacheFile=` pointing to the imported
     cache.
  4. Compare bbox output against the native build.

If validated, this would be the cleanest path to achieving native-build
accuracy from a cross-platform pipeline.

## Recommended setup

Per build server, per (model, SM, TRT version):

```
/share/timing-cache/
  ├── RF-DETR_sm86_trt10.8.cache    (used by 3-series build server)
  ├── RF-DETR_sm89_trt10.8.cache    (used by 4-series build server)
  ├── D-FINE_sm86_trt10.8.cache
  └── ...
```

Build wrapper picks the correct cache file based on host SM and target model.
Cache regeneration triggers:
- ONNX changes (architecture or layer count differs)
- TRT version upgrade
- Builder cmd changes (different cmd → different tactic candidates → cache
  invalid for new build but still produces a valid engine via re-measurement)

## Limitations

- Cache locks tactic *selection*, not numerical output across hosts. Two
  different build hosts with the same SM and cache will still produce slightly
  different engines if the cache had to re-measure any unmapped tactic.
- Cache does not survive TRT major-version upgrades. Plan re-baselining as
  part of any TRT upgrade procedure.
- Sharing a cache across build hosts requires shared storage (NFS, blob
  storage, or git-LFS for small caches). Cache files are typically 1–10 MB.
