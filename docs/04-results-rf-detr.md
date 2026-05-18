# RF-DETR 매트릭스 결과

**RF-DETR**: Flash Attention 사용 transformer 기반 detection 모델.

- ONNX: 119 MiB, 2362 nodes, opset 17
- Input: `data` (1, 3, 1024, 1024) — static shape
- Output: `output` (dynamic shape, FP32)

## 전체 매트릭스 (Python 실측, RTX 3080 Windows)

| Config | Disk MiB | Deser ms | Ctx ms | **Total ms** | Build s | 사용된 옵션 |
|---|---:|---:|---:|---:|---:|---|
| 01_base_ampere_directIO ⚠️ baseline | 212.88 | 99 | 25770 | **25870** | 308 | `--fp16 --directIO --tacticSources=+CUBLAS_LT --hwcompat=ampere+` |
| 02_base_ampere | 212.77 | 99 | 25918 | 26017 | 308 | 01 에서 `--directIO` 제거 |
| 03_ampere_lean | 213.16 | 100 | 25924 | 26024 | 305 | 02 + `--maxAuxStreams=0 --profilingVerbosity=none` |
| 04_ampere_LTonly | 212.65 | 102 | 26319 | 26421 | 306 | 03 + `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` |
| 05_ampere_lean_SIM | 212.25 | 98 | 25408 | 25506 | 267 | 03 + simplified ONNX |
| 06_ampere_stripped | 53.91 | 239 | 25457 | 25696 | 1145 | 03 + `--stripWeights` (deploy refit 필요) |
| 07_ampere_optLvl3 | 213.28 | 99 | 25835 | 25934 | 305 | 03 + `--builderOptimizationLevel=3` |
| 08_ampere_optLvl2 | 213.24 | 99 | 25371 | 25470 | 298 | 03 + `--builderOptimizationLevel=2` |
| **09_ampere_verCompat ⭐** | 202.94 | 98 | 456 | **554** | 207 | 03 + `--versionCompatible --excludeLeanRuntime` |
| 10_ampere_combined_SIM | 54.08 | 243 | 26356 | 26599 | 991 | stripped + LT-only + lean + SIM |
| 11_none_runtimeWin | 183.65 | 91 | 12906 | 12997 | 124 | hwcompat=none (sm_86 고정, 비추) |
| **12_ampere_strictFp16 ⭐** | 100.57 | 48 | 27698 | 27745 | 268 | 03 + `--precisionConstraints=obey` |
| 13_ampere_noTF32_optLvl3 | 212.79 | 102 | 27063 | 27165 | 305 | optLvl3 + `--noTF32` |
| 14_ampere_lean_SIM_optLvl3 | 212.51 | 99 | 25734 | 25833 | 263 | lean + SIM + optLvl3 |
| **15_strictFp16_verCompat ⭐⭐** | **90.98** | **42** | **465** | **508** | 171 | **12 + 09 결합 (권장)** |
| 16_strictFp16_stripped | 54.19 | 213 | 26493 | 26706 | 891 | 12 + stripWeights |
| 17_kitchen_sink | 57.15 | 253 | 544 | 796 | 747 | 15 + stripWeights + SIM |
| fp32_01_base_directIO | 158.59 | 75 | 537 | 612 | 75 | 01 에서 `--fp16` 제거 |

## 핵심 비교

### 베이스 vs 권장

| | FP16 baseline (01) | **FP16 권장 (15)** | 변화 |
|---|---:|---:|---:|
| Disk | 212.88 MiB | **90.98 MiB** | **−57.3%** |
| Deser ms | 99 | 42 | **−57%** |
| Ctx ms | 25770 | 465 | **−98%** |
| Total ms | 25870 | 508 | **−98%** |
| Build s | 308 | 171 | −44% (빠른 빌드도 덤) |

### 어떤 옵션이 어디서 절감했나

| 옵션 | Disk 효과 | Loading 효과 |
|---|---|---|
| `--precisionConstraints=obey` (12번 단독) | **213 → 101 MiB (−113 MiB)** | 변화 없음 |
| `--versionCompatible --excludeLeanRuntime` (09번 단독) | 213 → 203 MiB (−10 MiB) | **25870 → 554 ms (−98%)** |
| 둘 다 결합 (15번) | 213 → 91 MiB | 25870 → 508 ms |

→ Disk 절감은 `precisionConstraints` 가 차지 (Flash Attention FP32 fallback 제거)
→ Loading 절감은 `versionCompatible + excludeLeanRuntime` 가 차지 (lean runtime header)
→ 두 효과가 서로 다른 layer 의 비용을 줄이므로 깔끔하게 더해짐

## RF-DETR 가 특히 폭증한 이유

| 원인 | 메커니즘 | 영향 |
|---|---|---|
| Flash Attention plugin lib | LayerNorm/Softmax/MHA fused kernel 들이 plugin lib 참조 | full runtime header walking 시 plugin 매칭 비용 26초 |
| Transformer layer 수 | 244 layer (post-fusion), CNN 대비 3배 | layer 당 초기화 비용 누적 |
| FP16-위험 op 다수 | softmax, LayerNorm, reduce 가 attention 마다 등장 | FP32 fallback cubin 갯수 2배 |

이 세 가지가 동시에 작용해 `ampere+` 의 5 SM × 2 precision track multiplier 와 곱해져 disk 213 MiB / 로딩 26 초가 됨.

## 산출물 위치 (회사 환경)

```
engines/RF-DETR/
├─ <RECIPE>_01_base_ampere_directIO.trt    ← 베이스 (213 MiB)
├─ <RECIPE>_15_strictFp16_verCompat.trt    ⭐ 권장 (91 MiB)
├─ <RECIPE>_fp32_01_base_directIO.trt      ← FP32 참고 (160 MiB)
└─ ... 18 engines
```

## 정확도 검증 (deploy 측에서 필요)

`--precisionConstraints=obey` 가 LayerNorm/Softmax 를 강제 FP16 으로 만들기 때문에 deploy 후 1회 mAP / detection box 검증 권장.

RF-DETR 의 경우 학습 시 FP16 호환을 가정하므로 안전한 편이지만, 정확도 측정 결과 0.5%p 이상 떨어지면 `--precisionConstraints=obey` 만 제거하면 됨 (config 09 == 203 MiB / 0.56 s).
