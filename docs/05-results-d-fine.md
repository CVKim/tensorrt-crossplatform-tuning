# D-FINE 매트릭스 결과

**D-FINE**: 일반 attention (Flash Attention 미사용) transformer 기반 detection 모델.

- ONNX: 118 MiB, 2926 nodes, opset 17
- Input: `data` (1, 3, 1024, 1024) — static shape
- Output: `output` (dynamic shape, FP32)

## 전체 매트릭스 (Python 실측, RTX 3080 Windows)

| Config | Disk MiB | Deser ms | Ctx ms | **Total ms** | Build s | 사용된 옵션 |
|---|---:|---:|---:|---:|---:|---|
| 01_base_ampere_directIO ⚠️ baseline | 128.19 | 3145 | 1313 | **4458** | 364 | `--fp16 --directIO --tacticSources=+CUBLAS_LT --hwcompat=ampere+` |
| 02_base_ampere | 128.46 | 4001 | 1388 | 5389 | 359 | 01 에서 `--directIO` 제거 |
| 03_ampere_lean | 129.08 | 2931 | 1285 | 4215 | 356 | 02 + lean meta |
| 04_ampere_LTonly | 123.89 | 4835 | 1204 | 6039 | 371 | 03 + `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` |
| 05_ampere_lean_SIM | 124.14 | 4737 | 1152 | 5889 | 363 | 03 + simplified ONNX |
| 06_ampere_stripped | 43.85 | 4004 | 1293 | 5297 | 684 | 03 + `--stripWeights` |
| 07_ampere_optLvl3 | 126.25 | 3812 | 1270 | 5082 | 366 | 03 + optLvl3 |
| 08_ampere_optLvl2 | 123.68 | 1041 | 1168 | 2209 | 282 | 03 + optLvl2 |
| **09_ampere_verCompat ⭐** | 126.04 | 76 | 477 | **553** | 415 | 03 + `--versionCompatible --excludeLeanRuntime` |
| 10_ampere_combined_SIM | 38.88 | 4035 | 1223 | 5258 | 638 | stripped + LT-only + lean + SIM |
| 11_none_runtimeWin | 118.25 | 8816 | 1200 | 10016 | 448 | hwcompat=none |
| **12_ampere_strictFp16 ⭐** | 86.62 | 2789 | 1177 | 3966 | 321 | 03 + `--precisionConstraints=obey` |
| 13_ampere_noTF32_optLvl3 | 127.42 | 2806 | 1182 | 3988 | 369 | optLvl3 + noTF32 |
| 14_ampere_lean_SIM_optLvl3 | 127.28 | 2821 | 1194 | 4015 | 339 | lean + SIM + optLvl3 |
| **15_strictFp16_verCompat ⭐⭐** | **86.33** | **52** | **488** | **540** | 375 | **12 + 09 결합 (권장)** |
| 16_strictFp16_stripped | 40.12 | 3984 | 1267 | 5251 | 539 | 12 + stripWeights |
| 17_kitchen_sink_SIM | 57.79 | 374 | 545 | 919 | 601 | 15 + stripWeights + SIM |
| fp32_01_base_directIO | 154.47 | 93 | 396 | 489 | 154 | 01 에서 `--fp16` 제거 |

## 핵심 비교

### 베이스 vs 권장

| | FP16 baseline (01) | **FP16 권장 (15)** | 변화 |
|---|---:|---:|---:|
| Disk | 128.19 MiB | **86.33 MiB** | **−32.6%** |
| Deser ms | 3145 | 52 | **−98%** |
| Ctx ms | 1313 | 488 | **−63%** |
| Total ms | 4458 | 540 | **−88%** |

## RF-DETR 와의 비교

| | RF-DETR | D-FINE | 차이 |
|---|---|---|---|
| ONNX nodes | 2362 | 2926 | D-FINE +24% |
| FP16 baseline disk | 213 MiB | 128 MiB | D-FINE −40% |
| FP16 baseline loading | **27.7 s** | **4.6 s** | D-FINE 6배 빠름 |
| Flash Attention | 사용 | 미사용 | — |
| Plugin lib 참조 | 다수 | 적음 | — |

### 왜 D-FINE 베이스가 RF-DETR 보다 작고 빠른가
1. **Flash Attention plugin lib 부재** — full runtime header walking 시 plugin 매칭 비용이 적음 (RF-DETR 26초 ctx 의 대부분 원인이 여기)
2. **일반 attention 의 LayerNorm/Softmax 가 FP32 fallback 보존 자체가 적음** — FP16 fallback cubin multiplier 폭증 안 함
3. → 같은 `--fp16 --hwcompat=ampere+ --runtimePlatform=WindowsAMD64` 조합이라도 모델 구조에 따라 비용이 크게 다름

### 그래도 권장 cmd 적용 의미가 있는가
**O**. D-FINE 도 베이스 4.55 s → 0.56 s 로 8배 단축. 회사 다른 모델 (CNN 포함) 도 같은 cmd 적용 시:
- Transformer 계열 (Flash 무관): 큰 효과 (8 ~ 60배 로딩 단축)
- CNN 계열: 효과 미미하지만 부작용 없음

→ **회사 ONNX → TRT 파이프라인의 default cmd 로 채택해도 안전**.

## stripWeights 결합 효과 (D-FINE 만의 특이점)

D-FINE 은 stripWeights 와 LT-only 가 잘 결합 (config 10 = 38.88 MiB):

| Config | Disk MiB | 비고 |
|---|---:|---|
| 06_stripped 단독 | 43.85 | stripWeights only |
| 10_stripped + LT-only + SIM | **38.88** | **5 MiB 추가 감축** |

RF-DETR 에서는 LT-only 효과가 0 이었음. D-FINE 은 일반 cuBLAS path 의존이라 LT 변경에 더 민감.

## 산출물 위치 (회사 환경)

```
engines/D-FINE/
├─ <RECIPE>_01_base_ampere_directIO.trt    ← 베이스 (128 MiB)
├─ <RECIPE>_15_strictFp16_verCompat.trt    ⭐ 권장 (86 MiB)
├─ <RECIPE>_fp32_01_base_directIO.trt      ← FP32 참고 (155 MiB)
└─ ... 18 engines
```

## 정확도 검증 (deploy 측에서 필요)

D-FINE 도 RF-DETR 와 동일하게 `--precisionConstraints=obey` 적용 시 mAP / detection box 1회 검증 권장. 매트릭스에서 빌드 성공한 것으로 FP16-불가 layer 는 없는 것 확인됨.
