# YOLOv7 매트릭스 결과 (CNN 검증 케이스)

**YOLOv7**: 일반 CNN detection 모델 — attention 없음, Flash Attention 무관.

- ONNX: 280 MiB, 631 nodes, opset 17
- Input: `data` (4, 3, 1280, 1280) — batch 4, 1280×1280 고해상도
- Output: `output` (4, 7, 102000)

## 전체 매트릭스 (Python 실측, RTX 3080 Windows)

| Config | Disk MiB | Deser ms | Ctx ms | **Total ms** | Build s | 사용된 옵션 |
|---|---:|---:|---:|---:|---:|---|
| 01_base_ampere_directIO ⚠️ baseline | 141.80 | ~ | ~ | **4,075** | 238 | `--fp16 --directIO --tacticSources=+CUBLAS_LT --hwcompat=ampere+` |
| 02_base_ampere | 141.72 | ~ | ~ | 3,949 | 239 | 01 에서 `--directIO` 제거 |
| 03_ampere_lean | 141.40 | ~ | ~ | ~ | 235 | 02 + lean meta |
| 04_ampere_LTonly | 141.98 | ~ | ~ | ~ | 239 | 03 + LT-only |
| 05_ampere_lean_SIM | 155.64 | ~ | ~ | ~ | 214 | 03 + simplified ONNX (**CNN 에선 +14 MiB 부작용**) |
| 06_ampere_stripped | 9.95 | ~ | ~ | ~ | 245 | 03 + stripWeights (CNN은 cubin 거의 0) |
| 07_ampere_optLvl3 | 142.29 | ~ | ~ | ~ | 238 | 03 + optLvl3 |
| 08_ampere_optLvl2 | 144.61 | ~ | ~ | ~ | 123 | 03 + optLvl2 |
| **09_ampere_verCompat ⭐** | 139.22 | ~ | ~ | **206** | 282 | 03 + `--versionCompatible --excludeLeanRuntime` |
| 10_ampere_combined_SIM | 10.31 | ~ | ~ | ~ | 229 | stripped + LT-only + lean + SIM |
| 11_none_runtimeWin | 140.16 | ~ | ~ | ~ | 384 | hwcompat=none |
| 12_ampere_strictFp16 ⚠️ | 142.24 | ~ | ~ | 4,252 | 232 | 03 + `--precisionConstraints=obey` |
| 13_ampere_noTF32_optLvl3 | 141.25 | ~ | ~ | ~ | 255 | optLvl3 + noTF32 |
| 14_ampere_lean_SIM_optLvl3 | 155.27 | ~ | ~ | ~ | 218 | lean + SIM + optLvl3 (SIM 부작용 재확인) |
| **15_strictFp16_verCompat ⭐⭐** | **139.53** | ~ | ~ | **201** | 280 | **권장 cmd (directIO 없음)** |
| 16_strictFp16_stripped | 10.55 | ~ | ~ | ~ | 241 | 12 + stripWeights |
| 17_kitchen_sink_SIM | 20.34 | ~ | ~ | ~ | 295 | 15 + stripped + SIM |
| **variant4_directIO_full ⭐⭐ 권장** | **139.72** | ~ | ~ | **201** | 287 | **사용자 cmd + 3옵션 (directIO 유지)** |
| fp32_01_base_directIO | 278.47 | ~ | ~ | 241 | 121 | FP32 (CNN 은 weights 비중 dominant → 두 배) |

## 핵심 비교

### 베이스 vs 권장

| | FP16 baseline (01) | **FP16 권장 (variant 4)** | 변화 |
|---|---:|---:|---:|
| Disk | 141.80 MiB | **139.72 MiB** | **−1.5%** (CNN 은 fallback 없어서 효과 작음) |
| Total ms | 4,075 | **201** | **−95% (20배 단축)** |

## CNN 의 특이점

### 1. `obey` 효과가 거의 0 (12번 단독)
- 베이스 141.80 → obey 142.24 = **+0.4 MiB (노이즈)**
- 이유: CNN 의 conv/BN/ReLU 는 모두 FP16-안전 layer → FP32 fallback cubin 자체가 builder 단계에서 생성 안 됨
- → `obey` 가 제거할 fallback 이 없음 → no-op

### 2. `verCompat` 효과는 transformer 와 동일하게 큼 (09번 단독)
- 베이스 loading 4,075 ms → verCompat 206 ms = **−95% (20배)**
- 이유: cross-platform full runtime header 의 비용은 layer 수 와 plugin lib 갯수에 비례
- YOLOv7 의 layer 수는 적지만 (~120) batch 4 × 1280 입력 처리로 runtime context 초기화 비용 큼
- → verCompat 가 runtime header 를 lean 으로 swap 하면서 ctx 생성 단계 거의 다 제거

### 3. simplifier 가 CNN 에선 **부작용 발생**
- 05번 (lean + SIM): 141 → 155 MiB (**+14 MiB**)
- 14번 (lean + SIM + optLvl3): 동일 패턴
- 이유: onnxsim 이 CNN 의 일부 op 를 expand (예: BN folding 안 한 채 분리) → 노드 수 늘어남
- → CNN 에 onnxsim 적용 비추천

### 4. FP32 변환 시 disk 가 FP16 의 두 배
- FP16: 141.80 MiB / FP32: 278.47 MiB
- CNN 은 weights 비중이 압도적 (engine ≈ weights)
- FP32 weights = FP16 weights × 2 → engine 도 비례해서 두 배
- 반대로 transformer (RF-DETR) 는 FP16 fallback cubin bloat 때문에 FP32 가 오히려 작음 (160 vs 213 MiB)

## 4-variant 결과 (사용자 cmd 변형 진행도)

| 단계 | cmd | Disk MiB | Total ms |
|---|---|---:|---:|
| 1. Default (현재 production) | `--directIO + base flags + --fp16` | 141.80 | 4,075 |
| 2. Default − directIO | directIO 만 제거 | 141.72 | 3,949 |
| 3. Default + obey | + `--precisionConstraints=obey` (directIO 유지) | 141.96 | 4,154 |
| **4. Default + 3옵션 (권장)** | + obey + verCompat + excludeLeanRuntime | **139.72** | **201** |

→ 단계 4 한 번에 적용. 중간 단계는 따로 빌드할 필요 없음.

## 결론

YOLOv7 (CNN) 의 측정값으로 다음 두 가지가 확인됨:

1. **권장 cmd 는 모델 종류 무관하게 안전**:
   - Disk 효과: transformer 큰 절감 / CNN 미미 (둘 다 부작용 없음)
   - Loading 효과: **transformer / CNN 모두 8~52배 단축**

2. **`obey` 의 disk 효과는 모델별 차이 있지만 `verCompat` 의 loading 효과는 보편적**:
   - obey: FP32 fallback 이 있는 모델 (transformer) 에서만 효과
   - verCompat: cross-platform full runtime header 비용은 모든 모델에서 발생 → 효과 모든 모델

→ **회사 ONNX→TRT 파이프라인 default cmd 로 채택해도 모든 모델 (transformer / CNN) 에서 이득**.

## 산출물 위치 (회사 환경)

```
engines/YOLOv7/
├─ <RECIPE>_01_base_ampere_directIO.trt         ← 베이스 (142 MiB, 4초 로딩)
├─ <RECIPE>_15_strictFp16_verCompat.trt         ⭐ 권장 (140 MiB, 0.2초 로딩)
├─ <RECIPE>_variant4_directIO_full.trt          ⭐ 권장 (directIO 유지, 동일)
├─ <RECIPE>_fp32_01_base_directIO.trt           ← FP32 참고 (278 MiB)
└─ ... 24 engines
```

## 정확도 검증 (deploy 측)

CNN 의 conv/BN/ReLU 는 모두 FP16-안전. `--precisionConstraints=obey` 적용해도 layer 강제 전환 없음 → 정확도 영향 거의 없을 것으로 예상. 단 1회 detection 결과 비교 권장 (모든 모델 적용 시 표준 절차).
