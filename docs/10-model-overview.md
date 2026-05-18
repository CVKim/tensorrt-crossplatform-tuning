# 모델별 특징 & cmd 적용 가이드

본 매트릭스에서 검증된 세 모델 (RF-DETR, D-FINE, YOLOv7) 의 구조적 특징과 cross-platform 변환 시 권장 cmd 의 적용 효과 정리.

---

## RF-DETR (Roboflow DETR)

### 모델 개요
- **분류**: Transformer 기반 detection (DETR variant)
- **출처**: Roboflow 가 공개한 DETR 계열 detection model
- **특징**:
  - Transformer encoder-decoder + object query 구조
  - **Flash Attention** 통합 (학습 / inference 모두)
  - Mixed-precision (FP16) 학습이 기본
  - End-to-end set prediction (NMS 불필요)

### 구조적 특징 (cross-platform FP16 변환에 미치는 영향)

| 구성요소 | 갯수 / 비중 | 영향 |
|---|---|---|
| Attention block (self + cross) | encoder × 6 + decoder × 6 | Flash Attention plugin lib 다수 참조 |
| LayerNorm | attention block 마다 2개 → 24+ | **FP32 fallback cubin 다수 embed** (`--fp16` 단독 시) |
| Softmax | attention 마다 1개 | 동상 |
| Linear / GEMM | attention QKV 투영 등 | FP16-안전 |
| Object queries (learned embeddings) | 300개 | weights 비중 작음 |

### 측정값 (RTX 3080 build / Windows runtime)

| Config | Disk MiB | Total Load ms |
|---|---:|---:|
| Default (현재 production) | 212.88 | **26,130** |
| 권장 cmd (variant 4) | **91.38** | **502** |
| 절감률 | **−57%** | **52× 단축** |

### 왜 RF-DETR 가 cross-platform FP16 에서 유독 부풀었는가
- LayerNorm/Softmax/Reduce 가 transformer block 마다 등장 → FP32 fallback cubin 갯수 폭증
- `ampere+` multi-SM (5개) × FP16/FP32 dual track = layer 당 cubin 10배
- Flash Attention plugin lib 참조 다수 → full runtime header walking 시 ABI 검증 비용 26초

### 권장 cmd 적용 시
- ✅ Disk: **−57%** (precisionConstraints=obey 가 FP32 fallback cubin 제거)
- ✅ Loading: **52× 단축** (verCompat 가 lean runtime header 로 swap)
- ⚠️ 정확도: precisionConstraints=obey 적용 시 LayerNorm/Softmax 강제 FP16 → ~0.2% bbox 시프트 가능 (참고: 회사 다른 분 측정에서 850장 중 2 bbox 차이)

→ [정확도 검증 가이드](11-accuracy-guard.md) 참조

---

## D-FINE

### 모델 개요
- **분류**: Transformer 기반 detection (DETR family, fine-grained variant)
- **특징**:
  - DETR + Fine-grained decoder refinement
  - **Flash Attention 미사용** (일반 scaled dot-product attention)
  - 일부 attention 변종 (Cross-Stage Cross-Scale) 사용
  - FP16-호환 (학습 시점)

### 구조적 특징

| 구성요소 | 영향 |
|---|---|
| 일반 attention (Q@K^T → Softmax → V@... 분리 op) | TRT 가 Flash Attention 으로 auto-fuse 시도 가능하지만 패턴이 표준과 다를 시 분리 path 유지 |
| LayerNorm 적은 갯수 (RF-DETR 대비) | FP32 fallback cubin 도 적음 → disk bloat 도 적음 |
| Plugin lib 참조 적음 | Loading 시 ABI 검증 비용도 RF-DETR 보다 적음 |

### 측정값

| Config | Disk MiB | Total Load ms |
|---|---:|---:|
| Default | 128.19 | **4,690** |
| 권장 cmd (variant 4) | **88.07** | **576** |
| 절감률 | **−31%** | **8× 단축** |

### Cross-platform FP16 부담이 RF-DETR 보다 작은 이유
- Attention plugin 참조 적음 → loading 비용 4.7초 (RF-DETR 의 1/5)
- LayerNorm/Softmax 갯수가 적어 FP32 fallback 도 적음 → disk 128 MiB (RF-DETR 의 60%)

### 권장 cmd 적용 시
- ✅ Disk: **−31%**
- ✅ Loading: **8× 단축**
- ⚠️ 정확도: RF-DETR 와 동일하게 obey 시 미세한 bbox 시프트 가능성

---

## YOLOv7 (Pure CNN)

### 모델 개요
- **분류**: CNN-based detection (YOLO family)
- **특징**:
  - 100% CNN (attention 자체가 없음)
  - Conv / BN / SiLU / ReLU / SPP / Concat / Upsample 등 표준 CNN op
  - NMS-based post-processing (engine 외부)
  - 다양한 학습 precision (FP16/FP32) 모두 호환

### 구조적 특징 (다른 두 모델과 결정적 차이)

| 구성요소 | 특성 |
|---|---|
| Attention 없음 | Flash Attention / LayerNorm / Softmax 전부 부재 |
| **FP16-안전 op 만 사용** | Conv / BN / ReLU / Add 등은 모두 FP16 native, FP32 fallback 거의 생성 안 됨 |
| Weights 비중 압도적 | engine ≈ weights (cubin / metadata 합이 cubin 만으로는 ~10 MiB) |
| Batch 4 × 1280² 입력 | 입력 자체가 큼 → activation memory plan 비용 큼 |

### 측정값

| Config | Disk MiB | Total Load ms |
|---|---:|---:|
| Default | 141.80 | **4,075** |
| 권장 cmd (variant 4) | **139.72** | **201** |
| 절감률 | **−1.5%** | **20× 단축** |

### CNN 의 특이점 — Disk 효과는 없지만 Loading 은 큼

```
Disk:    141 MiB → 140 MiB (effectively 0)
  ↑ CNN 은 FP32 fallback cubin 자체가 없어서 precisionConstraints=obey 가 제거할 게 없음

Loading: 4,075 ms → 201 ms (20배 단축)
  ↑ Cross-platform full runtime header 의 파싱 비용은 layer 수와 batch 크기에 비례
  ↑ Flash Attention plugin lib 부재로 RF-DETR 만큼 극적이진 않지만 여전히 4초 → 0.2초
```

### 다른 CNN 모델 (YOLOv5/8/11, ResNet, EfficientNet) 으로의 유추
- 매트릭스에서 검증된 YOLOv7 패턴 (disk 변화 적음, loading 단축 큼) 이 거의 동일하게 적용 예상
- attention 무관, conv-heavy 모델은 같은 처방

### 권장 cmd 적용 시
- ✅ Disk: ~0 (CNN 은 fallback 없음)
- ✅ Loading: **20× 단축**
- ✅ 정확도: precisionConstraints=obey 가 제거할 fallback path 자체가 없어 **정확도 영향 사실상 0**

---

## 3 모델 종합 비교

<table>

| 항목 | RF-DETR | D-FINE | YOLOv7 |
|---|---|---|---|
| 아키텍처 | Transformer + Flash | Transformer 일반 | CNN |
| ONNX 크기 | 119 MiB | 118 MiB | 280 MiB |
| ONNX nodes | 2,362 | 2,926 | 631 |
| Input | 1×3×1024×1024 | 1×3×1024×1024 | 4×3×1280×1280 |
| FP16 baseline disk | 213 MiB | 128 MiB | 142 MiB |
| FP16 baseline loading | **26 s** | 4.7 s | 4.1 s |
| 권장 cmd disk | 91 MiB | 88 MiB | 140 MiB |
| 권장 cmd loading | 0.50 s | 0.58 s | 0.20 s |
| **Disk 절감** | **−57%** | **−31%** | −1.5% |
| **Loading 절감** | **52×** | **8×** | **20×** |
| obey 정확도 영향 | ⚠️ 0.1~0.2% 가능 | ⚠️ 0.1~0.2% 가능 | ✅ ~0 (fallback 없음) |

</table>

## 권장 cmd 의 보편적 적용 가능성

| 모델 종류 | 권장 cmd 효과 | 정확도 검증 필요? |
|---|---|---|
| Transformer + Flash Attention (RF-DETR 류) | Disk 큰 절감 + Loading 큰 단축 | ✅ 필수 |
| Transformer 일반 attention (D-FINE, DETR, ViT 류) | Disk 중간 절감 + Loading 큰 단축 | ✅ 필수 |
| CNN 일반 (YOLO, ResNet, EfficientNet 등) | Disk 변화 미미 + Loading 큰 단축 | △ 권장이지만 위험 낮음 |
| Multi-modal / VLM / LLM | 미검증 (FP32 fallback 비중 큼 예상) | ✅ 필수 |

→ **회사 모든 모델에 안전하게 적용 가능**한 default cmd 로 채택 가능. 단 transformer 계열은 deploy 전 mAP 검증 1회 수행.

## 같이 보면 좋은 문서

- [옵션별 역할 상세 (기술 구현 계층까지)](02-options-explained.md)
- [정확도 가드 옵션 (layerPrecisions, layerOutputTypes 활용)](11-accuracy-guard.md)
- [최종 검증 Table (3 모델 × 4 variant)](09-final-verification.md)
