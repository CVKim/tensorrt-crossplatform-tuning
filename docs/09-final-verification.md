# 최종 검증 Table (3 모델 × 4 variant)

권장 cmd 가 모델 종류 (transformer Flash / transformer 일반 / CNN) 불문하고 안정적으로 효과를 내는지 종합 정리.

## 종합 표

| 모델 | Default | + obey 만 | + verCompat + excludeLeanRuntime 만 | **+ 세 옵션 전부 (권장)** |
|------|--------:|---------:|--------------------------------:|---------------------------:|
| **RF-DETR** (Flash Attention) | 212.88 MiB / 26.13 s | 100.57 MiB / 25.61 s | 202.94 MiB / 0.55 s | **91.38 MiB / 0.50 s** |
| **D-FINE** (일반 attention) | 128.19 MiB / 4.69 s | 86.62 MiB / 4.09 s | 126.04 MiB / 0.57 s | **88.07 MiB / 0.58 s** |
| **YOLOv7** (CNN) | 141.80 MiB / 4.08 s | 142.24 MiB / 4.25 s | 139.22 MiB / 0.21 s | **139.72 MiB / 0.20 s** |

(Disk MiB / Total Load Seconds)

## 절감률

| 모델 | Disk 절감 | Loading 절감 (×배수) |
|------|----------:|--------------------:|
| RF-DETR | **−57.1%** | **52× 단축** (26.13 → 0.50 s) |
| D-FINE | **−31.3%** | **8× 단축** (4.69 → 0.58 s) |
| YOLOv7 | **−1.5%** | **20× 단축** (4.08 → 0.20 s) |

## 옵션별 기여도 분석

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│ --precisionConstraints=obey     │ Disk 절감 (FP32 fallback cubin 제거)     │
│                                 │                                          │
│   RF-DETR: −112 MiB ★★★         │   transformer 효과 큼 (Flash 의 LayerNorm/ │
│   D-FINE :  −41 MiB ★★          │   Softmax 다수 → fallback 다수)          │
│   YOLOv7 :   ±0 MiB             │   CNN 효과 0 (fallback 자체 없음)        │
└─────────────────────────────────┴──────────────────────────────────────────┘

┌─────────────────────────────────┬──────────────────────────────────────────┐
│ --versionCompatible             │ Loading 절감 (lean runtime header)       │
│ --excludeLeanRuntime            │                                          │
│                                 │                                          │
│   RF-DETR: −25.6 s ★★★          │   3 모델 모두 큰 효과                    │
│   D-FINE :  −4.1 s ★★★          │   cross-platform full runtime header 의  │
│   YOLOv7 :  −3.9 s ★★★          │   파싱 비용은 모든 모델에서 동일하게 발생  │
└─────────────────────────────────┴──────────────────────────────────────────┘

┌─────────────────────────────────┬──────────────────────────────────────────┐
│ --directIO                      │ 효과 0 (모든 모델)                       │
│                                 │ I/O format 강제 옵션, kernel/runtime 무관 │
└─────────────────────────────────┴──────────────────────────────────────────┘
```

## 결론

1. **Loading 단축 효과는 보편적 (모든 모델 8 ~ 52× 단축)**
   - `--versionCompatible --excludeLeanRuntime` 이 cross-platform 모드의 full runtime header 비용을 lean header 로 swap
   - 이 비용은 transformer (Flash plugin 매칭) 든 CNN (layer scratch 메모리 계획) 이든 모두에게 부담

2. **Disk 절감은 모델 구조 의존적**
   - `--precisionConstraints=obey` 는 FP32 fallback cubin 이 있는 모델에서만 효과
   - Transformer = LayerNorm/Softmax 다수 → fallback 다수 → 큰 효과
   - CNN = FP16-안전 op 만 → fallback 없음 → 효과 0

3. **CNN 에서도 권장 cmd 적용 안전**
   - obey 가 no-op 이어도 부작용 없음 (disk ±0)
   - verCompat 효과만 챙겨도 충분히 큰 가치 (loading 20× 단축)

→ **회사 ONNX→TRT 파이프라인의 default cmd 로 통합 채택 권장**.

## 최종 production cmd (재차 정리)

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

기존 사용자 cmd 에서 세 줄만 추가 (`--precisionConstraints=obey`, `--versionCompatible`, `--excludeLeanRuntime`), `--directIO` 는 유지해도 영향 없음.

FP32 변환은 기존 cmd 그대로 (3 옵션 추가하면 오히려 살짝 손해 — [06-fp32-comparison.md](06-fp32-comparison.md) 참조).
