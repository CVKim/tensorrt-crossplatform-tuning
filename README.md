# TensorRT Cross-Platform Tuning

Linux 서버에서 빌드 → Windows AMD64 런타임으로 deploy 하는 cross-platform TensorRT 엔진의 **파일 사이즈 + 로딩 시간 최적화** 매트릭스. RF-DETR / D-FINE 등 transformer 기반 detection 모델에서 FP16 cross-platform 변환 시 발생하는 disk 폭증 / 로딩 26초 이슈를 trtexec 옵션 3개로 해결한 가이드.

---

## TL;DR — FP16 / FP32 분기 처리

회사 파이프라인은 precision 별로 **다른 cmd** 적용:

### ✅ FP16 변환 (권장)

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

기존 사용자 cmd 에 **3 옵션 추가**:
- ➕ `--precisionConstraints=obey` — FP32 fallback cubin 제거 (disk 감축, transformer 에서 큰 효과)
- ➕ `--versionCompatible` — lean runtime header (loading 감축, **모든 모델**)
- ➕ `--excludeLeanRuntime` — versionCompatible 의 짝
- `--directIO` 는 그대로 유지 (효과 0 이지만 무해)

### ✅ FP32 변환 (기존 cmd 그대로)

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --directIO \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --onnx=$ONNX \
  --saveEngine=$ENGINE
```

**FP32 에는 3 옵션을 추가하지 마세요**:
- `--precisionConstraints=obey` 는 FP32 에서 no-op (제거할 fallback 없음)
- `--versionCompatible --excludeLeanRuntime` 은 FP32 에서 살짝 손해 (loading +50~150 ms)

이유는 [docs/06-fp32-comparison.md](docs/06-fp32-comparison.md) 참조.

### 파이프라인 wrapper 예시

```bash
if [[ "$PRECISION" == "fp16" ]]; then
  trtexec ... --fp16 \
    --precisionConstraints=obey \
    --versionCompatible \
    --excludeLeanRuntime
else
  trtexec ...      # FP32: 기존 cmd 유지
fi
```

## 검증된 효과 (TensorRT 10.8.0.43, RTX 3080 build / Windows runtime)

| 모델 | FP16 Default (베이스) | **FP16 + 3옵션 (권장)** | FP32 Default (참고) |
|------|---:|---:|---:|
| **RF-DETR** (Flash Attention 사용) | 213 MiB / 26.1 s | **91 MiB / 0.50 s** | 160 MiB / 0.61 s |
| **D-FINE** (일반 attention) | 128 MiB / 4.69 s | **88 MiB / 0.58 s** | 155 MiB / 0.52 s |
| **YOLOv7** (CNN, attention 없음) | 142 MiB / 4.08 s | **140 MiB / 0.20 s** | 278 MiB / 0.24 s |

- RF-DETR: disk **−57%**, 로딩 **52× 단축**
- D-FINE:  disk **−31%**, 로딩 **8× 단축**
- YOLOv7:  disk −1.5% (CNN 은 fallback 없음), 로딩 **20× 단축**
- FP32 default 는 3 모델 모두 이미 로딩 빠름 → FP32 변환엔 추가 옵션 불필요

## 검증 모델 요약

| 모델 | 아키텍처 | 특징 | 권장 cmd 효과 |
|---|---|---|---|
| **RF-DETR** | Transformer + Flash Attention | DETR + Flash 통합. LayerNorm/Softmax 다수, plugin lib 다수 | Disk −57%, Loading 52× |
| **D-FINE** | Transformer 일반 attention | DETR fine-grained, Flash 미사용. attention 변종 일부 | Disk −31%, Loading 8× |
| **YOLOv7** | Pure CNN | Conv/BN/SiLU 만 (attention 없음). FP16-안전 op 만 사용 | Disk ~0%, Loading 20× |

각 모델 상세 분석은 [모델별 특징 & cmd 적용 가이드](docs/10-model-overview.md) 참조.

## 핵심 발견

### 왜 FP16 cross-platform 만 폭증하는가
- `--fp16` 단독은 FP16-위험 layer (softmax, LayerNorm) 에 FP32 fallback cubin 을 같이 embed
- `--hardwareCompatibilityLevel=ampere+` 는 sm_80/86/87/89/90 모든 SM cubin embed → fallback 양도 5배 증폭
- Cross-platform default 는 보수적인 **full runtime header** 사용 → IExecutionContext 생성 26초

### Flash Attention 호환성
`--precisionConstraints=obey` 는 Flash Attention 을 **오히려 더 강하게** 적용시킴. FP32 fallback 으로 분리되던 attention 블록이 모두 fused MHA 로 collapse (layer 수 264 → 241).

### 정확도 영향 — `obey` 사용 시 주의
회사 다른 분 측정 (Det 모델, 850장 기준) 에서 `obey + fp16` 적용 시 약 **2 bbox 차이 (~0.2%)** 발생. CNN 에선 영향 사실상 0, transformer 에선 검증 필요. 통제하려면 [정확도 가드 가이드](docs/11-accuracy-guard.md) 의 Tier 1~3 단계별 옵션 적용.

## 문서

### 시작
- [문제 정의 (cross-platform FP16 26초 로딩 이슈)](docs/01-problem-statement.md)
- [옵션별 역할 상세 (기술 구현 계층)](docs/02-options-explained.md)
- [측정 방법론 (Docker on Windows + NGC TRT)](docs/03-benchmark-methodology.md)

### 모델별 결과
- [**모델별 특징 & cmd 적용 가이드 (RF-DETR / D-FINE / YOLOv7)**](docs/10-model-overview.md) ⭐
- [RF-DETR 매트릭스 결과 (transformer + Flash Attention)](docs/04-results-rf-detr.md)
- [D-FINE 매트릭스 결과 (transformer 일반 attention)](docs/05-results-d-fine.md)
- [YOLOv7 매트릭스 결과 (CNN)](docs/08-results-yolov7.md)

### Precision / 적용
- [FP16 vs FP32 비교](docs/06-fp32-comparison.md)
- [Production cmd & deploy 체크리스트](docs/07-recommended-cmd.md)
- [**정확도 가드 (layerPrecisions / layerOutputTypes 활용)**](docs/11-accuracy-guard.md) ⭐
- [**최종 검증 Table (3 모델 × 4 variant)**](docs/09-final-verification.md) ⭐

## 재현 (Reproduce)

```bash
# 1. Docker + WSL2 환경에서 NGC 컨테이너 pull
docker pull nvcr.io/nvidia/tensorrt:25.01-py3

# 2. 변환 매트릭스 실행 (모델당 17 configs)
bash scripts/run_matrix_generic.sh MODEL_TAG /path/to/model.onnx

# 3. (Windows 측) Python 으로 deserialize + IExecutionContext 시간 측정
python scripts/bench_load_python.py
```

자세한 환경 셋업은 [docs/03-benchmark-methodology.md](docs/03-benchmark-methodology.md).

## 적용 범위 (실측 검증)

| 모델 종류 | 권장 cmd 효과 | 정확도 영향 |
|---|---|---|
| Transformer + Flash Attention (RF-DETR 류) | disk **−57%**, 로딩 **52× 단축** | obey 적용 시 ~0.2% bbox 시프트 가능 (검증 필수) |
| Transformer 일반 attention (D-FINE, DETR, ViT 류) | disk **−31%**, 로딩 **8× 단축** | obey 적용 시 ~0.1~0.2% 시프트 가능 |
| CNN (YOLOv7, YOLO 시리즈, ResNet 등) | disk ~0% (fallback 없음), 로딩 **20× 단축** | obey 영향 사실상 0 |
| FP32 모드 | 추가 옵션 불필요 | — |

**결론**: 모델 종류 무관하게 loading 단축은 모든 모델에 효과적 (cross-platform full runtime header 의 비용은 보편적). Disk 절감은 transformer 에서 큰 효과. 정확도 우려 시 [정확도 가드 가이드](docs/11-accuracy-guard.md) 의 Tier 0~3 단계별 적용.

## 브랜치 정책

- `main` — 검증 완료된 production-ready
- `dev` — 신규 실험 및 검증 진행

## 측정 환경

- **TensorRT**: 10.8.0.43 (NGC container 25.01-py3)
- **CUDA**: 12.8
- **Build host**: Docker on Windows (WSL2 backend) + RTX 3080 (sm_86)
- **Deploy target**: Windows AMD64 + Ampere/Ada GPU (RTX 3080/4080 검증)
- **Driver**: NVIDIA 591.86+

## License

Internal use within AIVEX Product. Refer to company policy for external sharing.
