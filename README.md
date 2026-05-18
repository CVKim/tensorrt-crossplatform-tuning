# TensorRT Cross-Platform Tuning

Linux 서버에서 빌드 → Windows AMD64 런타임으로 deploy 하는 cross-platform TensorRT 엔진의 **파일 사이즈 + 로딩 시간 최적화** 매트릭스. RF-DETR / D-FINE 등 transformer 기반 detection 모델에서 FP16 cross-platform 변환 시 발생하는 disk 폭증 / 로딩 26초 이슈를 trtexec 옵션 3개로 해결한 가이드.

---

## TL;DR

기존 cross-platform FP16 변환 cmd 에 **3 옵션 추가** + 1 옵션 제거:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

기존 cmd 대비:
- ➕ `--precisionConstraints=obey` — FP32 fallback cubin 제거 (disk 감축, transformer 에서 큰 효과)
- ➕ `--versionCompatible --excludeLeanRuntime` — lean runtime header 사용 (loading 감축, **모든 모델**)
- `--directIO` 는 효과 0 이지만 유지해도 무해 (cmd 호환성 차원에서 유지 권장)

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

## 핵심 발견

### 왜 FP16 cross-platform 만 폭증하는가
- `--fp16` 단독은 FP16-위험 layer (softmax, LayerNorm) 에 FP32 fallback cubin 을 같이 embed
- `--hardwareCompatibilityLevel=ampere+` 는 sm_80/86/87/89/90 모든 SM cubin embed → fallback 양도 5배 증폭
- Cross-platform default 는 보수적인 **full runtime header** 사용 → IExecutionContext 생성 26초

### Flash Attention 호환성
`--precisionConstraints=obey` 는 Flash Attention 을 **오히려 더 강하게** 적용시킴. FP32 fallback 으로 분리되던 attention 블록이 모두 fused MHA 로 collapse (layer 수 264 → 241).

## 문서

- [문제 정의 (cross-platform FP16 26초 로딩 이슈)](docs/01-problem-statement.md)
- [옵션별 역할 상세](docs/02-options-explained.md)
- [측정 방법론 (Docker on Windows + NGC TRT)](docs/03-benchmark-methodology.md)
- [RF-DETR 매트릭스 결과](docs/04-results-rf-detr.md)
- [D-FINE 매트릭스 결과](docs/05-results-d-fine.md)
- [YOLOv7 (CNN) 매트릭스 결과](docs/08-results-yolov7.md)
- [FP16 vs FP32 비교](docs/06-fp32-comparison.md)
- [Production cmd & deploy 체크리스트](docs/07-recommended-cmd.md)
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

- ✅ Transformer + Flash Attention (RF-DETR): disk **−57%**, 로딩 **52× 단축**
- ✅ Transformer 일반 attention (D-FINE): disk **−31%**, 로딩 **8× 단축**
- ✅ **CNN (YOLOv7)**: disk −1.5% (CNN 은 fallback 없음), **로딩 20× 단축**
- ⚠️ FP32 모드: 추가 옵션 불필요 (기존 cmd 그대로)

**결론**: 모델 종류 무관하게 안전. cross-platform full runtime header 의 4~26초 로딩 비용은 모든 모델에 부담되었고, lean runtime 으로 swap 하면 모두 sub-1초로 단축.

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
