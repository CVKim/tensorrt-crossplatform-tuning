# Problem Statement

## 배경

AIVEX product 파이프라인은 **Linux 빌드 서버** 에서 ONNX→TRT 변환을 수행하고, 변환된 engine 을 **Windows AMD64** runtime 으로 deploy 합니다. 이를 위해 `--runtimePlatform=WindowsAMD64 --hardwareCompatibilityLevel=ampere+` 조합의 cross-platform 빌드 모드를 사용합니다.

## 기존 변환 cmd

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --directIO \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

대부분의 CNN 계열 모델 (YOLO, ResNet 등) 에선 이 cmd 로 잘 작동했음. 하지만 **transformer 기반 detection 모델** (RF-DETR) 을 변환할 때 두 가지 이슈 발견:

## 이슈

### 1. Engine 파일 size 폭증
- ONNX 원본 119 MiB
- Local Windows native FP16 build: **66 MiB** (정상)
- Cross-platform FP16 build: **211 MiB** (3배 이상)

### 2. Windows 측 로딩 시간 폭증
Production app 로그 (Windows side):

```
[17:28:02] [TRT] Loaded engine size: 211 MiB
[17:28:02] [MS] Number of aux streams is 3
[17:28:02] [MS] Number of total worker streams is 4
... 26초 멈춤 ...
[17:28:28] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation
```

`Number of aux streams` 출력 직후 IExecutionContext 생성 단계에서 **26초** 정체. 운영 단에서 매번 로딩에 30초가 걸리는 상황.

## FP32 변환은 정상 동작

같은 cmd 에서 `--fp16` 만 제거한 FP32 변환은 size / loading 둘 다 정상:
- Disk: 160 MiB
- Loading: 0.66 s

즉 이슈는 **cross-platform + FP16 + transformer** 의 3중 조합에서 발생.

## 조사 결과 요약

원인은 두 가지가 동시에 작용:

1. **FP32 fallback cubin 폭증**
   - `--fp16` 단독은 FP16-위험 layer (softmax, LayerNorm) 에 FP32 fallback cubin 을 같이 embed
   - `--hardwareCompatibilityLevel=ampere+` 는 sm_80/86/87/89/90 다섯 SM 의 cubin 을 다 embed
   - → fallback cubin 양이 5배 증폭, 113 MiB 추가 (213 MiB 의 절반)

2. **Full runtime header 사용**
   - Cross-platform 모드 default 는 plugin lib 매칭 + multi-SM cubin 검증을 위한 보수적 metadata 를 매번 walking
   - Layer 수가 많은 transformer 에서 26초 소요

→ 두 문제를 해결하는 trtexec 옵션 3개 발견. 자세한 내용은 [02-options-explained.md](02-options-explained.md).

## 검증 환경

- TensorRT 10.8.0.43, CUDA 12.8, opset 17
- Build: Docker on Windows (NGC `nvcr.io/nvidia/tensorrt:25.01-py3`, WSL2 backend), RTX 3080 (sm_86)
- Deploy: Windows AMD64 + RTX 3080 / RTX 4080
- Validation: 회사 Linux 서버 빌드 결과 (211 MiB) vs Docker on Windows 빌드 결과 (212.88 MiB) = **0.4% 일치**
