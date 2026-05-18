# 측정 방법론

## 환경 셋업

### 호스트
- **OS**: Windows 11 Pro
- **GPU**: NVIDIA RTX 3080 (sm_86) × 2
- **NVIDIA Driver**: 591.86
- **CUDA**: 12.8 (Driver 호환)

### 빌더 환경 (Docker on Windows)
- **Docker Desktop** 28.2.2, WSL2 backend
- **NGC TensorRT 컨테이너**: `nvcr.io/nvidia/tensorrt:25.01-py3`
  - TensorRT 10.8.0.43 (build b43)
  - CUDA 12.8.61
  - Ubuntu 24.04 base
- **GPU passthrough**: `--gpus all` 플래그 → NVIDIA Container Toolkit

### 측정 환경 (Windows Python)
- **Python**: 3.12.4 (Anaconda)
- **TensorRT Python**: 10.8.0.43 (`pip install tensorrt==10.8.0.43`)
- **별도 venv** 사용 (`.venv-bench`) — 시스템 Python 오염 방지

### 회사 Linux 서버 vs Docker on Windows 등가성

| 항목 | 회사 Linux 서버 | Docker on Windows |
|---|---|---|
| trtexec 바이너리 | NVIDIA 배포 Linux ELF | NGC 컨테이너의 동일한 Linux ELF |
| TensorRT 버전 | 10.8.0.43 | 10.8.0.43 (b43) |
| Linux kernel | 서버 kernel | WSL2 lightweight VM kernel |
| GPU driver | 서버 NVIDIA driver | Windows 591.86 (WSL passthrough) |
| 결과 engine 포맷 | Windows PE + cubin | 동일 |

**경험적 검증**: 회사 Linux 서버 결과 (211 MiB) vs Docker on Windows 결과 (212.88 MiB) = **오차 0.4%**

## 변환 매트릭스 절차

### 1. ONNX 입력 검수

```python
python scripts/inspect_onnx.py path/to/model.onnx
# 출력: input/output name, shape, opset, n_nodes 등
```

### 2. Docker 컨테이너 매트릭스 실행

```bash
docker run --rm --gpus all \
  -v ${PWD}:/work \
  -w /work \
  nvcr.io/nvidia/tensorrt:25.01-py3 \
  bash scripts/run_matrix_generic.sh MODEL_TAG /work/onnx/model.onnx
```

매트릭스는 17개 config:

| # | 이름 | 추가 옵션 |
|---|---|---|
| 01 | base_ampere_directIO | `--directIO --tacticSources=+CUBLAS_LT` (기존 production cmd) |
| 02 | base_ampere | 01 에서 `--directIO` 제거 |
| 03 | ampere_lean | 02 + `--maxAuxStreams=0 --profilingVerbosity=none` |
| 04 | ampere_LTonly | 03 + `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` |
| 05 | ampere_lean_SIM | 03 + simplified ONNX |
| 06 | ampere_stripped | 03 + `--stripWeights` (deploy refit 필요) |
| 07 | ampere_optLvl3 | 03 + `--builderOptimizationLevel=3` |
| 08 | ampere_optLvl2 | 03 + `--builderOptimizationLevel=2` |
| 09 | ampere_verCompat | 03 + `--versionCompatible --excludeLeanRuntime` ⭐ |
| 10 | ampere_combined_SIM | stripped + LT-only + lean + SIM |
| 11 | none_runtimeWin | `--hardwareCompatibilityLevel=none` (sm_86 only) |
| 12 | ampere_strictFp16 | 03 + `--precisionConstraints=obey` ⭐ |
| 13 | ampere_noTF32_optLvl3 | optLvl3 + `--noTF32` |
| 14 | ampere_lean_SIM_optLvl3 | lean + SIM + optLvl3 |
| 15 | strictFp16_verCompat | 12 + 09 결합 ⭐⭐ 권장 |
| 16 | strictFp16_stripped | 12 + `--stripWeights` |
| 17 | kitchen_sink | 15 + stripWeights + SIM |

FP32 매트릭스는 5 configs (`fp32_01` ~ `fp32_15`) — `--fp16` 만 제거 + 위 옵션 조합.

각 config 마다:
- Build 시간 측정 (wall clock)
- Engine bytes / MiB 기록
- `--skipInference` 로 빌드 후 inference 테스트 생략 (시간 절감)

### 3. Python 로딩 시간 측정 (Windows 측)

```bash
.venv-bench/Scripts/python.exe scripts/bench_load_python.py
```

각 engine 에 대해:

```python
runtime = trt.Runtime(LOGGER)

t0 = time.perf_counter()
engine = runtime.deserialize_cuda_engine(raw_bytes)  # → deserialize_ms
t1 = time.perf_counter()

ctx = engine.create_execution_context()              # → context_ms
t2 = time.perf_counter()
```

`total_ms = deserialize_ms + context_ms`

이 값이 운영 단의 "Loaded engine size: X MiB" 직후 `IExecutionContext creation` 까지의 멈춤 시간에 해당.

### 측정 일관성

- 각 engine 마다 fresh `Runtime` 인스턴스 생성 (warm-up effect 배제)
- GC `gc.collect()` 로 이전 engine reference 명시적 해제
- 동일 GPU, 동일 driver 에서 연속 측정 (cold pass)
- 노이즈 ±5% 이내 재현성

## 한계와 주의

### Build host GPU SM 의 영향
- 빌더는 build-host GPU 에서 candidate cubin 을 profile 해 fastest tactic 선택
- 우리 측정은 RTX 3080 (sm_86) 에서 build → 회사 Linux 서버의 GPU SM 이 다르면 tactic 약간 다를 수 있음
- 하지만 `--hardwareCompatibilityLevel=ampere+` 가 모든 Ampere+ cubin 을 embed 하므로 **engine size 는 build-host GPU SM 과 무관**
- Loading 시간도 size 와 무관한 runtime header issue 라 build-host GPU 무관
- → **size / loading 측정값은 회사 Linux 서버에서도 동일 비율로 재현됨**

### 측정에 포함되지 않은 것
- Inference 속도 (TPS / latency) — `--skipInference` 사용
- Detection accuracy / mAP — 사용자 production 검증 단계로 위임
- Memory 사용량 (engine.device_memory_size) 은 CSV 에는 있지만 본문 분석엔 활용 안 함

## 참고: 매트릭스 자동화 스크립트

[scripts/run_matrix_generic.sh](../scripts/run_matrix_generic.sh) 가 17 config 을 순차 실행.

매트릭스 1회 풀 빌드 시간 ≈ 1.5 ~ 3 시간 (모델 크기 의존). RF-DETR (2362 nodes) 약 2시간, D-FINE (2926 nodes) 약 2.5시간.
