# Benchmark Data

이 디렉토리는 17-config FP16 + 5-config FP32 매트릭스의 실측 데이터를 담고 있습니다.

## 파일 설명

| 파일 | 내용 | 측정 환경 |
|---|---|---|
| `load_bench_combined.csv` | Python TRT 로 측정한 deserialize + IExecutionContext 생성 시간 | Windows 11 + RTX 3080 + TRT 10.8.0.43 |
| `RF-DETR_fp16_build_summary.csv` | RF-DETR FP16 17 config 빌드 측정값 | Docker on Windows + NGC 25.01-py3 |
| `RF-DETR_fp32_build_summary.csv` | RF-DETR FP32 5 config 빌드 측정값 | 동상 |
| `D-FINE_fp16_build_summary.csv` | D-FINE FP16 17 config 빌드 측정값 | 동상 |
| `D-FINE_fp32_build_summary.csv` | D-FINE FP32 5 config 빌드 측정값 | 동상 |

## load_bench_combined.csv 필드

| 컬럼 | 단위 | 설명 |
|---|---|---|
| `model` | str | RF-DETR / D-FINE |
| `config` | str | 매트릭스 config 이름 (`01_base_ampere_directIO` 등) |
| `disk_bytes` | bytes | engine 파일 크기 |
| `disk_mib` | MiB | engine 파일 크기 (MiB 환산) |
| `deserialize_ms` | ms | `runtime.deserialize_cuda_engine(blob)` 시간 |
| `context_ms` | ms | `engine.create_execution_context()` 시간 |
| `total_ms` | ms | deserialize + context 합계 (≈ 운영 단의 "로딩 시간") |
| `num_layers` | int | post-fusion layer 수 (참고용) |
| `num_io` | int | 입출력 tensor 수 |
| `refittable` | 0/1 | stripWeights engine 식별용 |
| `device_memory_mib` | MiB | inference 시 GPU 메모리 필요량 |
| `status` | str | OK / FAILED_DESERIALIZE 등 |

## 빌드 summary CSV 필드

각 모델/precision 별:

| 컬럼 | 단위 | 설명 |
|---|---|---|
| `name` | str | 매트릭스 config 이름 |
| `onnx_variant` | str | base / sim / folded 등 입력 ONNX 변형 |
| `engine_bytes` | bytes | 빌드된 engine 크기 |
| `engine_mib` | MiB | 환산 |
| `build_seconds` | s | trtexec wall-clock 빌드 시간 |
| `exit_code` | int | 0 = 성공 |
| `extra_flags` | str | 사용된 trtexec 옵션들 |

## 측정 일자

2026-05-15 ~ 2026-05-18, TensorRT 10.8.0.43, RTX 3080 build host.

자세한 측정 방법론은 [../docs/03-benchmark-methodology.md](../docs/03-benchmark-methodology.md) 참조.
