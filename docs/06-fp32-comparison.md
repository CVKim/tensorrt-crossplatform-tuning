# FP16 vs FP32 비교

FP32 모드에서는 FP16 권장 옵션의 효과가 거의 없거나 오히려 약간 손해. **FP32 변환은 기존 cmd 그대로** 사용하면 됨.

## RF-DETR FP32 매트릭스

| config | disk MiB | total load ms | vs default |
|---|---:|---:|---|
| fp32_01_base_directIO (기존 cmd) | 158.59 | 612 | (baseline) |
| fp32_02_base (`--directIO` 제거) | 159.87 | 639 | ±0 (노이즈) |
| fp32_09_verCompat (`+ versionCompatible + excludeLeanRuntime`) | 160.78 | **686** | **+74 ms (살짝 손해)** |
| fp32_12_strict (`+ precisionConstraints=obey`) | 158.90 | 600 | ±0 |
| fp32_15_strict_verCompat (FP16 권장 옵션 모두 적용) | 159.22 | 614 | ±0 |

## D-FINE FP32 매트릭스

| config | disk MiB | total load ms | vs default |
|---|---:|---:|---|
| fp32_01_base_directIO (기존 cmd) | 154.47 | 489 | (baseline) |
| fp32_02_base | 155.82 | 490 | ±0 |
| fp32_09_verCompat | 154.27 | **554** | **+65 ms** |
| fp32_12_strict | 156.41 | 484 | ±0 |
| fp32_15_strict_verCompat | 156.27 | **655** | **+166 ms** |

## 왜 FP32 에선 효과가 없는가

### `--precisionConstraints=obey` 가 효과 0 인 이유
- FP32 모드는 처음부터 **단일 precision track** (FP32 only, fallback 자체가 없음)
- `obey` 가 봉쇄할 fallback path 가 존재하지 않음 → no-op
- → disk 변화 ±1 MiB 노이즈 수준

### `--versionCompatible --excludeLeanRuntime` 이 살짝 손해인 이유
- FP32 default 가 이미 가벼운 runtime header 사용 (cubin 변종이 적어서 보수적 검증 metadata 가 작음)
- `--versionCompatible` 추가하면 lean wrapper 어댑터가 추가됨 → runtime 초기화 단계 1개 늘어남
- → loading +50 ~ 150 ms 증가 (오히려 손해)
- 단, 절대값은 여전히 0.6 ~ 0.7 초 수준이라 사용에 무리 없음

## FP32 권장 cmd (사용자 기존 cmd 그대로)

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --onnx=$ONNX \
  --saveEngine=$ENGINE
```

(`--directIO` 는 효과 0 이라 있어도 없어도 됨. 회사 inference 코드에서 명시적 format 가정 안 하면 제거 추천 — cmd 단순화 효과)

## FP16 vs FP32 옵션 처방 비교

| 옵션 | FP16 모드 | FP32 모드 |
|---|---|---|
| `--precisionConstraints=obey` | ✅ **필수** (disk 절반 감축) | ⬜ no-op |
| `--versionCompatible --excludeLeanRuntime` | ✅ **필수** (로딩 60배 단축) | ⚠️ 살짝 손해 (사용 X) |
| `--directIO` | 효과 없음 | 효과 없음 |
| `--tacticSources=+CUBLAS_LT` | 기존 유지 | 기존 유지 |
| `--hardwareCompatibilityLevel=ampere+` | 기존 유지 | 기존 유지 |

→ **회사 파이프라인 정리**: FP16 변환은 새 cmd, FP32 변환은 기존 cmd 그대로 (분기 처리).

## 모델 × 정밀도 매트릭스

| | RF-DETR | D-FINE |
|---|---|---|
| **FP16 Default (베이스)** | 213 MiB / 27.7 s | 128 MiB / 4.55 s |
| **FP16 + 3옵션 (개선)** | **91 MiB / 0.48 s** | **86 MiB / 0.56 s** |
| **FP32 Default (기존 cmd)** | 160 MiB / 0.66 s | 155 MiB / 0.47 s |

## 정리

- FP32 cross-platform 은 그 자체로 이슈 없음 → 회사가 다른 FP32 모델에서 문제 없었던 이유 확인
- FP32 변환에 FP16 권장 옵션을 추가하는 건 효과 0 또는 약간 손해 → 적용하지 말 것
- FP16 변환과 FP32 변환을 cmd 레벨에서 **분기** 처리 권장
