# Production cmd & 배포 체크리스트

## FP16 변환 cmd (권장)

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

기존 cmd 대비 변경점:
- ➕ `--precisionConstraints=obey`
- ➕ `--versionCompatible`
- ➕ `--excludeLeanRuntime`
- ➖ `--directIO` (효과 0)

## FP32 변환 cmd (기존 유지)

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --onnx=$ONNX \
  --saveEngine=$ENGINE
```

(`--directIO` 는 있어도 없어도 동일. 빼는 편 권장)

## 적용 시 확인 사항

### 1. 빌드 측 (Linux 서버 파이프라인)

- [ ] trtexec 버전 = TensorRT 10.8.0.43 인지 확인 (`trtexec --version`)
- [ ] FP16 변환 시 위 권장 cmd 적용
- [ ] FP32 변환 시 기존 cmd 유지 (또는 `--directIO` 제거)
- [ ] 빌드 성공 — `--precisionConstraints=obey` 가 build 실패시키면 ONNX 에 FP16 불가 layer 존재 의심

### 2. 배포 측 (Windows AMD64 deploy 머신)

- [ ] 변환된 engine 을 production app 에 로드
- [ ] 로그에서 확인:
  - `Loaded engine size: X MiB` — FP16 권장 cmd 적용 시 RF-DETR ≈ 91 MiB, D-FINE ≈ 86 MiB
  - `Number of aux streams is N` 직후 다음 줄까지 시간이 0.5 초 이내인지

### 3. Inference 정확도 검증

- [ ] 기존 engine vs 새 engine 으로 동일 입력 detection 결과 비교
- [ ] mAP 측정 (있는 경우)
- [ ] Bounding box 좌표 / class confidence 의 의미 있는 차이 확인
- [ ] 차이 < 0.5%p 면 production 채택

## 정확도 실패 시 fallback

`--precisionConstraints=obey` 가 LayerNorm / Softmax 의 FP16 강제로 정확도 손실 가능성 있음. 그 경우 이 옵션만 제거:

```bash
trtexec \
  --tacticSources=+CUBLAS_LT \
  --runtimePlatform=WindowsAMD64 \
  --hardwareCompatibilityLevel=ampere+ \
  --versionCompatible \
  --excludeLeanRuntime \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16
```

결과 (config 09):
- RF-DETR: 203 MiB / 0.56 s — 정확도 100% 보존, 로딩 단축 효과는 그대로
- D-FINE: 126 MiB / 0.68 s

## 빌드 시간 비교 (참고)

| Config | RF-DETR build s | D-FINE build s |
|---|---:|---:|
| FP16 baseline (01) | 308 | 364 |
| **FP16 권장 (15)** | **171** (−44%) | **375** (+3%) |
| FP32 default | 75 | 154 |

FP16 권장 cmd 는 RF-DETR 에서 빌드 시간도 절반 가까이 단축. D-FINE 은 거의 동일.

## 다른 모델 적용 가능성

| 모델 종류 | 권장 cmd 적용 효과 | 검증 |
|---|---|---|
| Transformer (Flash Attention 사용) | 큰 효과 (RF-DETR 사례, 60배 로딩 단축) | ✓ |
| Transformer (일반 attention) | 중간 효과 (D-FINE 사례, 8배 로딩 단축) | ✓ |
| CNN (YOLO, ResNet 등) | 효과 미미하지만 부작용 없음 | 미검증 — 적용 안전 |
| ViT, Swin, LLM 류 | 큰 효과 예상 | 미검증 |

→ **회사 ONNX → TRT 파이프라인의 default cmd 로 채택 가능**. 모델 종류에 무관하게 안전.

## Engine 파일 명명 규칙 (제안)

기존:
```
<RECIPE>.trt
```

추가 옵션 적용 식별을 위해:
```
<RECIPE>__fp16_v2.trt   (FP16 권장 cmd 결과)
<RECIPE>__fp32.trt      (FP32 변환)
```

또는 metadata 파일에 변환 cmd hash 같이 보존 권장 (regression 추적 용도).

## Rollback 절차

새 cmd 적용 후 운영 이슈 발생 시:
1. 기존 cmd 로 재변환 (baseline engine 복원)
2. `--precisionConstraints=obey` 만 제거 (config 09 — 정확도 안전 fallback)
3. 두 단계 다 안 되면 cross-platform 자체를 일시 정지하고 local Windows build 로 전환

## 회사 파이프라인 적용 체크리스트

- [ ] Linux 빌드 서버의 trtexec wrapper script 에 위 cmd 반영
- [ ] FP16 / FP32 분기 처리
- [ ] CI 에서 1회 빌드해서 size + accuracy regression 확인
- [ ] Windows production 측 1대에 우선 배포 → 1일 모니터링
- [ ] 이상 없으면 전체 deploy
