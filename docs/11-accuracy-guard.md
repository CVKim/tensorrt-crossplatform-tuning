# 정확도 가드 (`--layerPrecisions`, `--layerOutputTypes`)

`--precisionConstraints=obey` 가 FP16 cubin 만 강제하면서 LayerNorm/Softmax/Reduce 같은 정밀도-민감 layer 도 FP16 으로 처리하기 때문에, 일부 detection task 에서 미세한 결과 시프트가 발생할 수 있다. 본 문서는 이를 통제하는 보조 옵션 정리.

## 회사 다른 분의 측정 결과 요약

ONNX Opset 13/17 + RTX 30xx/50xx 환경에서:

| Config | bbox 차이 (대비) |
|---|---|
| 옵션 무 (`--fp16` 만) | 1800장 중 1~4 bbox 차이 |
| `--fp16 --noTF32 --precisionConstraints=obey` | 동일하게 1~4 차이 (TF32 무관 확인) |
| `--fp16 --precisionConstraints=obey --layerPrecisions=*:fp16` | 850장 중 **2 bbox 차이** |
| `+ --layerOutputTypes=output:fp32` | **2 bbox 차이** (output FP32 강제) |
| `+ --layerPrecisions="*:fp16,*Resize:fp32"` | 동일 (Resize layer FP32 강제도 변화 X) |

→ Detection task 에서 `obey + fp16` 적용 시 **약 0.1~0.2% bbox 시프트** 가 일반적. 0 으로 만들기 어렵고 (TRT 의 내부 tactic 자유도가 남아 있음) 추가 가드를 통해 worst-case 통제 가능.

## 가드 옵션 3종

### 1. `--layerPrecisions=<pattern>:<precision>`

**의미**: 특정 layer (이름 기반 와일드카드) 에 명시적 precision 부여. `precisionConstraints=obey` 와 함께 사용 시 strict 강제.

**예시**:
```bash
# 모든 layer FP16 명시 (precisionConstraints=obey 와 사용)
--layerPrecisions=*:fp16

# 특정 layer 만 FP32 (Resize, LayerNorm 등 민감 layer 강제 FP32)
--layerPrecisions="*:fp16,/model/model.49/Resize:fp32"

# 여러 layer 동시 지정 (쉼표 구분)
--layerPrecisions="*:fp16,/model/model.49/Resize:fp32,/model/norm1/LayerNormalization:fp32"
```

**주의**:
- Layer 이름은 ONNX 의 정확한 노드 이름 (e.g. `/model/model.49/Resize`) — 와일드카드 패턴 제한적
- 정확한 layer 이름 알려면 `trtexec --verbose` 로 빌드 로그 추출 필요

### 2. `--layerOutputTypes=<pattern>:<type>`

**의미**: 특정 layer 의 **출력 tensor type** 강제. 마지막 output 을 FP32 로 보호하는 데 유용.

**예시**:
```bash
# 최종 output 만 FP32
--layerOutputTypes=output:fp32

# 여러 output (이름 별)
--layerOutputTypes="output_0:fp32,output_1:fp32"
```

**용도**: detection 의 최종 confidence / coordinate 출력을 FP32 로 보존해서 NMS / post-processing 의 정확도 안정성 확보.

### 3. `--noTF32`

**의미**: FP32 path 에서도 Tensor Core 의 TF32 가속 사용 안 함. FP32 의 mantissa 23bit 정밀도 그대로 보존.

**적용 범위**:
- FP32 layer (precisionConstraints=obey 적용 시 거의 없음) 에만 영향
- FP16 layer 자체엔 무영향

**FP16 cmd 에서 효과**: 거의 0 (FP32 path 가 없으므로). 회사 다른 분 측정에서도 FP16 모드에선 noTF32 효과 0 확인됨.

## 권장 적용 단계

### Tier 1: 기본 권장 cmd (이미 검증)

```bash
trtexec ... --fp16 \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime
```

- Disk + Loading 절감 최대
- 정확도 시프트: ~0.1~0.2% 가능 (대부분 모델에서 허용 범위)

### Tier 2: Output 보호 (정확도 민감 모델)

```bash
trtexec ... --fp16 \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --layerPrecisions=*:fp16 \
  --layerOutputTypes=output:fp32
```

- 명시적 layer/output type 강제로 worst-case 통제
- Loading / disk 효과는 Tier 1 과 동일
- detection 의 NMS-friendly 한 FP32 output 보장

### Tier 3: 특정 layer FP32 강제 (정확도 critical)

```bash
trtexec ... --fp16 \
  --precisionConstraints=obey \
  --versionCompatible \
  --excludeLeanRuntime \
  --layerPrecisions="*:fp16,<specific_layer>:fp32,..." \
  --layerOutputTypes=output:fp32
```

- 모델 분석으로 정밀도-민감 layer (보통 LayerNorm / Softmax / Resize / 마지막 Conv) 식별
- 그 layer 만 FP32 강제, 나머지는 FP16
- Layer 이름은 `--verbose` 빌드 로그에서 확인

### Tier 0 (Fallback): obey 제거

```bash
trtexec ... --fp16 \
  --versionCompatible \
  --excludeLeanRuntime
```

- obey 빼면 정확도 영향 0 (cubin 자체 변경 없음)
- Loading 단축은 그대로 (verCompat 효과)
- Disk 는 베이스 대비 약간만 줄음 (RF-DETR 203 MiB, D-FINE 126 MiB, YOLOv7 139 MiB)

## 정밀도-민감 layer 식별 절차

1. **`--verbose` 빌드**:
   ```bash
   trtexec ... --verbose 2>&1 | tee build.log
   ```

2. **로그에서 layer 이름 추출**:
   ```bash
   grep "Layer.*Conv\|LayerNormalization\|Softmax\|Resize" build.log
   ```
   또는 `--exportLayerInfo=layers.json` 으로 layer info 직접 추출.

3. **정밀도 시프트가 큰 layer 후보**:
   - 모든 `*LayerNormalization` layer
   - 모든 `*Softmax` layer
   - Detection head 의 마지막 `Conv` / `Linear`
   - Upsample / Resize layer (interpolation 정밀도)

4. **layer 별 FP32 강제 cmd 작성**:
   ```bash
   --layerPrecisions="*:fp16,/model/.../LayerNormalization:fp32,/head/.../Conv:fp32"
   ```

5. **정확도 비교**: 베이스 engine vs 가드 engine 으로 동일 입력 결과 비교, mAP 측정.

## 모델별 권장 Tier

| 모델 | 권장 시작 Tier | 비고 |
|---|---|---|
| RF-DETR (Flash Attention) | Tier 2 | LayerNorm 많음, output 보호로 worst-case 통제 |
| D-FINE (일반 attention) | Tier 1 → 정확도 OK 면 그대로, 아니면 Tier 2 | |
| YOLOv7 (CNN) | Tier 1 | fallback 자체가 적어 obey 영향 작음 |
| 미검증 새 모델 | Tier 0 부터 검증, OK 면 Tier 1 → Tier 2 | 단계적 적용 |

## 결론

`precisionConstraints=obey` 의 정확도 영향이 우려되는 경우:
1. **Tier 0**: obey 제거 (loading 효과만 챙김, 가장 안전)
2. **Tier 2**: output FP32 가드 (worst-case 통제)
3. **Tier 3**: 특정 layer FP32 명시 (정확도 critical 모델)

`versionCompatible + excludeLeanRuntime` 은 어떤 Tier 에서도 그대로 유지 → **loading 단축 효과는 정확도 영향 0** 로 항상 챙길 수 있음.
