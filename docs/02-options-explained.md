# 옵션별 역할 상세

세 가지 핵심 옵션 + `--directIO` 부산물 제거 + `--hardwareCompatibilityLevel` 참고 정리.

---

## 1. `--precisionConstraints=obey` ⭐ disk 절감

### 무엇을 하는가
Builder 가 layer 정밀도 결정 시 사용자가 지정한 precision (`--fp16`) 을 **반드시 따르도록 강제**. FP16 layer 에 대한 FP32 fallback path 생성을 금지.

### 동작 비교

| 설정 | builder 동작 | cubin 갯수 |
|---|---|---|
| `--fp16` 단독 (default `none`) | FP16-위험 layer (softmax, LayerNorm) 에 FP32 fallback cubin 추가 | 2배 (FP16 + FP32 트랙) |
| `--fp16 --precisionConstraints=prefer` | FP16 우선, 안 되면 fallback 허용 | 모델별 다름 |
| `--fp16 --precisionConstraints=obey` | FP16 강제, 불가능 layer 발견 시 build 실패 | **1배 (FP16 only)** |

### 실측 효과 (disk)

| 모델 | 옵션 없음 | 옵션 추가 | 변화 |
|---|---:|---:|---|
| RF-DETR | 213 MiB | 101 MiB | **−53%** (Flash Attention LayerNorm/Softmax fallback 다수 제거) |
| D-FINE  | 128 MiB |  87 MiB | **−32%** (일반 attention 이라 원래 fallback 적었음) |

### 주의사항

> ⚠️ 빌드 단계에서 FP16-불가 layer 가 있으면 **build 실패**. RF-DETR / D-FINE 은 둘 다 빌드 성공함.
> ⚠️ Deploy 후 detection accuracy / mAP 1회 검증 권장.
> ⚠️ 정확도 떨어질 경우 이 옵션만 제거하면 됨 (verCompat 효과는 그대로 살아 있음).

### Flash Attention 호환성

Flash Attention 은 FP16 native 라 `obey` 가 오히려 Flash path 만 살아남게 만듦.

| 설정 | RF-DETR layer 수 (post-fusion) |
|---|---:|
| baseline (`--fp16` 단독) | 264 (attention 22개가 개별 layer 로 보임) |
| `obey` 추가 | 241 (attention 들이 fused MHA 로 collapse) |

→ FP32 fallback 으로 분리되던 attention 블록이 모두 fused MHA 로 collapse. Flash path 만 살아남는 것이 확인됨.

---

## 2. `--versionCompatible` ⭐ 로딩 단축 (절반)

### 무엇을 하는가
Engine 을 TRT 버전 변경에 견딜 수 있게 빌드. Build TRT 와 deploy TRT 가 minor 버전 차이가 있어도 engine 이 로드되도록 lean runtime wrapper 추가.

### 동작 차이

| 설정 | Engine 안에 들어가는 것 | Loading 비용 |
|---|---|---|
| 옵션 없음 (default) | **Full runtime header** + 보수적 validation metadata | Plugin lib 매칭 + multi-SM cubin 검증 매번 walking |
| `--versionCompatible` 단독 | **Lean runtime header** + lean wrapper embed | Lean 매칭만, wrapper 크기만큼 disk ↑ |
| `--versionCompatible --excludeLeanRuntime` | **Lean runtime header** (wrapper 는 deploy TRT 설치본에서 가져옴) | Lean 매칭만, disk 도 최소 |

### 단독 효과 (config 09)

- RF-DETR loading: **27.7 s → 0.56 s** (약 50배)
- D-FINE  loading: **4.55 s → 0.68 s** (약 7배)
- Disk 변화: RF-DETR −10 MiB / D-FINE −2 MiB (부산물)

---

## 3. `--excludeLeanRuntime` ⭐ 로딩 단축 (짝)

### 무엇을 하는가
`--versionCompatible` 모드에서 추가되는 lean runtime wrapper 를 engine 에 embed 하지 않음. Deploy machine 의 TRT 설치본 (`nvinfer.dll`) 의 lean runtime 을 사용.

### 짝의 의미

| 조합 | Engine 구조 | 비고 |
|---|---|---|
| 둘 다 없음 (default) | Full runtime header | 로딩 26초 (RF-DETR) |
| `--versionCompatible` 만 | Lean header + embedded wrapper | Disk 가 살짝 늘 수도 |
| **둘 다 함께** | **Lean header only** | **최적** |

### 안전 조건

- Build 와 deploy 의 **TRT major 버전이 같으면** 안전 (build 10.8 → deploy 10.8 OK)
- Deploy 의 TRT 가 build TRT 보다 **이전 minor 버전이면** lean wrapper 누락으로 로드 실패 가능
- AIVEX 처럼 build/deploy 모두 동일 TRT 10.8.0.43 사용하면 **100% 안전**

---

## 효과 0 옵션 (시간 낭비 방지)

매트릭스에서 측정한 효과 없는 옵션들. 추가해도 disk / loading 변화 노이즈 수준.

| 옵션 | 실측 효과 | 이유 |
|---|---|---|
| `--directIO` | disk 0, 로딩 0 | I/O tensor format 강제 옵션 (입출력 name 과 무관). RF-DETR / D-FINE 모두 효과 없음 |
| `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` | RF-DETR ±0.5 MiB / D-FINE −5 MiB | 일반 attention 모델에선 약간 효과, transformer 에선 0 |
| `--maxAuxStreams=0 --profilingVerbosity=none` | <1 MiB | Metadata 절감만, 핵심 cubin 갯수엔 영향 없음 |
| `--builderOptimizationLevel=2/3` | 0 | Inference 성능에만 영향 (tactic 탐색 깊이), disk / 로딩 무관 |
| `--noTF32` | 0 | TF32 는 FP32 연산을 빠르게 만드는 path. FP16 경로엔 무관 |
| onnx-simplifier / polygraphy fold | <1 MiB | Transformer 는 constant folding 여지가 적음 |

---

## `--hardwareCompatibilityLevel` 비교 (참고)

| 값 | 의미 | Disk 영향 | Deploy 호환성 |
|---|---|---|---|
| **`ampere+`** (권장) | sm_80 / 86 / 87 / 89 / 90 등 모든 Ampere+ cubin embed | +30 MiB | Ampere / Ada / Hopper 모든 GPU 가능 |
| `none` | Build host GPU SM 하나만 embed | −30 MiB | Build host 와 동일 SM GPU 에서만 로드 가능 (비추) |

AIVEX 의 4080 (sm_89) deploy 환경이면 `ampere+` 유지 권장. `sameComputeCapability` 값은 TRT 10.9+ 에서 추가됨 (10.8 에선 미지원).

---

## 옵션 합산 효과 (논리 모델)

```
                  precisionConstraints=obey      versionCompatible + excludeLeanRuntime
                  ──────────────────────         ─────────────────────────────────────
효과 대상         FP32 fallback cubin            runtime header
어디서 줄이나     disk (cubin 갯수)              loading (ctx 생성)
                  ──────────────────────         ─────────────────────────────────────
                              ↓                                ↓
                  RF-DETR: −113 MiB               RF-DETR: −27.2 초
                  D-FINE : −41 MiB                D-FINE : −3.9 초
                  ──────────────────────         ─────────────────────────────────────
                              ↘                                ↙
                       세 옵션 = 두 효과 동시 적용
                  ─────────────────────────────────────────────────────────────────
                  RF-DETR: 213 MiB / 27.7s   →   91 MiB / 0.48s
                  D-FINE : 128 MiB / 4.55s   →   86 MiB / 0.56s
```

세 옵션이 **각자 다른 layer 의 비용을 절감**하기 때문에 효과가 깔끔하게 더해짐 (간섭 없음).
