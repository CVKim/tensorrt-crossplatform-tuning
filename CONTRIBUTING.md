# Contributing

## Branch 정책

- **`main`** — production-ready, 검증 완료된 변환 cmd / 문서. 직접 commit 금지.
- **`dev`** — 신규 실험 진행. 모든 실험 결과는 dev 에 먼저 push.

### Workflow

```
dev (실험 진행)
    │
    ├─ commit: 새 매트릭스 실행
    ├─ commit: 결과 분석 + 문서화
    └─ PR → main (review 후 merge)

main (검증 완료)
    └─ tag: v1.0.0, v1.1.0 등 release point
```

## 신규 실험 추가 절차

새 모델 또는 새 옵션 조합 테스트 시:

1. **dev 브랜치 checkout**
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **매트릭스 실행**
   ```bash
   bash scripts/run_in_docker.sh <MODEL_TAG> /path/to/model.onnx
   python scripts/bench_load_python.py
   ```

3. **결과 commit**
   - `benchmarks/<MODEL>_*.csv` 추가
   - `docs/<NN>-results-<model>.md` 새 모델 결과 분석 추가
   - `README.md` 의 TL;DR 표 업데이트 (필요시)

4. **dev push**
   ```bash
   git add benchmarks/ docs/
   git commit -m "Add <MODEL> matrix results"
   git push origin dev
   ```

5. **PR → main**
   - GitHub UI 에서 dev → main PR 생성
   - Reviewer 가 결과 sanity check 후 squash merge

## 새 옵션 매트릭스 추가 시

`scripts/run_matrix_generic.sh` 끝에 `run_cfg` 한 줄 추가:

```bash
run_cfg "18_my_new_flag" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --my-new-flag
```

매트릭스 재실행 후 결과 비교.

## 회사 IP 보호

이 repo 는 public/internal 어느 쪽에 있든 다음은 commit 하지 말 것:

- ❌ ONNX 파일 (`*.onnx`) — `.gitignore` 에 포함
- ❌ TRT engine (`*.trt`) — `.gitignore` 에 포함
- ❌ 회사 recipe name (예: `SW_CAM2_HBB_<UUID>-style 식별자`) — 매트릭스 config 이름만 commit
- ❌ 회사 내부 경로 (예: 빌드 서버의 temp / artifact 디렉토리) — `./model.onnx` 같은 상대 경로로 치환
- ❌ 회사 docker registry URL — 공개 가능한 NGC 컨테이너 URL 만 사용
- ✅ 익명화된 측정 결과 CSV (config 이름만 보존)
- ✅ 일반화된 trtexec 옵션 분석
- ✅ 측정 방법론 문서

## 코딩 스타일

- Bash: `set -uo pipefail` 필수, 줄바꿈 LF, UTF-8
- Python: PEP 8, 표준 lib 우선
- Markdown: GFM, 표 정렬, 코드 블록은 언어 명시
- 한글 본문 OK (회사 내부 가이드 목적)

## 측정 신뢰도 기준

새 결과 commit 전 확인:

- [ ] 동일 환경에서 2회 이상 측정 (노이즈 ±5%)
- [ ] baseline (`01_base_ampere_directIO`) 결과가 211 MiB ± 1% 범위인지
  - 회사 Linux 서버 결과와 일치 여부 검증용
- [ ] FP16 권장 (`15_strictFp16_verCompat`) 결과가 90 MiB ± 1 MiB 범위인지
- [ ] CSV 의 모든 `status` 컬럼이 `OK` 인지

벗어나면 환경 점검 후 재측정.
