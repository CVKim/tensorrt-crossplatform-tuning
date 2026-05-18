#!/usr/bin/env bash
# FP32 conversion matrix (full 5-config version).
# Tests whether the FP16-effective flags also help in FP32 mode.
# Usage: run_matrix_fp32.sh <MODEL_TAG> <ONNX_PATH>
set -uo pipefail

MODEL_TAG="${1:?model tag required}"
ONNX_RAW="${2:?onnx raw path required}"

ENG_DIR="/work/engines/${MODEL_TAG}"
LOG_DIR="/work/logs/${MODEL_TAG}"
SUMMARY="${LOG_DIR}/_fp32_summary.csv"
mkdir -p "${ENG_DIR}" "${LOG_DIR}"
echo "name,onnx_variant,engine_bytes,engine_mib,build_seconds,exit_code,extra_flags" > "${SUMMARY}"

ONNX_STEM=$(basename "${ONNX_RAW}" .onnx)
# NOTE: no --fp16 anywhere
COMMON_CROSS="--runtimePlatform=WindowsAMD64 --skipInference"

run_cfg () {
  local cfg="$1"; shift
  local extra="$*"
  local out="${ENG_DIR}/${ONNX_STEM}__${cfg}.trt"
  local log="${LOG_DIR}/${cfg}.log"
  echo "=== [${MODEL_TAG}] [${cfg}] ==="
  local t0=$(date +%s)
  trtexec --onnx="${ONNX_RAW}" --saveEngine="${out}" ${COMMON_CROSS} ${extra} > "${log}" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local sec=$((t1 - t0))
  local bytes=0
  if [[ -f "${out}" ]]; then bytes=$(stat -c %s "${out}"); fi
  local mib=$(awk -v b="${bytes}" 'BEGIN{printf "%.2f", b/1024/1024}')
  printf "%s,%s,%d,%s,%d,%d,\"%s\"\n" "${cfg}" "${ONNX_STEM}.onnx" "${bytes}" "${mib}" "${sec}" "${rc}" "${extra}" >> "${SUMMARY}"
  echo "    bytes=${bytes} (${mib} MiB)  build=${sec}s  rc=${rc}"
}

# fp32_01: 사용자 베이스 cmd (--fp16 만 제거) — directIO 포함
run_cfg "fp32_01_base_directIO" \
  --hardwareCompatibilityLevel=ampere+ --directIO --tacticSources=+CUBLAS_LT

# fp32_02: --directIO 제거
run_cfg "fp32_02_base" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT

# fp32_09: verCompat 적용 (FP32에서도 loading 단축 효과 검증)
run_cfg "fp32_09_verCompat" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --versionCompatible --excludeLeanRuntime

# fp32_12: precisionConstraints=obey 만 단독 (FP32에서 의미 있는지 확인)
run_cfg "fp32_12_strict" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --precisionConstraints=obey

# fp32_15: strict + verCompat + excludeLeanRuntime (FP32 풀패키지)
run_cfg "fp32_15_strict_verCompat" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --precisionConstraints=obey --versionCompatible --excludeLeanRuntime

echo ""
echo "================ ${MODEL_TAG} FP32 done ================"
column -t -s, "${SUMMARY}"
