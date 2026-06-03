#!/usr/bin/env bash
# Isolate the effect of --hardwareCompatibilityLevel=ampere+ in the
# dispatcher-aware production cmd.
#
# Builds two engines per model:
#   ampere_on  : current production cmd (_123 + ampere+)
#   ampere_off : same cmd with --hardwareCompatibilityLevel=ampere+ removed
#
# Compare resulting engine size, load time, and inference bbox coordinates to
# decide whether the dispatcher-aware cmd is safe for your environment. See
# docs/13-dispatcher-aware-cmd.md.
#
# Usage:
#   bash scripts/run_ampere_isolation.sh <MODEL_TAG> /work/onnx/<MODEL>/<RECIPE>.onnx
set -uo pipefail

MODEL_TAG="${1:?model tag required (e.g. RF-DETR, D-FINE)}"
ONNX_RAW="${2:?onnx absolute path required}"

ENG_DIR="/work/engines/${MODEL_TAG}"
LOG_DIR="/work/logs/${MODEL_TAG}"
SUMMARY="${LOG_DIR}/_ampere_isolation_summary.csv"
ONNX_STEM=$(basename "${ONNX_RAW}" .onnx)
mkdir -p "${ENG_DIR}" "${LOG_DIR}"
echo "name,onnx_variant,engine_bytes,engine_mib,build_seconds,exit_code,extra_flags" > "${SUMMARY}"

COMMON_CROSS="--runtimePlatform=WindowsAMD64 --fp16 --skipInference"
COMMON_BASE="--tacticSources=+CUBLAS_LT --directIO --precisionConstraints=obey --versionCompatible --excludeLeanRuntime"

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

# Variant A: production _123 with ampere+ (current prod)
run_cfg "ampere_on" ${COMMON_BASE} --hardwareCompatibilityLevel=ampere+

# Variant B: production _123 without ampere+ (dispatcher-aware)
run_cfg "ampere_off" ${COMMON_BASE}

echo ""
echo "================ ${MODEL_TAG} AMPERE ISOLATION DONE ================"
column -t -s, "${SUMMARY}"
