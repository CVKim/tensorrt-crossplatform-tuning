#!/usr/bin/env bash
# Generic conversion matrix runner.
# Usage:  run_matrix_generic.sh <MODEL_TAG> <ONNX_PATH> [SIM_ONNX_PATH]
# Example:
#   run_matrix_generic.sh D-FINE /work/onnx/D-FINE/<RECIPE_DFINE>.onnx \
#                                /work/onnx/D-FINE/<RECIPE_DFINE>.sim.onnx
# Targets: Linux build -> Windows AMD64 runtime, FP16, TRT 10.8.0.43.
set -uo pipefail

MODEL_TAG="${1:?model tag required (e.g. D-FINE, RF-DETR)}"
ONNX_RAW="${2:?onnx raw path required}"
ONNX_SIM="${3:-${ONNX_RAW%.onnx}.sim.onnx}"

ENG_DIR="/work/engines/${MODEL_TAG}"
LOG_DIR="/work/logs/${MODEL_TAG}"
SUMMARY="${LOG_DIR}/_summary.csv"

mkdir -p "${ENG_DIR}" "${LOG_DIR}"
echo "name,onnx_variant,engine_bytes,engine_mib,build_seconds,exit_code,extra_flags" > "${SUMMARY}"

# Derive ONNX stem (= original recipe name) for engine file naming
ONNX_STEM=$(basename "${ONNX_RAW}" .onnx)

COMMON_CROSS="--runtimePlatform=WindowsAMD64 --fp16 --skipInference"

run_cfg () {
  local cfg="$1"; local onnx="$2"; shift 2
  local extra="$*"
  local out="${ENG_DIR}/${ONNX_STEM}__${cfg}.trt"
  local log="${LOG_DIR}/${cfg}.log"
  echo "=== [${MODEL_TAG}] [${cfg}] ==="
  local t0=$(date +%s)
  trtexec --onnx="${onnx}" --saveEngine="${out}" ${COMMON_CROSS} ${extra} > "${log}" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local sec=$((t1 - t0))
  local bytes=0
  if [[ -f "${out}" ]]; then bytes=$(stat -c %s "${out}"); fi
  local mib=$(awk -v b="${bytes}" 'BEGIN{printf "%.2f", b/1024/1024}')
  printf "%s,%s,%d,%s,%d,%d,\"%s\"\n" "${cfg}" "${onnx##*/}" "${bytes}" "${mib}" "${sec}" "${rc}" "${extra}" >> "${SUMMARY}"
  echo "    bytes=${bytes} (${mib} MiB)  build=${sec}s  rc=${rc}"
}

# 01: User's original baseline (ampere+ with directIO)
run_cfg "01_base_ampere_directIO" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --directIO --tacticSources=+CUBLAS_LT

# 02: directIO removed
run_cfg "02_base_ampere" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT

# 03: ampere+ lean metadata
run_cfg "03_ampere_lean" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none

# 04: tactic library exclusion
run_cfg "04_ampere_LTonly" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none

# 05: lean + simplified ONNX
if [[ -f "${ONNX_SIM}" ]]; then
  run_cfg "05_ampere_lean_SIM" "${ONNX_SIM}" \
    --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
    --maxAuxStreams=0 --profilingVerbosity=none
fi

# 06: stripWeights
run_cfg "06_ampere_stripped" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --stripWeights

# 07: builderOptimizationLevel=3
run_cfg "07_ampere_optLvl3" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --builderOptimizationLevel=3

# 08: builderOptimizationLevel=2
run_cfg "08_ampere_optLvl2" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --builderOptimizationLevel=2

# 09: versionCompatible (lean runtime header)  ⭐ load-time winner
run_cfg "09_ampere_verCompat" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --versionCompatible --excludeLeanRuntime

# 10: stripped + LT-only + lean + SIM combo
if [[ -f "${ONNX_SIM}" ]]; then
  run_cfg "10_ampere_combined_SIM" "${ONNX_SIM}" \
    --hardwareCompatibilityLevel=ampere+ --tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT \
    --maxAuxStreams=0 --profilingVerbosity=none --stripWeights
fi

# 11: hwcompat=none + cross-platform (single-SM engine)
run_cfg "11_none_runtimeWin" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=none --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none

# 12: precisionConstraints=obey  ⭐ disk winner
run_cfg "12_ampere_strictFp16" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --precisionConstraints=obey

# 13: noTF32 + optLvl3
run_cfg "13_ampere_noTF32_optLvl3" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --noTF32 --builderOptimizationLevel=3

# 14: lean + SIM + optLvl3
if [[ -f "${ONNX_SIM}" ]]; then
  run_cfg "14_ampere_lean_SIM_optLvl3" "${ONNX_SIM}" \
    --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
    --maxAuxStreams=0 --profilingVerbosity=none --builderOptimizationLevel=3
fi

# 15: strictFp16 + verCompat  ⭐ overall winner (no code change)
run_cfg "15_strictFp16_verCompat" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none \
  --precisionConstraints=obey --versionCompatible --excludeLeanRuntime

# 16: strictFp16 + stripWeights
run_cfg "16_strictFp16_stripped" "${ONNX_RAW}" \
  --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
  --maxAuxStreams=0 --profilingVerbosity=none --precisionConstraints=obey --stripWeights

# 17: kitchen sink (strict + stripped + verCompat + SIM)
if [[ -f "${ONNX_SIM}" ]]; then
  run_cfg "17_kitchen_sink_SIM" "${ONNX_SIM}" \
    --hardwareCompatibilityLevel=ampere+ --tacticSources=+CUBLAS_LT \
    --maxAuxStreams=0 --profilingVerbosity=none \
    --precisionConstraints=obey --stripWeights --versionCompatible --excludeLeanRuntime
fi

echo ""
echo "================ ${MODEL_TAG} SUMMARY ================"
column -t -s, "${SUMMARY}"
