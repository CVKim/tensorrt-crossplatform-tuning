#!/usr/bin/env bash
# Master entry-point: launch NGC TRT container and run matrix for a given model.
# Usage:
#   bash scripts/run_in_docker.sh <MODEL_TAG> /absolute/path/to/model.onnx
#
# Notes:
# - On Windows host, prepend MSYS_NO_PATHCONV=1 if running from git-bash.
# - The matrix output goes to engines/<MODEL_TAG>/ and logs/<MODEL_TAG>/.
set -uo pipefail

MODEL_TAG="${1:?model tag required (e.g. RF-DETR, D-FINE)}"
ONNX_HOST="${2:?onnx absolute host path required}"

WORKSPACE="$(pwd)"
ONNX_BASENAME=$(basename "${ONNX_HOST}")
HOST_ONNX_DIR=$(dirname "${ONNX_HOST}")

echo "Workspace: ${WORKSPACE}"
echo "Model    : ${MODEL_TAG}"
echo "ONNX     : ${ONNX_HOST}"

mkdir -p "${WORKSPACE}/engines/${MODEL_TAG}"
mkdir -p "${WORKSPACE}/logs/${MODEL_TAG}"
mkdir -p "${WORKSPACE}/onnx/${MODEL_TAG}"

# Copy ONNX into workspace if not already present
if [[ ! -f "${WORKSPACE}/onnx/${MODEL_TAG}/${ONNX_BASENAME}" ]]; then
  cp "${ONNX_HOST}" "${WORKSPACE}/onnx/${MODEL_TAG}/"
fi

docker run --rm --gpus all \
  -v "${WORKSPACE}:/work" \
  -w /work \
  --name "${MODEL_TAG,,}_convert" \
  nvcr.io/nvidia/tensorrt:25.01-py3 \
  bash -c "
    set -e
    pip install --quiet onnx onnx-simplifier polygraphy onnxruntime 2>&1 | tail -3
    bash /work/scripts/run_matrix_generic.sh ${MODEL_TAG} /work/onnx/${MODEL_TAG}/${ONNX_BASENAME}
    bash /work/scripts/run_matrix_fp32.sh ${MODEL_TAG} /work/onnx/${MODEL_TAG}/${ONNX_BASENAME}
  "
