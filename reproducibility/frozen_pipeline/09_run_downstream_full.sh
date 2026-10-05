#!/usr/bin/env bash
set -euo pipefail
ROOT=${RIMBIND_ROOT:?Set RIMBIND_ROOT to the prepared run directory}
PIPE=$(cd "$(dirname "$0")" && pwd)
PY=${RIMBIND_PYTHON:-python}
ESM_PY=${RIMBIND_ESM3_PYTHON:-python}
SAPROT_MODEL=${SAPROT_MODEL_DIR:?Set SAPROT_MODEL_DIR to SaProt_650M_AF2}
MEMORY_PDB_ROOT=${RIMBIND_MEMORY_PDB_ROOT:?Set RIMBIND_MEMORY_PDB_ROOT to the directory containing dna/ and rna/}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

"$PY" "$PIPE/build_3di.py" --root "$ROOT"

# SaProt and ESM3 can run concurrently because they use separate GPUs.
(
  CUDA_VISIBLE_DEVICES=0 "$PY" "$PIPE/generate_embeddings.py" --root "$ROOT" --kind saprot --saprot-model "$SAPROT_MODEL" --device cuda:0 --scope full --domain dna
  CUDA_VISIBLE_DEVICES=0 "$PY" "$PIPE/generate_embeddings.py" --root "$ROOT" --kind saprot --saprot-model "$SAPROT_MODEL" --device cuda:0 --scope full --domain rna
) &
SAPROT_PID=$!
(
  CUDA_VISIBLE_DEVICES=1 "$ESM_PY" "$PIPE/generate_embeddings.py" --root "$ROOT" --kind esm3 --device cuda:0 --scope full --domain dna
  CUDA_VISIBLE_DEVICES=1 "$ESM_PY" "$PIPE/generate_embeddings.py" --root "$ROOT" --kind esm3 --device cuda:0 --scope full --domain rna
) &
ESM3_PID=$!
wait "$SAPROT_PID"
wait "$ESM3_PID"

CUDA_VISIBLE_DEVICES=0 "$PY" "$PIPE/run_c0_prediction.py" --root "$ROOT" --domain dna --device cuda:0 --scope full &
C0_DNA_PID=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIPE/run_c0_prediction.py" --root "$ROOT" --domain rna --device cuda:0 --scope full &
C0_RNA_PID=$!
wait "$C0_DNA_PID"
wait "$C0_RNA_PID"

"$PY" "$PIPE/run_v1_memory_prediction.py" --root "$ROOT" --memory-pdb-root "$MEMORY_PDB_ROOT" --domain dna --scope full
"$PY" "$PIPE/run_v1_memory_prediction.py" --root "$ROOT" --memory-pdb-root "$MEMORY_PDB_ROOT" --domain rna --scope full
"$PY" "$ROOT/code/evaluate_predicted_monomer.py" --root "$ROOT" --scope full
