#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate

ESI_DATASET="data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json"
EI_DATASET="data/EI_Dataset_MACCS_Continuous.json"
ESI_CAL_PKL="data/splits/custom/calibration-shuffled.pkl"
ESI_TEST_PKL="data/splits/custom/test-shuffled.pkl"
EI_CAL_PKL="data/splits/synthetic-ei/calibration-shuffled.pkl"
EI_TEST_PKL="data/splits/synthetic-ei/test-shuffled.pkl"
ALPHA=0.05

echo "=== Evaluating ESI MLP ==="
mkdir -p eval_results/esi-model/mlp
python evaluate.py \
    --checkpoint models/final/ESI/CONTINUOUS_ESI_BEST.pt \
    --split-pkl "$ESI_TEST_PKL" \
    --full-dataset "$ESI_DATASET" \
    --output-folder eval_results/esi-model/mlp \
    --conformal \
    --alpha "$ALPHA" \
    --thresholds-type joint

echo "=== Evaluating ESI CNN ==="
mkdir -p eval_results/esi-model/cnn
python evaluate.py \
    --checkpoint models/final/ESI/CONTINUOUS_ESI_BEST_CNN.pt \
    --split-pkl "$ESI_TEST_PKL" \
    --full-dataset "$ESI_DATASET" \
    --output-folder eval_results/esi-model/cnn \
    --conformal \
    --alpha "$ALPHA" \
    --thresholds-type joint

echo "=== Evaluating Synthetic EI MLP ==="
mkdir -p eval_results/synthetic-ei/mlp
python evaluate.py \
    --checkpoint models/final/synthetic-EI/multitask_mlp.best.pt \
    --split-pkl "$EI_TEST_PKL" \
    --full-dataset "$EI_DATASET" \
    --output-folder eval_results/synthetic-ei/mlp \
    --conformal \
    --alpha "$ALPHA" \
    --thresholds-type joint

echo "=== Evaluating Synthetic EI CNN ==="
mkdir -p eval_results/synthetic-ei/mlp
python evaluate.py \
    --checkpoint models/final/synthetic-EI/multitask_cnn_with_quantile_heads.best.pt \
    --split-pkl "$EI_TEST_PKL" \
    --full-dataset "$EI_DATASET" \
    --output-folder eval_results/synthetic-ei/mlp \
    --conformal \
    --alpha "$ALPHA" \
    --thresholds-type joint

echo "=== Done ==="
