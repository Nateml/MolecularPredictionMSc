#!/usr/bin/env bash
# Calibrate and evaluate the synthetic-EI CNN at 11 alpha values.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/bin/activate"

CKPT="$ROOT/models/final/synthetic-EI/multitask_cnn_with_quantile_heads.best.pt"
CAL_PKL="$ROOT/data/splits/synthetic-ei/calibration-shuffled.pkl"
TEST_SPLIT="$ROOT/data/splits/synthetic-ei/test-shuffled_solvable_5s.pkl"
DATASET="$ROOT/data/EI_Dataset_MACCS_Continuous.json"
OUTBASE="$ROOT/eval_results/synthetic-ei/cnn/molecule_eval/alpha-experiment"

ALPHAS=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)

for ALPHA in "${ALPHAS[@]}"; do
    ALPHA_TAG="alpha_${ALPHA}"
    OUTDIR="$OUTBASE/$ALPHA_TAG"
    mkdir -p "$OUTDIR"

    echo ""
    echo "=== Calibrating alpha=${ALPHA} ==="
    python "$ROOT/calibrate.py" \
        --checkpoint    "$CKPT" \
        --calibration-pkl "$CAL_PKL" \
        --full-dataset  "$DATASET" \
        --alpha         "$ALPHA" \
        --joint

    echo ""
    echo "=== Evaluating alpha=${ALPHA} → $OUTDIR ==="
    python "$ROOT/evaluate_molecules.py" \
        --split-pkl     "$TEST_SPLIT" \
        --full-dataset  "$DATASET" \
        --checkpoint    "$CKPT" \
        --thresholds-type joint \
        --alpha         "$ALPHA" \
        --use-gt-counts \
        --asp-timeout   10 \
        --output        "$OUTDIR"
done

echo ""
echo "Done. Results in $OUTBASE/"
