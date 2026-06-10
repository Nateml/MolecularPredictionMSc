#!/usr/bin/env bash
# Runs evaluate_molecules.py for all four models (EI-MLP, EI-CNN, ESI-MLP, ESI-CNN)
# using 5s solvability splits, joint conformal thresholds, alpha=0.05.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/bin/activate"

EI_DATASET="$ROOT/data/EI_Dataset_MACCS_Continuous.json"
ESI_DATASET="$ROOT/data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json"

EI_SPLIT="$ROOT/data/splits/synthetic-ei/test-shuffled_solvable_5s.pkl"
ESI_SPLIT="$ROOT/data/splits/custom/test-shuffled_solvable_5s.pkl"

EI_MLP_CKPT="$ROOT/models/final/synthetic-EI/multitask_mlp.best.pt"
EI_CNN_CKPT="$ROOT/models/final/synthetic-EI/multitask_cnn_with_quantile_heads.best.pt"
ESI_MLP_CKPT="$ROOT/models/final/ESI/CONTINUOUS_ESI_BEST.pt"
ESI_CNN_CKPT="$ROOT/models/final/ESI/CONTINUOUS_ESI_BEST_CNN.pt"

run_eval() {
    local label="$1"
    local ckpt="$2"
    local split="$3"
    local dataset="$4"
    local outdir="$5"

    echo ""
    echo "=== $label ==="
    mkdir -p "$outdir"
    python "$ROOT/evaluate_molecules.py" \
        --split-pkl   "$split" \
        --full-dataset "$dataset" \
        --checkpoint  "$ckpt" \
        --thresholds-type joint \
        --alpha 0.05 \
        --use-gt-counts \
        --asp-timeout 10 \
        --output      "$outdir"
}

run_eval "EI-MLP"  "$EI_MLP_CKPT"  "$EI_SPLIT"  "$EI_DATASET"  "$ROOT/eval_results/synthetic-ei/mlp/molecule_eval"
run_eval "EI-CNN"  "$EI_CNN_CKPT"  "$EI_SPLIT"  "$EI_DATASET"  "$ROOT/eval_results/synthetic-ei/cnn/molecule_eval"
run_eval "ESI-MLP" "$ESI_MLP_CKPT" "$ESI_SPLIT" "$ESI_DATASET" "$ROOT/eval_results/esi-model/mlp/molecule_eval"
run_eval "ESI-CNN" "$ESI_CNN_CKPT" "$ESI_SPLIT" "$ESI_DATASET" "$ROOT/eval_results/esi-model/cnn/molecule_eval"

echo ""
echo "Done. Results in eval_results/{synthetic-ei,esi-model}/{mlp,cnn}/molecule_eval/"
