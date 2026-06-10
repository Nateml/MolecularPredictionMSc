"""Compare molecular property distributions between two index split files."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split_a", type=Path, help="First split .pkl (indices)")
    p.add_argument("split_b", type=Path, help="Second split .pkl (indices)")
    p.add_argument("--dataset", "-d", type=Path, required=True,
                   help="Full dataset .json file")
    p.add_argument("--schema", "-s", type=Path, default=None,
                   help="Schema .json (default: resolved from config)")
    p.add_argument("--sample", type=int, default=2000,
                   help="Max records to sample from each split (default: 2000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-a", default=None, help="Label for split A")
    p.add_argument("--label-b", default=None, help="Label for split B")
    return p.parse_args()


FEATURES = [
    "molecular_weight_da",
    "heavy_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "double_bond_count",
    "degree_of_unsaturation",
]


def load_features(ds, indices, sample_n, seed):
    rng = np.random.default_rng(seed)
    arr = np.asarray(indices)
    idxs = arr if len(arr) <= sample_n else rng.choice(arr, sample_n, replace=False)
    out = {f: [] for f in FEATURES}
    for i in idxs:
        rec = ds._load_record(i)
        sf = rec.get("structural_features", {})
        for f in FEATURES:
            out[f].append(sf.get(f, float("nan")))
    return {f: np.array(v) for f, v in out.items()}


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from config import config
    from data.datasets import dataset_from_schema

    schema_path = args.schema or config.data.resolved_schema_path()
    ds = dataset_from_schema(args.dataset, schema_path,
                             spectrum_len=config.model.spectrum_length)

    with open(args.split_a, "rb") as f:
        idx_a = pickle.load(f)
    with open(args.split_b, "rb") as f:
        idx_b = pickle.load(f)

    label_a = args.label_a or args.split_a.stem
    label_b = args.label_b or args.split_b.stem

    print(f"Dataset : {args.dataset.name}  ({len(ds)} records)")
    print(f"Split A : {label_a}  (n={len(idx_a)}, sampling up to {args.sample})")
    print(f"Split B : {label_b}  (n={len(idx_b)}, sampling up to {args.sample})")

    feats_a = load_features(ds, idx_a, args.sample, args.seed)
    feats_b = load_features(ds, idx_b, args.sample, args.seed + 1)

    col = 22
    header = (f"{'Feature':<{col}} | {label_a[:18]:>18} mean±std   "
              f"| {label_b[:18]:>18} mean±std   | KS stat |   p-value")
    print()
    print(header)
    print("-" * len(header))

    any_significant = False
    for feat in FEATURES:
        va = feats_a[feat]
        vb = feats_b[feat]
        va = va[~np.isnan(va)]
        vb = vb[~np.isnan(vb)]
        if len(va) == 0 or len(vb) == 0:
            print(f"{feat:<{col}} | (no data)")
            continue
        ks, p = stats.ks_2samp(va, vb)
        flag = " ***" if p < 0.001 else (" *" if p < 0.05 else "")
        p_str = "< 5e-324 (underflow)" if p == 0.0 else f"{p:.3e}"
        print(f"{feat:<{col}} | {np.mean(va):8.2f} ± {np.std(va):6.2f}        "
              f"| {np.mean(vb):8.2f} ± {np.std(vb):6.2f}        "
              f"| {ks:.4f}  | {p_str}{flag}")
        if p < 0.05:
            any_significant = True

    print()
    if any_significant:
        print("WARNING: Significant distributional differences detected (*** p<0.001, * p<0.05).")
        print("Exchangeability assumption for conformal prediction may be violated.")
    else:
        print("No significant distributional differences detected.")


if __name__ == "__main__":
    main()
