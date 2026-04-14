"""
Filter a MoNA JSON file by molecular mass, keeping only records <= threshold.

Expected record schema:
    {
      "smiles": "...",
      "spectrum": "...",
      "maccs": [...]
    }

Usage:
    pip install rdkit
    python filter_by_mass.py
    python filter_by_mass.py --max-mass 600
    python filter_by_mass.py --input "C:/path/in.json" --output "C:/path/out.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


def mol_mass_from_smiles(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def filter_by_mass(input_path: Path, output_path: Path, max_mass: float) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Input JSON must be a list of records")

    total = len(data)
    if total == 0:
        print("Input file is empty. Writing empty output.")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump([], f_out)
        return

    kept = 0
    removed_above_mass = 0
    invalid_smiles = 0

    print(f"Filtering {total:,} records with max mass <= {max_mass:.2f} Da ...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for i, record in enumerate(data, start=1):
            smiles = str(record.get("smiles", ""))
            mass = mol_mass_from_smiles(smiles)

            if mass is None:
                invalid_smiles += 1
                continue

            if mass > max_mass:
                removed_above_mass += 1
                continue

            if not first:
                f_out.write(",\n")
            json.dump(record, f_out)
            first = False
            kept += 1

            if i % 100_000 == 0:
                print(
                    f"  {i:,} / {total:,} processed | kept {kept:,} | removed {removed_above_mass:,} | invalid {invalid_smiles:,}",
                    flush=True,
                )

        f_out.write("\n]\n")

    print("\nDone.")
    print(f"  Total input records:       {total:,}")
    print(f"  Kept (<= {max_mass:.2f} Da):      {kept:,}")
    print(f"  Removed (> {max_mass:.2f} Da):    {removed_above_mass:,}")
    print(f"  Dropped invalid SMILES:    {invalid_smiles:,}")
    print(f"Output written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter MoNA records by molecular mass.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs.json"
        ),
        help="Path to source JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs-max600Da.json"
        ),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--max-mass",
        type=float,
        default=600.0,
        help="Keep only molecules with molecular mass <= this value.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    if args.max_mass <= 0:
        sys.exit("--max-mass must be > 0")

    filter_by_mass(args.input, args.output, args.max_mass)
