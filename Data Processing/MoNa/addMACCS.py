"""
Add MACCS keys fingerprints to a filtered MoNA JSON file.

Each output record will have the form:
    {
        "smiles":   "...",
        "spectrum": "109.07:999 138.07:80 ...",
        "maccs":    [0, 1, 0, 1, ...]   (167-bit list of ints)
    }

Uses streaming output to avoid loading the entire file into memory at once.

Usage:
    pip install rdkit
    python add_maccs.py
"""

import sys
import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


def smiles_to_maccs(smiles: str) -> list[int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(fp)


def add_maccs_keys(input_path: Path, output_path: Path) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total   = len(data)
    kept    = 0
    dropped = 0

    print(f"Processing {total:,} records ...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for i, record in enumerate(data):
            smiles = record.get("smiles", "")
            maccs  = smiles_to_maccs(smiles)

            if maccs is None:
                dropped += 1
                continue

            out = {
                "smiles":   smiles,
                "spectrum": record.get("spectrum", ""),
                "maccs":    maccs,
            }

            if not first:
                f_out.write(",\n")
            json.dump(out, f_out)
            first = False
            kept += 1

            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} / {total:,} processed, kept {kept:,} ...", flush=True)

        f_out.write("\n]")

    print(f"\nFinished.")
    print(f"  Kept:    {kept:,} / {total:,}")
    print(f"  Dropped: {dropped:,}  (SMILES no longer parseable)")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    input_path  = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-All_Spectra.json")
    output_path = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs.json")

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    add_maccs_keys(input_path, output_path)