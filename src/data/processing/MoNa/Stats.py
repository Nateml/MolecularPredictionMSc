"""
Inspect a filtered MoNA JSON file and report dataset statistics.

Usage:
    pip install rdkit
    python inspect_mona.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


def parse_spectrum(spectrum_str: str) -> list[tuple[float, float]]:
    """Parse 'mz:intensity ...' string into list of (mz, intensity) tuples."""
    peaks = []
    for token in spectrum_str.strip().split():
        try:
            mz, intensity = token.split(":")
            peaks.append((float(mz), float(intensity)))
        except ValueError:
            pass
    return peaks


def inspect(input_path: Path) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        print("File is empty.")
        return

    print(f"Analysing {total:,} records ...\n")

    unique_smiles      = set()
    canonical_smiles   = set()
    peak_counts        = []
    mz_ranges          = []
    mol_weights        = []
    heavy_atom_counts  = []
    element_counter    = Counter()
    invalid_mol        = 0

    for i, record in enumerate(data):
        smiles   = record.get("smiles", "")
        spectrum = record.get("spectrum", "")

        # SMILES-level stats
        unique_smiles.add(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canon = Chem.MolToSmiles(mol)
            canonical_smiles.add(canon)
            mol_weights.append(round(Descriptors.MolWt(mol), 4))
            heavy_atom_counts.append(mol.GetNumHeavyAtoms())
            # Expand implicit hydrogens so element totals reflect full formulas.
            expanded_mol = Chem.AddHs(mol)
            for atom in expanded_mol.GetAtoms():
                element_counter[atom.GetSymbol()] += 1
        else:
            invalid_mol += 1

        # Spectrum-level stats
        peaks = parse_spectrum(spectrum)
        if peaks:
            peak_counts.append(len(peaks))
            mzs = [p[0] for p in peaks]
            mz_ranges.append((min(mzs), max(mzs)))

        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,} / {total:,} processed ...", flush=True)

    def mean(lst): return sum(lst) / len(lst) if lst else 0
    def median(lst):
        s = sorted(lst)
        n = len(s)
        return (s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2) if s else 0

    print("=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)

    print(f"\n--- Records ---")
    print(f"  Total records:              {total:,}")
    print(f"  Unique SMILES (as-is):      {len(unique_smiles):,}")
    print(f"  Unique canonical SMILES:    {len(canonical_smiles):,}")
    print(f"  Duplicate canonical SMILES: {total - len(canonical_smiles):,}")

    print(f"\n--- Molecular weight (Da) ---")
    print(f"  Min:    {min(mol_weights):.2f}")
    print(f"  Max:    {max(mol_weights):.2f}")
    print(f"  Mean:   {mean(mol_weights):.2f}")
    print(f"  Median: {median(mol_weights):.2f}")

    print(f"\n--- Heavy atom count ---")
    print(f"  Min:    {min(heavy_atom_counts)}")
    print(f"  Max:    {max(heavy_atom_counts)}")
    print(f"  Mean:   {mean(heavy_atom_counts):.1f}")
    print(f"  Median: {median(heavy_atom_counts):.1f}")

    print(f"\n--- Spectrum peaks per record ---")
    print(f"  Min:    {min(peak_counts)}")
    print(f"  Max:    {max(peak_counts)}")
    print(f"  Mean:   {mean(peak_counts):.1f}")
    print(f"  Median: {median(peak_counts):.1f}")

    print(f"\n--- m/z range ---")
    all_min_mz = [r[0] for r in mz_ranges]
    all_max_mz = [r[1] for r in mz_ranges]
    print(f"  Lowest  m/z seen:  {min(all_min_mz):.2f}")
    print(f"  Highest m/z seen:  {max(all_max_mz):.2f}")
    print(f"  Mean min m/z:      {mean(all_min_mz):.2f}")
    print(f"  Mean max m/z:      {mean(all_max_mz):.2f}")

    print(f"\n--- Top 10 elements (after H expansion) ---")
    for elem, count in element_counter.most_common(10):
        print(f"  {elem:<4} {count:,}")

    print("=" * 55)


if __name__ == "__main__":
    input_path = Path("C:\\Users\\Tom\\Documents\\Planning\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs-max600Da.json")

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    inspect(input_path)