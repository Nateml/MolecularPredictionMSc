"""
Filter MoNA JSON export to only records containing RDKit-parseable SMILES,
and strip each record down to just SMILES + mass spectral data.

Produces three output files:
  - *_all.json     : records with any valid SMILES (computed or not)
  - *_curated.json : records where SMILES was explicitly provided (computed: False)
  - *_model.json   : curated SMILES + strict spectral filters for MACCS prediction:
                       - MS2 only
                       - Positive ionisation mode only
                       - [M+H]+ adduct only
                       - ESI instrument (excludes EI/GC-MS)
                       - Minimum 5 peaks

Uses streaming JSON parsing (ijson) to handle files larger than available RAM.

Usage:
    pip install ijson rdkit
    python filter_mona.py
"""

import sys
import json
from pathlib import Path

try:
    import ijson
except ImportError:
    sys.exit("ijson is required: pip install ijson")

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


MIN_PEAKS = 5


def is_valid_smiles(smiles: str) -> bool:
    if not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def extract_smiles(record: dict) -> tuple[str | None, str | None]:
    """Returns (curated_smiles, computed_smiles) from compound[0].metaData."""
    curated = None
    computed = None
    try:
        compound = record.get("compound", [])
        if not compound:
            return None, None
        for meta in compound[0].get("metaData", []):
            if meta.get("name", "").upper() == "SMILES":
                value = meta.get("value")
                if meta.get("computed", False):
                    computed = value
                else:
                    curated = value
    except (AttributeError, TypeError, IndexError):
        pass
    return curated, computed


def get_metadata_value(meta_list: list, name: str) -> str | None:
    """Case-insensitive lookup of a value from a MoNA metaData list."""
    name_upper = name.upper()
    for meta in meta_list:
        if meta.get("name", "").upper() == name_upper:
            return str(meta.get("value", "")).strip()
    return None


def count_peaks(spectrum: str) -> int:
    return len(spectrum.strip().split())


def passes_model_filters(record: dict, spectrum: str) -> tuple[bool, str]:
    """
    Returns (passes, reason_if_failed).
    Checks MS2, positive mode, [M+H]+, ESI instrument, and minimum peak count.
    """
    meta_list = record.get("metaData", [])

    ms_level = get_metadata_value(meta_list, "ms level")
    if not ms_level or "ms2" not in ms_level.lower():
        return False, "not MS2"

    ion_mode = get_metadata_value(meta_list, "ionization mode")
    if not ion_mode or "positive" not in ion_mode.lower():
        return False, "not positive mode"

    precursor_type = get_metadata_value(meta_list, "precursor type")
    if not precursor_type or "[m+h]+" not in precursor_type.lower():
        return False, "not [M+H]+"

    instrument_type = get_metadata_value(meta_list, "instrument type")
    if not instrument_type or not any(t in instrument_type.upper() for t in ("ESI", "QTOF", "ORBITRAP", "Q-TOF")):
        return False, "not ESI instrument"

    if count_peaks(spectrum) < MIN_PEAKS:
        return False, f"fewer than {MIN_PEAKS} peaks"

    return True, ""


def filter_mona_json(
    input_path: Path,
    output_all: Path,
    output_curated: Path,
    output_model: Path,
) -> None:
    total       = 0
    kept_all    = 0
    kept_cur    = 0
    kept_model  = 0
    no_smiles   = 0
    no_spectrum = 0
    model_drop  = {}   # reason -> count

    print(f"Streaming {input_path} ...")

    with open(input_path, "rb") as f_in, \
         open(output_all,     "w", encoding="utf-8") as f_all, \
         open(output_curated, "w", encoding="utf-8") as f_cur, \
         open(output_model,   "w", encoding="utf-8") as f_mod:

        f_all.write("[\n")
        f_cur.write("[\n")
        f_mod.write("[\n")
        first_all = first_cur = first_mod = True

        for record in ijson.items(f_in, "item"):
            total += 1

            curated_smiles, computed_smiles = extract_smiles(record)
            smiles_all = curated_smiles or computed_smiles

            if not smiles_all or not is_valid_smiles(smiles_all):
                no_smiles += 1
                continue

            spectrum = record.get("spectrum")
            if not spectrum:
                no_spectrum += 1
                continue
            spectrum = spectrum.strip()

            # --- all ---
            slim_all = {"smiles": smiles_all.strip(), "spectrum": spectrum}
            if not first_all:
                f_all.write(",\n")
            json.dump(slim_all, f_all)
            first_all = False
            kept_all += 1

            # --- curated ---
            if curated_smiles and is_valid_smiles(curated_smiles):
                slim_cur = {"smiles": curated_smiles.strip(), "spectrum": spectrum}
                if not first_cur:
                    f_cur.write(",\n")
                json.dump(slim_cur, f_cur)
                first_cur = False
                kept_cur += 1

                # --- model (must also be curated) ---
                passes, reason = passes_model_filters(record, spectrum)
                if passes:
                    slim_mod = {"smiles": curated_smiles.strip(), "spectrum": spectrum}
                    if not first_mod:
                        f_mod.write(",\n")
                    json.dump(slim_mod, f_mod)
                    first_mod = False
                    kept_model += 1
                else:
                    model_drop[reason] = model_drop.get(reason, 0) + 1

            if total % 10_000 == 0:
                print(
                    f"  Processed {total:,} | "
                    f"all: {kept_all:,} | "
                    f"curated: {kept_cur:,} | "
                    f"model: {kept_model:,} ...",
                    flush=True,
                )

        f_all.write("\n]")
        f_cur.write("\n]")
        f_mod.write("\n]")

    print(f"\nFinished processing {total:,} records.")
    print(f"  Dropped — no valid SMILES: {no_smiles:,}")
    print(f"  Dropped — no spectrum:     {no_spectrum:,}")
    print(f"  Kept (all):                {kept_all:,} / {total:,}  →  {output_all}")
    print(f"  Kept (curated):            {kept_cur:,} / {total:,}  →  {output_curated}")
    print(f"  Kept (model-ready):        {kept_model:,} / {total:,}  →  {output_model}")
    if model_drop:
        print(f"\n  Model filter drop reasons:")
        for reason, count in sorted(model_drop.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30} {count:,}")


if __name__ == "__main__":
    input_path     = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-export-All_Spectra.json")
    output_all     = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-All_Spectra.json")
    output_curated = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-All_Spectra.json")
    output_model   = Path("C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-model-All_Spectra.json")
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    filter_mona_json(input_path, output_all, output_curated, output_model)