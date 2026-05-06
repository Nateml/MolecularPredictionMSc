"""
Assess how uniquely a merged structural feature vector identifies molecules.

This script is a companion to unique.py and expects records that include:

    {
      "smiles": "...",
      "maccs": [...],
      "structural_vector": [...],
      "feature_vector": [...]   # usually maccs + structural_vector
    }

It reports collision statistics in the same style as unique.py and compares:
  1) MACCS only
  2) MACCS + elemental composition
  3) Combined feature vector

Usage:
    pip install rdkit
    python unique_feature_vector.py
    python unique_feature_vector.py --input "C:/path/to/MoNA-with-structure.json" --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
	from rdkit import Chem
	from rdkit import RDLogger

	RDLogger.DisableLog("rdApp.*")
except ImportError:
	sys.exit("RDKit is required: pip install rdkit")


def to_int_tuple(values: object) -> tuple[int, ...] | None:
	if not isinstance(values, list):
		return None
	try:
		return tuple(int(x) for x in values)
	except (TypeError, ValueError):
		return None


def to_number_tuple(values: object) -> tuple[float, ...] | None:
	if not isinstance(values, list):
		return None
	try:
		return tuple(float(x) for x in values)
	except (TypeError, ValueError):
		return None


def element_composition_signature(mol: Chem.Mol) -> tuple[tuple[str, int], ...]:
	"""Return sorted (element, count) pairs after adding implicit hydrogens."""
	expanded = Chem.AddHs(mol)
	counts: Counter[str] = Counter()
	for atom in expanded.GetAtoms():
		counts[atom.GetSymbol()] += 1
	return tuple(sorted(counts.items()))


def pct(part: int, whole: int) -> float:
	return (100.0 * part / whole) if whole else 0.0


def report_scheme(
	title: str,
	key_name: str,
	record_count: Counter,
	to_canonical: dict,
	example_smiles: dict,
	top_n: int,
	used_records: int,
	total_unique_molecules: int,
) -> None:
	unique_keys = len(record_count)
	if unique_keys == 0:
		print(f"\n--- {title} ---")
		print("  No valid keys to report.")
		return

	collision_groups = sum(1 for _key, c in record_count.items() if c > 1)
	unique_record_groups = unique_keys - collision_groups
	records_in_unique_groups = sum(c for c in record_count.values() if c == 1)
	records_in_collision_groups = used_records - records_in_unique_groups

	perfect_id_groups = sum(1 for key in record_count if len(to_canonical[key]) == 1)
	ambiguous_groups = unique_keys - perfect_id_groups
	records_in_ambiguous_groups = sum(
		record_count[key] for key in record_count if len(to_canonical[key]) > 1
	)
	molecules_in_ambiguous_groups = sum(
		len(to_canonical[key]) for key in record_count if len(to_canonical[key]) > 1
	)

	max_records_per_key = max(record_count.values())
	max_molecules_per_key = max(len(s) for s in to_canonical.values())

	print(f"\n--- {title} (record view) ---")
	print(f"  Unique {key_name} keys:                        {unique_keys:,}")
	print(f"  Keys seen once (no record collision):          {unique_record_groups:,}")
	print(f"  Keys with record collisions:                   {collision_groups:,}")
	print(f"  Records in unique-key groups:                  {records_in_unique_groups:,} ({pct(records_in_unique_groups, used_records):.2f}%)")
	print(f"  Records in collided-key groups:                {records_in_collision_groups:,} ({pct(records_in_collision_groups, used_records):.2f}%)")
	print(f"  Max records sharing one key:                   {max_records_per_key:,}")

	print(f"\n--- {title} (molecule view) ---")
	print(f"  Keys mapping to exactly 1 molecule:            {perfect_id_groups:,} ({pct(perfect_id_groups, unique_keys):.2f}% of keys)")
	print(f"  Keys mapping to >1 molecule:                   {ambiguous_groups:,} ({pct(ambiguous_groups, unique_keys):.2f}% of keys)")
	print(f"  Records in ambiguous key groups:               {records_in_ambiguous_groups:,} ({pct(records_in_ambiguous_groups, used_records):.2f}% of records)")
	print(f"  Molecules in ambiguous key groups:             {molecules_in_ambiguous_groups:,} ({pct(molecules_in_ambiguous_groups, total_unique_molecules):.2f}% of molecules)")
	print(f"  Max molecules sharing one key:                 {max_molecules_per_key:,}")

	print(f"\n--- Top {top_n} ambiguous {key_name} keys (by molecule count) ---")
	ambiguous_sorted = sorted(
		(
			(
				key,
				len(to_canonical[key]),
				record_count[key],
				example_smiles[key],
			)
			for key in record_count
			if len(to_canonical[key]) > 1
		),
		key=lambda x: (x[1], x[2]),
		reverse=True,
	)

	if not ambiguous_sorted:
		print("  None. Every observed key maps to a single canonical molecule.")
	else:
		for idx, (_key, mol_count, rec_count, example) in enumerate(ambiguous_sorted[:top_n], start=1):
			print(
				f"  {idx:>2}. molecules={mol_count:,}, records={rec_count:,}, example_smiles={example}"
			)


def analyse(input_path: Path, top_n: int, vector_field: str) -> None:
	print(f"Loading {input_path} ...")
	with open(input_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list) or not data:
		print("Input JSON is empty or not a list of records.")
		return

	fp_record_count: Counter[tuple[int, ...]] = Counter()
	fp_to_canonical: dict[tuple[int, ...], set[str]] = defaultdict(set)
	fp_example_smiles: dict[tuple[int, ...], str] = {}

	combo_record_count: Counter[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]] = Counter()
	combo_to_canonical: dict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], set[str]] = defaultdict(set)
	combo_example_smiles: dict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], str] = {}

	vec_record_count: Counter[tuple[float, ...]] = Counter()
	vec_to_canonical: dict[tuple[float, ...], set[str]] = defaultdict(set)
	vec_example_smiles: dict[tuple[float, ...], str] = {}

	missing_maccs = 0
	invalid_maccs = 0
	missing_vector = 0
	invalid_vector = 0
	missing_vector_but_rebuilt = 0
	invalid_smiles = 0

	for record in data:
		smiles = str(record.get("smiles", ""))
		maccs = record.get("maccs")

		if maccs is None:
			missing_maccs += 1
			continue

		fp = to_int_tuple(maccs)
		if fp is None:
			invalid_maccs += 1
			continue

		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			invalid_smiles += 1
			continue

		canon = Chem.MolToSmiles(mol)
		composition = element_composition_signature(mol)
		combo_key = (fp, composition)

		vector_payload = record.get(vector_field)
		vector_key = to_number_tuple(vector_payload)

		# Fallback: rebuild combined vector if split parts are present.
		if vector_key is None:
			structural = record.get("structural_vector")
			structural_tuple = to_number_tuple(structural)
			if structural_tuple is not None:
				vector_key = tuple(float(x) for x in fp) + structural_tuple
				missing_vector_but_rebuilt += 1
			elif vector_payload is None:
				missing_vector += 1
			else:
				invalid_vector += 1

		fp_record_count[fp] += 1
		fp_to_canonical[fp].add(canon)
		fp_example_smiles.setdefault(fp, smiles)

		combo_record_count[combo_key] += 1
		combo_to_canonical[combo_key].add(canon)
		combo_example_smiles.setdefault(combo_key, smiles)

		if vector_key is not None:
			vec_record_count[vector_key] += 1
			vec_to_canonical[vector_key].add(canon)
			vec_example_smiles.setdefault(vector_key, smiles)

	total_records = len(data)
	used_records = sum(fp_record_count.values())
	if used_records == 0:
		print("No valid records with usable maccs + smiles were found.")
		return

	all_canonical = set()
	for canon_set in fp_to_canonical.values():
		all_canonical.update(canon_set)
	total_unique_molecules = len(all_canonical)

	vector_used_records = sum(vec_record_count.values())

	print("=" * 78)
	print("DISTINGUISHABILITY REPORT (FEATURE VECTOR)")
	print("=" * 78)

	print("\n--- Input records ---")
	print(f"  Total records in file:                        {total_records:,}")
	print(f"  Records used for MACCS analysis:              {used_records:,}")
	print(f"  Records used for feature-vector analysis:     {vector_used_records:,}")
	print(f"  Missing maccs field:                          {missing_maccs:,}")
	print(f"  Invalid maccs payload:                        {invalid_maccs:,}")
	print(f"  Missing '{vector_field}' field:                {missing_vector:,}")
	print(f"  Missing '{vector_field}' but rebuilt:          {missing_vector_but_rebuilt:,}")
	print(f"  Invalid '{vector_field}' payload:              {invalid_vector:,}")
	print(f"  Invalid smiles (for canonicalization):        {invalid_smiles:,}")
	print(f"  Unique canonical molecules analysed:          {total_unique_molecules:,}")

	report_scheme(
		title="MACCS only",
		key_name="MACCS",
		record_count=fp_record_count,
		to_canonical=fp_to_canonical,
		example_smiles=fp_example_smiles,
		top_n=top_n,
		used_records=used_records,
		total_unique_molecules=total_unique_molecules,
	)

	report_scheme(
		title="MACCS + elemental composition (H/C/N/O/etc., with implicit H)",
		key_name="MACCS+composition",
		record_count=combo_record_count,
		to_canonical=combo_to_canonical,
		example_smiles=combo_example_smiles,
		top_n=top_n,
		used_records=used_records,
		total_unique_molecules=total_unique_molecules,
	)

	report_scheme(
		title=f"Combined vector ('{vector_field}' or rebuilt maccs+structural_vector)",
		key_name="feature_vector",
		record_count=vec_record_count,
		to_canonical=vec_to_canonical,
		example_smiles=vec_example_smiles,
		top_n=top_n,
		used_records=vector_used_records,
		total_unique_molecules=total_unique_molecules,
	)

	print("=" * 78)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Measure uniqueness/collision statistics for MACCS and merged structural feature vectors."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path(
			"C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs-and-structure.json"
		),
		help="Path to JSON file containing smiles, maccs, and feature_vector fields.",
	)
	parser.add_argument(
		"--vector-field",
		type=str,
		default="feature_vector",
		help="Field name to use for the merged vector.",
	)
	parser.add_argument(
		"--top",
		type=int,
		default=10,
		help="How many largest ambiguous groups to print.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if not args.input.exists():
		sys.exit(f"Input file not found: {args.input}")
	if args.top < 1:
		sys.exit("--top must be >= 1")
	analyse(args.input, args.top, args.vector_field)