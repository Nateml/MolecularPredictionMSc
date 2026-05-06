"""Analyze MassBank JSON records for molecule occurrence statistics.

This script reports:
- total number of data entries
- how many molecules occur 1 time, 2 times, etc.

It supports both raw MassBank records and reduced records.

Usage:
	py "Data Processing/Browsing.py"
	py "Data Processing/Browsing.py" --input massbank_records.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


DEFAULT_INPUT = "massbank_records.json"


def load_json_records(path: str) -> list[dict[str, Any]]:
	with open(path, "r", encoding="utf-8-sig") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError("Input JSON must be a list of records.")

	return [item for item in payload if isinstance(item, dict)]


def _inchikey_from_compound_links(compound: dict[str, Any]) -> str | None:
	links = compound.get("link")
	if not isinstance(links, list):
		return None

	for link in links:
		if not isinstance(link, dict):
			continue
		db = str(link.get("database", "")).strip().upper()
		identifier = link.get("identifier")
		if db == "INCHIKEY" and isinstance(identifier, str) and identifier.strip():
			return identifier.strip()
	return None


def molecule_identifier(record: dict[str, Any]) -> str | None:
	direct_inchikey = record.get("inchikey")
	if isinstance(direct_inchikey, str) and direct_inchikey.strip():
		return f"INCHIKEY:{direct_inchikey.strip()}"

	direct_smiles = record.get("smiles")
	if isinstance(direct_smiles, str) and direct_smiles.strip():
		return f"SMILES:{direct_smiles.strip()}"

	compound = record.get("compound")
	if isinstance(compound, dict):
		nested_inchikey = (
			compound.get("inchikey")
			or compound.get("inchi_key")
			or compound.get("inchiKey")
			or _inchikey_from_compound_links(compound)
		)
		if isinstance(nested_inchikey, str) and nested_inchikey.strip():
			return f"INCHIKEY:{nested_inchikey.strip()}"

		nested_smiles = compound.get("smiles")
		if isinstance(nested_smiles, str) and nested_smiles.strip():
			return f"SMILES:{nested_smiles.strip()}"

	return None


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
	ids = [molecule_identifier(record) for record in records]
	known_ids = [mol_id for mol_id in ids if mol_id is not None]

	molecule_counts = Counter(known_ids)
	frequency_distribution = Counter(molecule_counts.values())

	return {
		"total_entries": len(records),
		"entries_with_molecule_id": len(known_ids),
		"entries_without_molecule_id": len(records) - len(known_ids),
		"unique_molecules": len(molecule_counts),
		"molecules_by_occurrence": dict(sorted(frequency_distribution.items())),
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Report total entries and molecule occurrence frequencies from MassBank JSON."
	)
	parser.add_argument(
		"--input",
		default=DEFAULT_INPUT,
		help=f"Input JSON path (default: {DEFAULT_INPUT})",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	records = load_json_records(args.input)
	result = analyze(records)

	print(f"Input file: {args.input}")
	print(f"Total data entries: {result['total_entries']}")
	print(f"Entries with molecule identifier: {result['entries_with_molecule_id']}")
	print(f"Entries without molecule identifier: {result['entries_without_molecule_id']}")
	print(f"Unique molecules: {result['unique_molecules']}")
	print("Molecule occurrence distribution:")

	distribution = result["molecules_by_occurrence"]
	if not distribution:
		print("  No molecule identifiers found.")
	else:
		for occurrence, molecule_count in distribution.items():
			print(f"  {occurrence} time(s): {molecule_count} molecule(s)")


if __name__ == "__main__":
	main()