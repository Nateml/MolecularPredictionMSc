"""Reduce MassBank records to only SMILES, InChIKey, and mass spectral data.

Usage:
	python "Data Processing/Browsing.py"
	python "Data Processing/Browsing.py" --input massbank_records.json --output massbank_filtered.json --pretty
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


DEFAULT_INPUT = "massbank_records.json"
DEFAULT_OUTPUT = "massbank_filtered.json"


def load_records(path: str) -> list[dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError("Input JSON must be a list of MassBank records.")

	return [record for record in payload if isinstance(record, dict)]


def extract_inchikey(record: dict[str, Any]) -> str | None:
	compound = record.get("compound")
	if not isinstance(compound, dict):
		return None

	direct_keys = ("inchikey", "inchi_key", "inchiKey")
	for key in direct_keys:
		value = compound.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()

	links = compound.get("link")
	if isinstance(links, list):
		for item in links:
			if not isinstance(item, dict):
				continue
			database = str(item.get("database", "")).strip().upper()
			identifier = item.get("identifier")
			if database == "INCHIKEY" and isinstance(identifier, str) and identifier.strip():
				return identifier.strip()

	return None


def simplify_record(record: dict[str, Any]) -> dict[str, Any]:
	compound = record.get("compound")
	smiles: str | None = None

	if isinstance(compound, dict):
		raw_smiles = compound.get("smiles")
		if isinstance(raw_smiles, str) and raw_smiles.strip():
			smiles = raw_smiles.strip()

	simplified: dict[str, Any] = {
		"smiles": smiles,
		"inchikey": extract_inchikey(record),
		"mass_spectral_data": record.get("peak"),
	}
	return simplified


def transform_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return [simplify_record(record) for record in records]


def save_records(path: str, records: list[dict[str, Any]], pretty: bool = False) -> None:
	directory = os.path.dirname(os.path.abspath(path))
	if directory:
		os.makedirs(directory, exist_ok=True)

	with open(path, "w", encoding="utf-8") as handle:
		if pretty:
			json.dump(records, handle, ensure_ascii=False, indent=2)
		else:
			json.dump(records, handle, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Keep only smiles, inchikey, and mass_spectral_data from MassBank JSON records."
	)
	parser.add_argument(
		"--input",
		default=DEFAULT_INPUT,
		help=f"Path to source JSON file (default: {DEFAULT_INPUT})",
	)
	parser.add_argument(
		"--output",
		default=DEFAULT_OUTPUT,
		help=f"Path to filtered JSON file (default: {DEFAULT_OUTPUT})",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Write indented JSON for readability.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	records = load_records(args.input)
	filtered = transform_records(records)
	save_records(args.output, filtered, pretty=args.pretty)

	print(f"Loaded {len(records)} records from: {args.input}")
	print(f"Saved {len(filtered)} filtered records to: {args.output}")


if __name__ == "__main__":
	main()
