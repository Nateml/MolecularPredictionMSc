"""Download all MassBank records and write them to a JSON file.

Usage:
	python "Data Processing/MassBank.py"
	python "Data Processing/MassBank.py" --output massbank_records.json --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from urllib import error, parse, request


DEFAULT_API_BASE = "https://massbank.eu/MassBank-api"
DEFAULT_OUTPUT_FILE = "massbank_records.json"


def _build_url(api_base: str, endpoint: str, params: dict[str, Any] | None = None) -> str:
	base = api_base.rstrip("/")
	path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
	url = f"{base}{path}"
	if params:
		query = parse.urlencode(params, doseq=True)
		url = f"{url}?{query}"
	return url


def _fetch_json(
	api_base: str,
	endpoint: str,
	params: dict[str, Any] | None = None,
	*,
	timeout: int = 120,
	retries: int = 5,
) -> Any:
	url = _build_url(api_base, endpoint, params)
	last_error: Exception | None = None

	for attempt in range(1, retries + 1):
		try:
			with request.urlopen(url, timeout=timeout) as response:
				payload = response.read().decode("utf-8")
				return json.loads(payload)
		except error.HTTPError as exc:
			last_error = exc
			# Retry only transient API failures and rate limits.
			if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
				break
			wait_seconds = min(30, 2 ** (attempt - 1))
			print(f"Retrying {url} after HTTP {exc.code} ({wait_seconds}s)...")
			time.sleep(wait_seconds)
		except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
			last_error = exc
			if attempt == retries:
				break
			wait_seconds = min(30, 2 ** (attempt - 1))
			print(f"Retrying {url} after {type(exc).__name__} ({wait_seconds}s)...")
			time.sleep(wait_seconds)

	raise RuntimeError(f"Failed to fetch {url}") from last_error


def _extract_contributor_codes(browse_payload: Any) -> list[str]:
	if not isinstance(browse_payload, dict):
		return []

	raw_items = browse_payload.get("contributor") or browse_payload.get("contributors") or []
	codes: set[str] = set()

	for item in raw_items:
		if isinstance(item, str) and item.strip():
			codes.add(item.strip())
			continue

		if isinstance(item, dict):
			for key in ("short", "short_name", "code", "contributor", "id", "name"):
				value = item.get(key)
				if isinstance(value, str) and value.strip():
					codes.add(value.strip())
					break

	return sorted(codes)


def fetch_total_count(api_base: str) -> int | None:
	raw = _fetch_json(api_base, "/records/count")
	if isinstance(raw, int):
		return raw
	if isinstance(raw, str) and raw.strip().isdigit():
		return int(raw.strip())
	return None


def fetch_all_records(api_base: str) -> list[dict[str, Any]]:
	print("Fetching browse options...")
	browse = _fetch_json(api_base, "/filter/browse")
	contributor_codes = _extract_contributor_codes(browse)

	records_by_accession: dict[str, dict[str, Any]] = {}

	if contributor_codes:
		print(f"Found {len(contributor_codes)} contributors. Downloading contributor by contributor...")
		for index, code in enumerate(contributor_codes, start=1):
			try:
				batch = _fetch_json(api_base, "/records", {"contributor": code})
			except Exception as exc:  # noqa: BLE001
				print(f"Skipping contributor {code} due to error: {exc}")
				continue

			if not isinstance(batch, list):
				print(f"Skipping contributor {code}: unexpected response type {type(batch).__name__}")
				continue

			for record in batch:
				if not isinstance(record, dict):
					continue
				accession = record.get("accession")
				if isinstance(accession, str) and accession.strip():
					records_by_accession[accession] = record

			print(
				f"[{index}/{len(contributor_codes)}] {code}: "
				f"{len(batch)} records (unique total: {len(records_by_accession)})"
			)

	# Fallback for API variants that do not provide contributor browse options.
	if not records_by_accession:
		print("Contributor-based download unavailable. Fetching from /records directly...")
		payload = _fetch_json(api_base, "/records")
		if not isinstance(payload, list):
			raise RuntimeError("Unexpected /records response. Expected a JSON array.")
		for record in payload:
			if not isinstance(record, dict):
				continue
			accession = record.get("accession")
			if isinstance(accession, str) and accession.strip():
				records_by_accession[accession] = record

	all_records = list(records_by_accession.values())
	all_records.sort(key=lambda item: str(item.get("accession", "")))
	return all_records


def write_json(path: str, records: list[dict[str, Any]], pretty: bool = False) -> None:
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
		description="Download all MassBank records and save them to a JSON file."
	)
	parser.add_argument(
		"--output",
		default=DEFAULT_OUTPUT_FILE,
		help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})",
	)
	parser.add_argument(
		"--api-base",
		default=DEFAULT_API_BASE,
		help=f"MassBank API base URL (default: {DEFAULT_API_BASE})",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Write indented JSON for readability.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print(f"MassBank API: {args.api_base}")
	expected_count = fetch_total_count(args.api_base)
	if expected_count is not None:
		print(f"Reported record count: {expected_count}")

	records = fetch_all_records(args.api_base)
	write_json(args.output, records, pretty=args.pretty)

	print(f"Saved {len(records)} unique records to: {args.output}")
	if expected_count is not None and len(records) != expected_count:
		print(
			"Warning: downloaded record count does not match /records/count. "
			f"expected={expected_count}, downloaded={len(records)}"
		)


if __name__ == "__main__":
	main()