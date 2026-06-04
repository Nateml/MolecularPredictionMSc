#!/usr/bin/env python3
"""
Convert IN_SILICO_LIBRARY.db SQLite database to JSON format
matching MoNA.py output structure: [{"smiles": "...", "spectrum": "..."}, ...]

The spectrum field is constructed from MZS and INTENSITYS columns as space-separated pairs.

Usage:
    python scripts/convert_db_to_json.py <input_db> <output_json>
    python scripts/convert_db_to_json.py data/IN_SILICO_LIBRARY.db output/IN_SILICO_LIBRARY.json
"""

import sys
import json
import sqlite3
import pickle
import ast
from pathlib import Path


def deserialize_blob(blob_data):
    """Deserialize a blob to a Python list.
    
    Tries multiple methods:
    1. Pickle deserialization (if bytes)
    2. ast.literal_eval (if string representation of list)
    3. Returns empty list on failure
    """
    if blob_data is None:
        return []
    
    # Try pickle first if it's bytes
    if isinstance(blob_data, bytes):
        try:
            return pickle.loads(blob_data)
        except Exception:
            pass
        
        # Try ast.literal_eval on decoded string
        try:
            decoded = blob_data.decode('utf-8')
            return ast.literal_eval(decoded)
        except Exception:
            pass
    
    # If it's already a string, try ast.literal_eval
    if isinstance(blob_data, str):
        try:
            return ast.literal_eval(blob_data)
        except Exception:
            pass
    
    # Fallback
    return []


def format_spectrum(mz_list, intensity_list):
    """
    Format m/z and intensity lists as a colon-separated spectrum string.
    Each pair is formatted as "mz:intensity", space-separated between pairs.
    
    Example: [15, 26, 27] + [0.093, 0.051, 0.037] -> "15:0.093 26:0.051 27:0.037"
    """
    if not mz_list or not intensity_list:
        return ""
    
    # Ensure both are lists/iterables
    if not isinstance(mz_list, (list, tuple)):
        try:
            mz_list = list(mz_list) if hasattr(mz_list, '__iter__') else [mz_list]
        except:
            return ""
    
    if not isinstance(intensity_list, (list, tuple)):
        try:
            intensity_list = list(intensity_list) if hasattr(intensity_list, '__iter__') else [intensity_list]
        except:
            return ""
    
    pairs = []
    for mz, intensity in zip(mz_list, intensity_list):
        # Convert to float to ensure proper numeric formatting
        try:
            mz_val = float(mz)
            int_val = float(intensity)
            pairs.append(f"{mz_val}:{int_val}")
        except (ValueError, TypeError):
            # If conversion fails, skip this pair
            continue
    
    return " ".join(pairs)


def convert_db_to_json(input_db: Path, output_json: Path, limit: int = None) -> None:
    """
    Convert SQLite database to JSON file.
    
    Args:
        input_db: Path to IN_SILICO_LIBRARY.db
        output_json: Path to output JSON file
        limit: Optional limit on number of records to convert (for testing)
    """
    if not input_db.exists():
        print(f"❌ Input database not found: {input_db}")
        sys.exit(1)

    print(f"Opening database: {input_db}")
    print(f"Output file: {output_json}")

    conn = sqlite3.connect(input_db)
    cursor = conn.cursor()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM IN_SILICO_LIBRARY")
    total_count = cursor.fetchone()[0]
    print(f"📊 Total records in database: {total_count:,}")

    if limit:
        total_count = min(limit, total_count)
        print(f"⚠️  Processing first {limit:,} records only")

    # Create output directory
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Stream to JSON
    print(f"\n🔄 Converting to JSON...")
    count = 0
    errors = 0

    with open(output_json, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        query = "SELECT COMPID, SMILES, EXACTMOLWT, MZS, INTENSITYS FROM IN_SILICO_LIBRARY"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        skips = 0
        for row in cursor:
            compid, smiles, exactmolwt, mz_blob, intensity_blob = row

            #skip if molecule containts atoms other than c, h, n, o, s, p
            invalid = False
            for letter in smiles:
                if letter.lower() not in ['c', 'h', 'n', 'o', 's', 'p', '(', ')', '=', '#', '[', ']', '@', '+', '-', '/', '\\'] and not letter.isdigit():
                    invalid = True
                    break
            if invalid:
                skips += 1
                if skips % 10000 == 0:
                    print(f"  Skipped {skips:,} records with unsupported atoms ...", flush=True)
                    print(f"    Example skipped SMILES: {smiles}")
                continue

            try:
                # Deserialize blobs
                mz_list = deserialize_blob(mz_blob)
                intensity_list = deserialize_blob(intensity_blob)

                # Debug output for first record
                if count == 0:
                    print(f"\n🔍 Debug - First record:")
                    print(f"   compid: {compid} (type: {type(compid)})")
                    print(f"   smiles: {smiles} (type: {type(smiles)})")
                    print(f"   mz_list: {mz_list} (type: {type(mz_list)}, len: {len(mz_list) if hasattr(mz_list, '__len__') else 'N/A'})")
                    print(f"   intensity_list: {intensity_list} (type: {type(intensity_list)}, len: {len(intensity_list) if hasattr(intensity_list, '__len__') else 'N/A'})")
                    if isinstance(mz_list, list) and len(mz_list) > 0:
                        print(f"   first mz: {mz_list[0]} (type: {type(mz_list[0])})")
                    if isinstance(intensity_list, list) and len(intensity_list) > 0:
                        print(f"   first intensity: {intensity_list[0]} (type: {type(intensity_list[0])})")

                # Format spectrum
                spectrum = format_spectrum(mz_list, intensity_list)

                if count == 0:
                    print(f"   spectrum string: {spectrum[:100]}... (type: {type(spectrum)}, len: {len(spectrum)})\n")

                # Create record in MoNA format
                record = {
                    "smiles": smiles.strip(),
                    "spectrum": spectrum,
                }

                # Write to JSON
                if not first:
                    f.write(",\n")
                json.dump(record, f)
                first = False
                count += 1

                if count % 10_000 == 0:
                    print(f"  Processed: {count:,} / {total_count:,} ...", flush=True)

            except Exception as e:
                errors += 1
                if errors <= 5:  # Print first 5 errors
                    print(f"  ⚠️  Error processing record {compid}: {e}")
                continue

        f.write("\n]")

    conn.close()

    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {count:,} records")
    print(f"   Errors: {errors:,}")
    print(f"   Output: {output_json}")
    print(f"   Size: {output_json.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_db_to_json.py <input_db> [output_json] [--limit N]")
        print("\nExamples:")
        print("  python scripts/convert_db_to_json.py data/IN_SILICO_LIBRARY.db")
        print("  python scripts/convert_db_to_json.py data/IN_SILICO_LIBRARY.db output/in_silico.json")
        print("  python scripts/convert_db_to_json.py data/IN_SILICO_LIBRARY.db output/test.json --limit 1000")
        sys.exit(1)

    input_db = Path(sys.argv[1])

    # Default output path
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        output_json = Path(sys.argv[2])
    else:
        output_json = input_db.parent / f"{input_db.stem}.json"

    # Check for --limit flag
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    convert_db_to_json(input_db, output_json, limit=limit)


if __name__ == "__main__":
    main()
    # test=[1.342, 2.345, 3.456]
    # print(format_spectrum(test, test))
