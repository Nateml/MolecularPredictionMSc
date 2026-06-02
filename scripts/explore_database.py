#!/usr/bin/env python3
"""
Explore and inspect the IN_SILICO_LIBRARY.db database.
Detects format, schema, and sample data.
"""

import sys
import sqlite3
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def explore_sqlite(db_path: Path) -> None:
    """Inspect a SQLite database: schema, tables, and sample rows."""
    print(f"\n{'='*80}")
    print(f"SQLite Database: {db_path.name}")
    print(f"{'='*80}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ── list all tables ───────────────────────────────────────────────────
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            print("⚠️  No tables found in database.")
            conn.close()
            return

        print(f"\n📊 Tables found: {len(tables)}")
        for tbl in tables:
            print(f"   • {tbl}")

        # ── inspect each table ────────────────────────────────────────────────
        for table_name in tables:
            print(f"\n{'-'*80}")
            print(f"TABLE: {table_name}")
            print(f"{'-'*80}")

            # Schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print(f"\n📋 Schema ({len(columns)} columns):")
            for col_id, col_name, col_type, notnull, dflt_value, pk in columns:
                nullable = "NULL" if not notnull else "NOT NULL"
                pk_str = " [PRIMARY KEY]" if pk else ""
                print(f"   {col_name:30} {col_type:15} {nullable}{pk_str}")

            # Row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"\n📈 Row count: {row_count:,}")

            # Sample data
            print(f"\n📋 First 5 rows:")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            col_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            # Print header
            col_widths = [max(len(name), 20) for name in col_names]
            header = " | ".join(
                name.ljust(width) for name, width in zip(col_names, col_widths)
            )
            print(f"   {header}")
            print(f"   {'-' * len(header)}")

            # Print rows
            for row in rows:
                row_str = " | ".join(
                    str(val)[:width].ljust(width)
                    for val, width in zip(row, col_widths)
                )
                print(f"   {row_str}")

            # Index info
            cursor.execute(f"PRAGMA index_list({table_name});")
            indices = cursor.fetchall()
            if indices:
                print(f"\n🔍 Indices:")
                for idx in indices:
                    print(f"   • {idx[1]}")

        conn.close()
        print(f"\n{'='*80}\n")

    except sqlite3.Error as e:
        print(f"❌ SQLite Error: {e}")
        return


def check_file_format(db_path: Path) -> bool:
    """Check if file is SQLite by reading magic bytes."""
    try:
        with open(db_path, "rb") as f:
            header = f.read(16)
            # SQLite 3 magic: "SQLite format 3\x00"
            if header.startswith(b"SQLite format 3"):
                print(f"✓ File identified as SQLite 3 database")
                return True
            else:
                print(f"⚠️  File header: {header[:8]}")
                print(f"   This doesn't appear to be a standard SQLite database.")
                return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def main() -> None:
    db_path = Path(r"C:\Users\Tom\Desktop\Nieuwe map\MolecularPredictionMSc\data\IN_SILICO_LIBRARY.db")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Searched in: {db_path.resolve()}")
        sys.exit(1)

    print(f"\n🔍 Exploring database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Check file format
    if check_file_format(db_path):
        explore_sqlite(db_path)
    else:
        print("\n💡 Try opening the file with a hex editor or database viewer.")


if __name__ == "__main__":
    main()
