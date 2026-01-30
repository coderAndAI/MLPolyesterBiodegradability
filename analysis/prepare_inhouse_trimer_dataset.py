"""Prepare an ML-ready dataset of in-house trimer SMILES and biodegradability."""

from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_PATH = DATA_DIR / "inhouse_data.csv"
OUTPUT_PATH = DATA_DIR / "inhouse_trimer_ml_dataset.csv"


def main() -> None:
    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    output_rows = []
    for row in rows:
        trimer = (row.get("Trimer") or "").strip()
        biodeg = (row.get("Biodegradability_yes") or "").strip()
        if not trimer or biodeg == "":
            continue
        output_rows.append(
            {
                "trimer_smiles": trimer,
                "biodegradability_label": biodeg,
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trimer_smiles", "biodegradability_label"])
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
