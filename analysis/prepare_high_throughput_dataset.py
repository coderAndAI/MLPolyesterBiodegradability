"""Prepare high-throughput reaction data for machine learning."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SD01_PATH = DATA_DIR / "pnas.2220021120.sd01.csv"
SD02_PATH = DATA_DIR / "pnas.2220021120.sd02.csv"
SD03_PATH = DATA_DIR / "pnas.2220021120.sd03.csv"

OUTPUT_SUCCESS_PATH = DATA_DIR / "high_throughput_ml_dataset.csv"
OUTPUT_UNSUCCESS_PATH = DATA_DIR / "high_throughput_unsuccessful_combinations.csv"
OUTPUT_REPORT_PATH = Path(__file__).resolve().parent / "high_throughput_dataset_report.md"


ROLE_ALIASES = {
    "monomer": "monomer",
    "base": "base",
    "solvent": "solvent",
    "catalyst": "catalyst",
    "co catalyst": "co-catalyst",
    "co-catalyst": "co-catalyst",
    "cocatalyst": "co-catalyst",
    "initiator": "initiator",
    "quenching agent": "quenching agent",
}


def normalize_role(role: str) -> Optional[str]:
    cleaned = " ".join(role.strip().lower().replace("_", " ").replace("-", " ").split())
    if not cleaned:
        return None
    return ROLE_ALIASES.get(cleaned, cleaned)


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_reaction_table(path: Path, id_label: str) -> Dict[str, List[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return {}

    header = rows[0]
    data_rows = rows[2:]
    expected_header = [id_label, "Reagent Role", "Reagent", "Mass", "Volume"]
    if header[:5] != expected_header:
        raise ValueError(f"Unexpected header in {path}: {header}")

    reactions: Dict[str, List[dict]] = defaultdict(list)
    current_id: Optional[str] = None
    for row in data_rows:
        if not row:
            continue
        reaction_id = row[0].strip()
        if reaction_id:
            current_id = reaction_id
        if current_id is None:
            continue

        role = normalize_role(row[1]) if len(row) > 1 else None
        reagent = row[2].strip() if len(row) > 2 else ""
        mass = parse_float(row[3]) if len(row) > 3 else None
        volume = parse_float(row[4]) if len(row) > 4 else None

        if role is None and not reagent and mass is None and volume is None:
            continue

        reactions[current_id].append(
            {
                "role": role,
                "reagent": reagent,
                "mass_mg": mass,
                "volume_ml": volume,
            }
        )
    return reactions


def read_property_table(path: Path) -> Dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    header = rows[0]
    expected = ["Polymer Number", "Solubility", "Physical State", "Biodegradability"]
    if header != expected:
        raise ValueError(f"Unexpected header in {path}: {header}")
    properties: Dict[str, dict] = {}
    for row in rows[2:]:
        if not row:
            continue
        polymer_id = row[0].strip()
        if not polymer_id:
            continue
        properties[polymer_id] = {
            "solubility": row[1].strip().lower(),
            "physical_state": row[2].strip().lower(),
            "biodegradability": row[3].strip().lower(),
        }
    return properties


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def build_feature_row(reaction_id: str, entries: List[dict]) -> dict:
    role_entries: Dict[str, List[dict]] = defaultdict(list)
    for entry in entries:
        role = entry["role"]
        if role is None:
            continue
        role_entries[role].append(entry)

    row = {"polymer_number": reaction_id}

    monomers = role_entries.get("monomer", [])
    for idx in range(2):
        prefix = f"monomer_{idx + 1}"
        if idx < len(monomers):
            row[prefix] = monomers[idx]["reagent"]
            row[f"{prefix}_mass_mg"] = monomers[idx]["mass_mg"]
        else:
            row[prefix] = ""
            row[f"{prefix}_mass_mg"] = None

    if len(monomers) > 2:
        row["monomer_other"] = "; ".join(
            dedupe_preserve([m["reagent"] for m in monomers[2:]])
        )
    else:
        row["monomer_other"] = ""

    for role in sorted(role_entries.keys()):
        if role == "monomer":
            continue
        entries_for_role = role_entries[role]
        reagents = dedupe_preserve([entry["reagent"] for entry in entries_for_role])
        total_mass = sum(
            entry["mass_mg"] for entry in entries_for_role if entry["mass_mg"] is not None
        )
        total_volume = sum(
            entry["volume_ml"] for entry in entries_for_role if entry["volume_ml"] is not None
        )
        row[f"{role}_reagents"] = "; ".join(reagents)
        row[f"{role}_mass_mg_total"] = total_mass if entries_for_role else None
        row[f"{role}_volume_ml_total"] = total_volume if entries_for_role else None

    return row


def ensure_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def write_dataset(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: ensure_float(value) if isinstance(value, (float, int)) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def main() -> None:
    success_reactions = read_reaction_table(SD01_PATH, "Polymer Number")
    unsuccessful_reactions = read_reaction_table(
        SD02_PATH, "Unsuccessful Combination Attempt Number"
    )
    properties = read_property_table(SD03_PATH)

    success_rows: List[dict] = []
    missing_properties = []
    for reaction_id, entries in success_reactions.items():
        if reaction_id not in properties:
            missing_properties.append(reaction_id)
            continue
        row = build_feature_row(reaction_id, entries)
        props = properties[reaction_id]
        row.update(
            {
                "solubility": props["solubility"],
                "physical_state": props["physical_state"],
                "biodegradability": props["biodegradability"],
                "solubility_label": 1 if props["solubility"] == "yes" else 0,
                "biodegradability_label": 1 if props["biodegradability"] == "yes" else 0,
            }
        )
        success_rows.append(row)

    unsuccess_rows: List[dict] = []
    for reaction_id, entries in unsuccessful_reactions.items():
        row = build_feature_row(reaction_id, entries)
        row["successful"] = 0
        unsuccess_rows.append(row)

    def sort_key(value: str) -> tuple[int, str]:
        try:
            return (0, f"{int(value):06d}")
        except ValueError:
            return (1, value)

    success_rows.sort(key=lambda item: sort_key(item["polymer_number"]))
    unsuccess_rows.sort(key=lambda item: sort_key(item["polymer_number"]))

    base_fields = [
        "polymer_number",
        "monomer_1",
        "monomer_1_mass_mg",
        "monomer_2",
        "monomer_2_mass_mg",
        "monomer_other",
    ]

    role_fields = set()
    for row in success_rows + unsuccess_rows:
        for key in row:
            if key not in base_fields and key not in {
                "solubility",
                "physical_state",
                "biodegradability",
                "solubility_label",
                "biodegradability_label",
                "successful",
            }:
                role_fields.add(key)

    role_fields_sorted = sorted(role_fields)

    success_fields = (
        base_fields
        + role_fields_sorted
        + [
            "solubility",
            "physical_state",
            "biodegradability",
            "solubility_label",
            "biodegradability_label",
        ]
    )
    unsuccess_fields = base_fields + role_fields_sorted + ["successful"]

    write_dataset(OUTPUT_SUCCESS_PATH, success_rows, success_fields)
    write_dataset(OUTPUT_UNSUCCESS_PATH, unsuccess_rows, unsuccess_fields)

    report_lines = [
        "# High-throughput dataset preparation report",
        "",
        f"Source reactions (successful): {len(success_reactions)}",
        f"Source reactions (unsuccessful): {len(unsuccessful_reactions)}",
        f"Property rows: {len(properties)}",
        f"Rows in ML dataset (successful): {len(success_rows)}",
        f"Rows in ML dataset (unsuccessful): {len(unsuccess_rows)}",
        f"Missing properties for polymer numbers: {', '.join(sorted(missing_properties, key=int)) if missing_properties else 'None'}",
        "",
        "Generated files:",
        f"- {OUTPUT_SUCCESS_PATH.relative_to(Path.cwd())}",
        f"- {OUTPUT_UNSUCCESS_PATH.relative_to(Path.cwd())}",
    ]
    OUTPUT_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
