import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_csv(filename):
    data_dir = Path(__file__).resolve().parent / "data"
    path = data_dir / filename

    if not path.exists():
        # Render/Linux is case-sensitive, so find the CSV ignoring case
        for file in data_dir.iterdir():
            if file.name.lower() == filename.lower():
                path = file
                break

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)

def normalize_level_id(level_input: str, df):
    if not level_input:
        return None

    level_input = level_input.lower()

    if "basement" in level_input or "lower" in level_input:
        target_num = 0
    elif "ground" in level_input:
        target_num = 0
    else:
        match = re.search(r"\d+", level_input)
        if not match:
            return None
        target_num = int(match.group())

    unique_levels = df["LEVEL_ID"].dropna().unique()

    for lvl in unique_levels:
        lvl_match = re.search(r"\d+", str(lvl))
        if lvl_match and int(lvl_match.group()) == target_num:
            return lvl

    return None

def friendly_level_label(level_id: str) -> str:
    if not level_id:
        return "Unknown"

    level_id_lower = str(level_id).lower()

    if level_id_lower.endswith(".l0"):
        return "Basement"
    if level_id_lower.endswith(".l01"):
        return "Level 1"
    if level_id_lower.endswith(".l02"):
        return "Level 2"

    match = re.search(r"\.l0*(\d+)$", level_id_lower)
    if match:
        num = int(match.group(1))
        return f"Level {num}"

    return str(level_id)



#Seat Count in a section
def normalize_section_name(section_input: str) -> str:
    if not section_input:
        return None

    section_input = str(section_input).strip().upper()

    # remove common words
    for word in ["SECTION", "SEC"]:
        section_input = section_input.replace(word, "").strip()

    return section_input

#Step 19 — Extract row from seat names
def extract_row_from_seat_name(seat_name: str):
    if not seat_name:
        return None

    parts = str(seat_name).split("-")

    if len(parts) < 3:
        return None

    # Middle part is row
    return parts[1].strip().upper()

#Step 21  — level-aware first row logic
def get_expected_first_rows_for_level(level_id: str):
    level_str = str(level_id).lower() if level_id else ""

    if level_str.endswith(".l0"):
        return ["A"]

    if level_str.endswith(".l01"):
        return ["G", "H"]

    if level_str.endswith(".l02"):
        return ["A"]

    return []

def get_first_row_for_section(rows, level_id: str = None):
    cleaned_rows = []

    for row in rows:
        if row is None:
            continue
        row_str = str(row).strip().upper()
        if row_str:
            cleaned_rows.append(row_str)

    unique_rows = sorted(set(cleaned_rows))

    if not unique_rows:
        return None

    # Try expected starting rows based on level first
    expected_rows = get_expected_first_rows_for_level(level_id)
    for expected in expected_rows:
        if expected in unique_rows:
            return expected

    # Fallback
    return unique_rows[0]
