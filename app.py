from pathlib import Path
from fastapi import FastAPI
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from fastapi.middleware.cors import CORSMiddleware
from intent_engine import parse_intent
from fastapi.middleware.cors import CORSMiddleware
import json
import requests


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from helpers import load_csv, normalize_level_id, friendly_level_label, normalize_section_name, extract_row_from_seat_name, get_first_row_for_section
from dictionaries import normalize_unit_categories, normalize_poi_categories


app = FastAPI(title="Indoor AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary for quick deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3001",
        "http://127.0.0.1:3000",
        "https://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Indoor AI Agent is running locally"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/data/preview")
def data_preview():
    units = load_csv("units.csv")
    poi = load_csv("poi.csv")
    poi_seats = load_csv("poi_seats.csv")

    return {
        "units_rows": int(len(units)),
        "poi_rows": int(len(poi)),
        "poi_seats_rows": int(len(poi_seats)),
        "units_columns": units.columns.tolist(),
        "poi_columns": poi.columns.tolist(),
        "poi_seats_columns": poi_seats.columns.tolist(),
    }

@app.get("/units/restrooms")
def get_restrooms(level_id: str = None):
    units = load_csv("units.csv")

    # Filter only restrooms
    restrooms = units[units["SYMBOL_CAT"] == "RESTROOMS"]

    # Optional level filter
    if level_id:
        restrooms = restrooms[restrooms["LEVEL_ID"] == level_id]

    # Return selected fields only (clean output)
    result = restrooms[[
        "UNIT_ID",
        "NAME",
        "USE_TYPE",
        "LEVEL_ID",
        "AREA_GROSS"
    ]]

    return result.to_dict(orient="records")

#Count restrooms by level
@app.get("/units/restrooms/count-by-level")
def count_restrooms_by_level():
    units = load_csv("units.csv")

    restrooms = units[units["SYMBOL_CAT"] == "RESTROOMS"]

    grouped = restrooms.groupby("LEVEL_ID").size().reset_index(name="count")

    return grouped.to_dict(orient="records")

#Count ADA Seats
@app.get("/seats/ada")
def get_ada_seats(level_id: str = None):
    seats = load_csv("poi_seats.csv")

    # Correct ADA logic using apostrophe
    ada_seats = seats[seats["NAME"].str.contains("'", na=False)]

    if level_id:
        ada_seats = ada_seats[ada_seats["LEVEL_ID"] == level_id]

    result = ada_seats[[
        "POI_ID",
        "NAME",
        "LEVEL_ID"
    ]]

    return result.to_dict(orient="records")

@app.get("/seats/ada/count")
def count_ada_seats(level_id: str = None):
    seats = load_csv("poi_seats.csv")

    ada_seats = seats[seats["NAME"].str.contains("'", na=False)]

    if level_id:
        ada_seats = ada_seats[ada_seats["LEVEL_ID"] == level_id]

    return {"count": int(len(ada_seats))}

#Top largest rooms
@app.get("/units/top-largest")
def top_largest_rooms(
    limit: int = 10,
    level_id: str = None,
    category: str = None,
    exclude_categories: str = None
):
    units = load_csv("units.csv").copy()

    if limit > 100:
        limit = 100

    units["AREA_GROSS"] = pd.to_numeric(units["AREA_GROSS"], errors="coerce")
    units = units[units["AREA_GROSS"].notna()]

    # 🔥 Normalize level input
    if level_id:
        normalized_level = normalize_level_id(level_id, units)
        if normalized_level:
            units = units[units["LEVEL_ID"] == normalized_level]

    if category:
        units = units[units["SYMBOL_CAT"] == category]

    if exclude_categories:
        exclude_list = [c.strip() for c in exclude_categories.split(",")]
        units = units[~units["SYMBOL_CAT"].isin(exclude_list)]

    top = units.sort_values(by="AREA_GROSS", ascending=False).head(limit)

    result = top[[
        "UNIT_ID",
        "NAME",
        "SYMBOL_CAT",
        "USE_TYPE",
        "LEVEL_ID",
        "AREA_GROSS"
    ]]

    return result.to_dict(orient="records")

#Adding for charts

@app.get("/units/area-by-category")
def units_area_by_category(level_id: str = None, limit: int = 20):
    units = load_csv("units.csv").copy()

    if limit > 50:
        limit = 50

    units["AREA_GROSS"] = pd.to_numeric(units["AREA_GROSS"], errors="coerce")
    units = units[units["AREA_GROSS"].notna()]

    if level_id:
        normalized_level = normalize_level_id(level_id, units)
        if normalized_level:
            units = units[units["LEVEL_ID"] == normalized_level]

    grouped = (
        units.groupby("SYMBOL_CAT", dropna=False)
        .agg(
            total_area_gross=("AREA_GROSS", "sum"),
            unit_count=("UNIT_ID", "count")
        )
        .reset_index()
        .rename(columns={"SYMBOL_CAT": "category"})
    )

    grouped["category"] = grouped["category"].fillna("UNKNOWN")
    grouped = grouped.sort_values(by="total_area_gross", ascending=False).head(limit)

    return grouped.to_dict(orient="records")

@app.get("/units/count-by-category")
def units_count_by_category(level_id: str = None, limit: int = 20):
    units = load_csv("units.csv").copy()

    if limit > 50:
        limit = 50

    if level_id:
        normalized_level = normalize_level_id(level_id, units)
        if normalized_level:
            units = units[units["LEVEL_ID"] == normalized_level]

    grouped = (
        units.groupby("SYMBOL_CAT", dropna=False)
        .agg(
            unit_count=("UNIT_ID", "count"),
            total_area_gross=("AREA_GROSS", "sum")
        )
        .reset_index()
        .rename(columns={"SYMBOL_CAT": "category"})
    )

    grouped["category"] = grouped["category"].fillna("UNKNOWN")
    grouped = grouped.sort_values(by="unit_count", ascending=False).head(limit)

    return grouped.to_dict(orient="records")

@app.get("/poi/count-by-category")
def poi_count_by_category(level_id: str = None, limit: int = 20):
    poi = load_csv("poi.csv").copy()

    if limit > 50:
        limit = 50

    if level_id:
        normalized_level = normalize_level_id(level_id, poi)
        if normalized_level:
            poi = poi[poi["LEVEL_ID"] == normalized_level]

    grouped = (
        poi.groupby("SYMBOL_CAT", dropna=False)
        .agg(
            poi_count=("POI_ID", "count")
        )
        .reset_index()
        .rename(columns={"SYMBOL_CAT": "category"})
    )

    grouped["category"] = grouped["category"].fillna("UNKNOWN")
    grouped = grouped.sort_values(by="poi_count", ascending=False).head(limit)

    return grouped.to_dict(orient="records")

@app.get("/units/area-by-level")
def units_area_by_level(limit: int = 20):
    units = load_csv("units.csv").copy()

    if limit > 50:
        limit = 50

    units["AREA_GROSS"] = pd.to_numeric(units["AREA_GROSS"], errors="coerce")
    units = units[units["AREA_GROSS"].notna()]

    grouped = (
        units.groupby("LEVEL_ID", dropna=False)
        .agg(
            total_area_gross=("AREA_GROSS", "sum"),
            unit_count=("UNIT_ID", "count")
        )
        .reset_index()
        .rename(columns={"LEVEL_ID": "level_id"})
    )

    grouped["level"] = grouped["level_id"].apply(friendly_level_label)
    grouped = grouped.sort_values(by="total_area_gross", ascending=False).head(limit)

    return grouped.to_dict(orient="records")

#Step 11 — Compare men’s vs women’s restroom area
#Analytics endpoint starts

@app.get("/units/restrooms/area-compare")
def compare_restroom_area(level_id: str = None):
    units = load_csv("units.csv").copy()

    # Make area numeric
    units["AREA_GROSS"] = pd.to_numeric(units["AREA_GROSS"], errors="coerce")
    units = units[units["AREA_GROSS"].notna()]

    # Only restroom units
    restrooms = units[units["SYMBOL_CAT"] == "RESTROOMS"].copy()

    # Optional level filter with normalization
    if level_id:
        normalized_level = normalize_level_id(level_id, restrooms)
        if normalized_level:
            restrooms = restrooms[restrooms["LEVEL_ID"] == normalized_level]

    # Keep only men's and women's restrooms
    restrooms = restrooms[restrooms["USE_TYPE"].isin(["MEN'S RESTROOM", "WOMEN'S RESTROOM"])]

    grouped = (
        restrooms.groupby("USE_TYPE", dropna=False)["AREA_GROSS"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    grouped = grouped.rename(columns={
        "USE_TYPE": "restroom_type",
        "count": "room_count",
        "sum": "total_area_gross",
        "mean": "average_area_gross"
    })

    return grouped.to_dict(orient="records")

#Step 12 — Count exits by level

@app.get("/poi/exits/count-by-level")
def count_exits_by_level():
    poi = load_csv("poi.csv").copy()

    exits = poi[poi["SYMBOL_CAT"] == "EXITS"].copy()

    grouped = (
        exits.groupby("LEVEL_ID", dropna=False)
        .size()
        .reset_index(name="exit_count")
    )

    grouped["level_label"] = grouped["LEVEL_ID"].apply(friendly_level_label)

    return grouped[["LEVEL_ID", "level_label", "exit_count"]].to_dict(orient="records")

#Step 13 — Show spaces by category

#Step 13.1 — Add the endpoint
@app.get("/units/by-category")
def get_units_by_category(
    category: str,
    level_id: str = None,
    limit: int = 100
):
    units = load_csv("units.csv").copy()

    if limit > 500:
        limit = 500

    categories = normalize_unit_categories(category)

    filtered = units[units["SYMBOL_CAT"].isin(categories)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, filtered)
        if normalized_level:
            filtered = filtered[filtered["LEVEL_ID"] == normalized_level]

    result = filtered[[
        "UNIT_ID",
        "NAME",
        "USE_TYPE",
        "SYMBOL_CAT",
        "LEVEL_ID",
        "AREA_GROSS"
    ]].head(limit)

    return result.to_dict(orient="records")

#Step 13.2 — Add POI category filtering too
#Add the endpoint
@app.get("/poi/by-category")
def get_poi_by_category(
    category: str,
    level_id: str = None,
    limit: int = 100
):
    poi = load_csv("poi.csv").copy()

    if limit > 500:
        limit = 500

    categories = normalize_poi_categories(category)

    filtered = poi[poi["SYMBOL_CAT"].isin(categories)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, filtered)
        if normalized_level:
            filtered = filtered[filtered["LEVEL_ID"] == normalized_level]

    result = filtered[[
        "POI_ID",
        "NAME",
        "USE_TYPE",
        "SYMBOL_CAT",
        "LEVEL_ID"
    ]].head(limit)

    return result.to_dict(orient="records")

#New Update: Adding Generic search endpoints
@app.get("/search/category")
def search_category(
    term: str,
    level_id: str = None,
    limit: int = 500
):
    if not term:
        return {
            "term": term,
            "target_layer": None,
            "records": [],
            "units": [],
            "poi": [],
            "unit_count": 0,
            "poi_count": 0,
            "total_count": 0
        }

    if limit > 500:
        limit = 500

    search_term = str(term).strip().upper()

    # Build flexible terms
    compact = search_term.replace(" ", "")

    known_compound_terms = {
        "FIRSTAID": "FIRST AID",
        "FIREPUMP": "FIRE PUMP",
        "FIREPUMPS": "FIRE PUMP",
        "FIREEXTINGUISHER": "FIRE EXTINGUISHER",
        "FIREEXTINGUISHERS": "FIRE EXTINGUISHER",
        "LOCKERROOM": "LOCKER ROOM",
        "LOCKERROOMS": "LOCKER ROOM",
        "RESTROOM": "RESTROOM",
        "BATHROOM": "RESTROOM",
        "TOILET": "RESTROOM"
    }

    phrase_terms = [search_term]

    if compact in known_compound_terms:
        phrase_terms.append(known_compound_terms[compact])

    if search_term.endswith("S"):
        phrase_terms.append(search_term[:-1])

    if search_term.endswith("IES"):
        phrase_terms.append(search_term[:-3] + "Y")

    phrase_terms.append(compact)

    # Clean duplicates
    phrase_terms = list(dict.fromkeys([t.strip().upper() for t in phrase_terms if t.strip()]))

    # Word fallback terms
    stop_words = {"ROOM", "ROOMS", "SPACE", "SPACES", "AREA", "AREAS", "ALL"}
    word_terms = [
        part.strip().upper()
        for part in search_term.split()
        if part.strip().upper() not in stop_words
    ]

    units = load_csv("units.csv").copy()
    poi = load_csv("poi.csv").copy()

    if level_id:
        normalized_unit_level = normalize_level_id(level_id, units)
        if normalized_unit_level:
            units = units[units["LEVEL_ID"] == normalized_unit_level]

        normalized_poi_level = normalize_level_id(level_id, poi)
        if normalized_poi_level:
            poi = poi[poi["LEVEL_ID"] == normalized_poi_level]

    def search_dataframe(df, terms, fields):
        mask = False

        for t in terms:
            for field in fields:
                if field in df.columns:
                    mask = mask | df[field].astype(str).str.upper().str.contains(t, na=False)

        return df[mask].copy()

    unit_fields = ["USE_TYPE", "SYMBOL_CAT", "NAME"]
    poi_fields = ["USE_TYPE", "SYMBOL_CAT", "NAME"]

    # First try phrase/compound search
    unit_results = search_dataframe(units, phrase_terms, unit_fields)
    poi_results = search_dataframe(poi, phrase_terms, poi_fields)

    # If no phrase matches, fallback to individual useful words
    if len(unit_results) == 0 and len(poi_results) == 0 and word_terms:
        unit_results = search_dataframe(units, word_terms, unit_fields)
        poi_results = search_dataframe(poi, word_terms, poi_fields)

    unit_records = unit_results[[
        "UNIT_ID",
        "NAME",
        "USE_TYPE",
        "SYMBOL_CAT",
        "LEVEL_ID",
        "AREA_GROSS"
    ]].head(limit).to_dict(orient="records")

    poi_records = poi_results[[
        "POI_ID",
        "NAME",
        "USE_TYPE",
        "SYMBOL_CAT",
        "LEVEL_ID"
    ]].head(limit).to_dict(orient="records")

    combined_records = []

    for record in unit_records:
        record["_layer"] = "Units"
        combined_records.append(record)

    for record in poi_records:
        record["_layer"] = "POI"
        combined_records.append(record)

    if len(unit_records) > 0 and len(poi_records) > 0:
        target_layer = "Mixed"
    elif len(unit_records) > 0:
        target_layer = "Units"
    elif len(poi_records) > 0:
        target_layer = "POI"
    else:
        target_layer = None

    return {
        "term": term,
        "target_layer": target_layer,
        "records": combined_records,
        "units": unit_records,
        "poi": poi_records,
        "unit_count": len(unit_records),
        "poi_count": len(poi_records),
        "total_count": len(combined_records)
    }
#Step 14 — Seat counts by level

@app.get("/seats/count-by-level")
def count_seats_by_level():
    seats = load_csv("poi_seats.csv").copy()

    grouped = (
        seats.groupby("LEVEL_ID", dropna=False)
        .size()
        .reset_index(name="seat_count")
    )

    grouped["level_label"] = grouped["LEVEL_ID"].apply(friendly_level_label)

    return grouped[["LEVEL_ID", "level_label", "seat_count"]].to_dict(orient="records")

# 14.2 Add ADA-seat counts by level
@app.get("/seats/ada/count-by-level")
def count_ada_seats_by_level():
    seats = load_csv("poi_seats.csv").copy()

    ada_seats = seats[seats["NAME"].str.contains("'", na=False)].copy()

    grouped = (
        ada_seats.groupby("LEVEL_ID", dropna=False)
        .size()
        .reset_index(name="ada_seat_count")
    )

    grouped["level_label"] = grouped["LEVEL_ID"].apply(friendly_level_label)

    return grouped[["LEVEL_ID", "level_label", "ada_seat_count"]].to_dict(orient="records")

#Step 16 — Count seats on a specific level
@app.get("/seats/count")
def count_seats(level_id: str = None, ada_only: bool = False):
    seats = load_csv("poi_seats.csv").copy()
    normalized_level = None

    if ada_only:
        seats = seats[seats["NAME"].str.contains("'", na=False)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, seats)
        if normalized_level:
            seats = seats[seats["LEVEL_ID"] == normalized_level]

    return {
        "level_id": normalized_level,
        "level_label": friendly_level_label(normalized_level) if normalized_level else None,
        "ada_only": ada_only,
        "count": int(len(seats))
    }

#Step 17 — Check section data before row/section queries
@app.get("/sections/preview")
def preview_sections(limit: int = 20):
    sections = load_csv("Sections.csv").copy()

    if limit > 100:
        limit = 100

    sample_df = sections.head(limit).copy()
    sample_df = sample_df.astype(object)
    sample_df = sample_df.where(pd.notnull(sample_df), None)

    return {
        "row_count": int(len(sections)),
        "columns": [str(col) for col in sections.columns.tolist()],
        "sample": sample_df.to_dict(orient="records")
    }

#Step 17 — Count seating sections by level
@app.get("/sections/count-by-level")
def count_sections_by_level():
    sections = load_csv("Sections.csv").copy()

    grouped = (
        sections.groupby("LEVEL_ID", dropna=False)
        .size()
        .reset_index(name="section_count")
    )

    grouped["level_label"] = grouped["LEVEL_ID"].apply(friendly_level_label)

    return grouped[["LEVEL_ID", "level_label", "section_count"]].to_dict(orient="records")

#Step 18 — Count seats in a section
@app.get("/seats/count-by-section")
def count_seats_by_section(section: str, level_id: str = None, ada_only: bool = False):
    seats = load_csv("poi_seats.csv").copy()
    normalized_level = None
    normalized_section = normalize_section_name(section)

    if ada_only:
        seats = seats[seats["NAME"].str.contains("'", na=False)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, seats)
        if normalized_level:
            seats = seats[seats["LEVEL_ID"] == normalized_level]

    # Count seats whose NAME starts with section prefix like 215-
    seats_in_section = seats[
        seats["NAME"].astype(str).str.upper().str.startswith(f"{normalized_section}-", na=False)
    ].copy()

    return {
    "section": normalized_section,
    "level_id": normalized_level,
    "level_label": friendly_level_label(normalized_level) if normalized_level else "All Levels",
    "ada_only": ada_only,
    "count": int(len(seats_in_section))
}

#Step 19 — Extract row from seat names
#Step 20 — Count seats in a row of a section
@app.get("/seats/count-by-row")
def count_seats_by_row(
    section: str,
    row: str,
    level_id: str = None,
    ada_only: bool = False
):
    seats = load_csv("poi_seats.csv").copy()

    normalized_section = normalize_section_name(section)
    normalized_row = str(row).strip().upper()
    normalized_level = None

    if ada_only:
        seats = seats[seats["NAME"].str.contains("'", na=False)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, seats)
        if normalized_level:
            seats = seats[seats["LEVEL_ID"] == normalized_level]

    # Filter section
    seats = seats[
        seats["NAME"].astype(str).str.upper().str.startswith(f"{normalized_section}-", na=False)
    ]

    # Extract row
    seats["ROW"] = seats["NAME"].apply(extract_row_from_seat_name)

    # Filter row
    seats = seats[seats["ROW"] == normalized_row]

    return {
        "section": normalized_section,
        "row": normalized_row,
        "level_id": normalized_level,
        "level_label": friendly_level_label(normalized_level) if normalized_level else "All Levels",
        "ada_only": ada_only,
        "count": int(len(seats))
    }

#Step 21 — level-aware first row logic
@app.get("/seats/count-first-row")
def count_first_row_seats(
    section: str,
    level_id: str = None,
    ada_only: bool = False
):
    seats = load_csv("poi_seats.csv").copy()

    normalized_section = normalize_section_name(section)
    normalized_level = None

    if ada_only:
        seats = seats[seats["NAME"].str.contains("'", na=False)].copy()

    if level_id:
        normalized_level = normalize_level_id(level_id, seats)
        if normalized_level:
            seats = seats[seats["LEVEL_ID"] == normalized_level]

    # Filter section
    seats = seats[
        seats["NAME"].astype(str).str.upper().str.startswith(f"{normalized_section}-", na=False)
    ].copy()

    # Extract row
    seats["ROW"] = seats["NAME"].apply(extract_row_from_seat_name)

    first_row = get_first_row_for_section(seats["ROW"].tolist(), normalized_level)

    if first_row:
        seats = seats[seats["ROW"] == first_row]
    else:
        seats = seats.iloc[0:0]

    return {
        "section": normalized_section,
        "first_row": first_row,
        "level_id": normalized_level,
        "level_label": friendly_level_label(normalized_level) if normalized_level else "All Levels",
        "ada_only": ada_only,
        "count": int(len(seats))
    }
#Step 26 — Add /ask endpoint (mapping only)

@app.get("/ask")
def ask_ai(question: str):
    try:
        prompt = f"""
You convert user questions into API calls.

Return only API path.

User question:
{question}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return {"api_call": response.choices[0].message.content.strip()}

    except Exception as e:
        return {
            "error": "AI call failed",
            "details": str(e),
            "fallback": "Use manual endpoints for now"
        }

# ------------------------------------------------------------
# Phase 1 Indoor AI Agent ask endpoint
# ------------------------------------------------------------
# ------------------------------------------------------------
# Nearest facility routing using ArcGIS Closest Facility service
# ------------------------------------------------------------

CLOSEST_FACILITY_SOLVE_URL = (
    "https://server.indoorma.ps/server/rest/services/"
    "GeoC_Demo/ArenaDemoRoute_V2/NAServer/"
    "Closest%20Facility/solveClosestFacility"
)

def clean_route_name(name):
    """
    Cleans ArcGIS route name like:
    Selected Origin - 152 (CONCESSIONS)
    into:
    152 (CONCESSIONS)
    """
    if not name:
        return "nearest facility"

    text = str(name)

    if " - " in text:
        parts = text.split(" - ")
        return parts[-1].strip()

    return text.strip()


def friendly_direction_level(text):
    """
    Converts common level labels into friendly text.
    """
    if not text:
        return text

    replacements = {
        "L0": "Basement",
        "L01": "Level 1",
        "L1": "Level 1",
        "L02": "Level 2",
        "L2": "Level 2"
    }

    cleaned = str(text).strip()

    return replacements.get(cleaned, cleaned)


def extract_level_from_direction_text(text):
    """
    Finds level references from direction text like:
    Continue forward on L0
    Continue forward on L01
    """
    if not text:
        return None

    matches = re.findall(r"\bL0?\d\b|\bL0\b", str(text))

    if matches:
        return matches[-1]

    return None

# ------------------------------------------------------------
# Transition-aware directions from traversed network edges
# ------------------------------------------------------------

PATHWAYS_SOURCE_ID = 1

TRANSITIONS_FIRST_SOURCE_OID = 25
TRANSITIONS_SOURCE_ID = 2

_PATHWAY_LOOKUP_CACHE = None
_TRANSITION_LOOKUP_CACHE = None


def get_attr_any(attrs, names):
    if not attrs:
        return None

    for name in names:
        if name in attrs and attrs[name] not in [None, ""]:
            return attrs[name]

    return None

def load_pathway_lookup():
    """
    Loads pathways.csv and builds a lookup by network SourceOID.

    The NAServer traversedEdges returns:
    SourceID = 1 for Pathways
    SourceOID = pathway row/source object id

    For this local CSV export, SourceOID maps to row number, so:
    SourceOID 1 = first CSV row
    SourceOID 2 = second CSV row
    """
    global _PATHWAY_LOOKUP_CACHE

    if _PATHWAY_LOOKUP_CACHE is not None:
        return _PATHWAY_LOOKUP_CACHE

    data_dir = Path(__file__).resolve().parent / "data"

    possible_files = [
        data_dir / "pathways.csv",
        data_dir / "Pathways.dbf.csv"
    ]

    csv_path = None

    for path in possible_files:
        if path.exists():
            csv_path = path
            break

    if not csv_path:
        _PATHWAY_LOOKUP_CACHE = {}
        return _PATHWAY_LOOKUP_CACHE

    pathways = pd.read_csv(csv_path)
    lookup = {}

    for idx, row in pathways.iterrows():
        attrs = row.to_dict()

        object_id = get_attr_any(attrs, [
            "OBJECTID",
            "ObjectID",
            "ObjectId",
            "OID",
            "FID"
        ])

        if object_id is not None:
            try:
                lookup[int(object_id)] = attrs
            except Exception:
                pass

        # Network SourceOID style fallback.
        network_source_oid = idx + 1
        lookup[int(network_source_oid)] = attrs

    _PATHWAY_LOOKUP_CACHE = lookup
    return _PATHWAY_LOOKUP_CACHE


def pathway_level_from_attrs(attrs):
    """
    Gets the friendly level name from a pathway row.
    """
    level = get_attr_any(attrs, [
        "LEVEL_NAME",
        "LEVEL_NA_1",
        "Level Name",
        "LEVEL_ID",
        "level_id"
    ])

    if not level:
        return None

    # If full LEVEL_ID, extract last part.
    level_text = str(level).strip()

    if "." in level_text:
        level_text = level_text.split(".")[-1]

    return friendly_direction_level(level_text)

def load_transition_lookup():
    """
    Loads transitions.csv and builds a lookup by ObjectID / SourceOID.

    Important:
    The DBF CSV export may not include ObjectID.
    In this Arena dataset, the first transition ObjectID is 25, so we also
    build a fallback lookup using TRANSITIONS_FIRST_SOURCE_OID + row index.
    """
    global _TRANSITION_LOOKUP_CACHE

    if _TRANSITION_LOOKUP_CACHE is not None:
        return _TRANSITION_LOOKUP_CACHE

    data_dir = Path(__file__).resolve().parent / "data"

    possible_files = [
        data_dir / "transitions.csv",
        data_dir / "Transitions.dbf.csv"
    ]

    csv_path = None

    for path in possible_files:
        if path.exists():
            csv_path = path
            break

    if not csv_path:
        _TRANSITION_LOOKUP_CACHE = {}
        return _TRANSITION_LOOKUP_CACHE

    transitions = pd.read_csv(csv_path)
    lookup = {}

    for idx, row in transitions.iterrows():
        attrs = row.to_dict()

        # If ObjectID exists in a future/exported CSV, use it directly.
        object_id = get_attr_any(attrs, [
            "OBJECTID",
            "ObjectID",
            "ObjectId",
            "OID",
            "FID"
        ])

        if object_id is not None:
            try:
                lookup[int(object_id)] = attrs
            except Exception:
                pass

        # Fallback 1:
        # Geodatabase ObjectID style.
        # Screenshot shows first transition ObjectID = 25.
        inferred_object_id = TRANSITIONS_FIRST_SOURCE_OID + idx
        lookup[int(inferred_object_id)] = attrs

        # Fallback 2:
        # Network service SourceOID style.
        # The NAServer traversedEdges returned SourceOID = 3 for the 3rd transition row.
        network_source_oid = idx + 1
        lookup[int(network_source_oid)] = attrs

    _TRANSITION_LOOKUP_CACHE = lookup
    return _TRANSITION_LOOKUP_CACHE


def transition_type_name(code):
    try:
        code = int(code)
    except Exception:
        return None

    if code == 2:
        return "stairs"
    if code == 3:
        return "ramp"
    if code == 4:
        return "elevator"
    if code == 5:
        return "escalator"
    if code == 6:
        return "moving walkway"

    return None


def build_transition_instruction_from_attrs(attrs, reverse=False):
    transition_type = get_attr_any(attrs, [
        "TRANSITION_TYPE",
        "TRANSITI_1",
        "Transition Type",
        "Transition_Type"
    ])

    transition_name = transition_type_name(transition_type)

    if not transition_name:
        return None

    from_level = get_attr_any(attrs, [
        "LEVEL_NAME_FROM",
        "LEVEL_NAME",
        "From Level Name"
    ])

    to_level = get_attr_any(attrs, [
        "LEVEL_NAME_TO",
        "LEVEL_NA_1",
        "To Level Name"
    ])

    from_level = friendly_direction_level(from_level)
    to_level = friendly_direction_level(to_level)

    # If the route traversed the transition backward, swap levels.
    if reverse:
        from_level, to_level = to_level, from_level

    if from_level and to_level and from_level != to_level:
        return f"Take the {transition_name} from {from_level} to {to_level}."

    return f"Take the {transition_name}."

def build_transition_steps_from_traversed_edges(traversed_edges):
    """
    Reads NAServer traversedEdges and inserts indoor transition instructions
    using local transitions.csv lookup.
    """
    transition_lookup = load_transition_lookup()
    

    if not transition_lookup:
        return []

    if not isinstance(traversed_edges, dict):
        return []

    features = traversed_edges.get("features", [])

    if not features:
        return []

    transition_steps = []
    seen = set()

    for edge in features:
        attrs = edge.get("attributes", {}) if isinstance(edge, dict) else {}

        source_id = attrs.get("SourceID")
        source_oid = attrs.get("SourceOID")


        # Only process transition source edges.
        # In your route output, SourceID 2 is the Transitions source.
        try:
            source_id_int = int(source_id)
        except Exception:
            continue

        if source_id_int != TRANSITIONS_SOURCE_ID:
            continue

        if source_oid is None:
            continue

        try:
            source_oid = int(source_oid)
        except Exception:
            continue

        transition_attrs = transition_lookup.get(source_oid)

        if not transition_attrs:
            continue

        from_position = attrs.get("FromPosition")
        to_position = attrs.get("ToPosition")

        reverse = False

        try:
            if from_position is not None and to_position is not None:
                reverse = float(from_position) > float(to_position)
        except Exception:
            reverse = False

        instruction = build_transition_instruction_from_attrs(
            transition_attrs,
            reverse=reverse
        )

        if not instruction:
            continue

        key = instruction.lower()

        if key in seen:
            continue

        seen.add(key)

        transition_steps.append({
            "text": instruction,
            "type": "transition"
        })

    return transition_steps

def build_route_floor_steps_from_traversed_edges(traversed_edges):
    """
    Builds floor-aware route steps from actual traversed network edges.

    This reads:
    - Pathways from SourceID 1
    - Transitions from SourceID 2

    It produces reliable floor text like:
    Continue on Level 2.
    Take the stairs from Level 2 to Level 1.
    Continue on Level 1.
    """
    pathway_lookup = load_pathway_lookup()
    transition_lookup = load_transition_lookup()

    if not isinstance(traversed_edges, dict):
        return []

    features = traversed_edges.get("features", [])

    if not features:
        return []

    route_steps = []
    last_level = None
    seen_transition_texts = set()

    for edge in features:
        attrs = edge.get("attributes", {}) if isinstance(edge, dict) else {}

        source_id = attrs.get("SourceID")
        source_oid = attrs.get("SourceOID")

        try:
            source_id_int = int(source_id)
            source_oid_int = int(source_oid)
        except Exception:
            continue

        # Pathways
        if source_id_int == PATHWAYS_SOURCE_ID:
            pathway_attrs = pathway_lookup.get(source_oid_int)

            if not pathway_attrs:
                continue

            level_name = pathway_level_from_attrs(pathway_attrs)

            if not level_name:
                continue

            if level_name != last_level:
                route_steps.append({
                    "text": f"Continue on {level_name}.",
                    "type": "level",
                    "level": level_name
                })

                last_level = level_name

        # Transitions
        elif source_id_int == TRANSITIONS_SOURCE_ID:
            transition_attrs = transition_lookup.get(source_oid_int)

            if not transition_attrs:
                continue

            from_position = attrs.get("FromPosition")
            to_position = attrs.get("ToPosition")

            reverse = False

            try:
                if from_position is not None and to_position is not None:
                    reverse = float(from_position) > float(to_position)
            except Exception:
                reverse = False

            instruction = build_transition_instruction_from_attrs(
                transition_attrs,
                reverse=reverse
            )

            if not instruction:
                continue

            key = instruction.lower()

            if key in seen_transition_texts:
                continue

            seen_transition_texts.add(key)

            route_steps.append({
                "text": instruction,
                "type": "transition"
            })

            # After a transition, let the next pathway edge add the new level.
            last_level = None

    return route_steps


def parse_transition_levels_from_text(text):
    """
    Reads:
    Take the stairs from Level 2 to Level 1.
    Returns:
    ("Level 2", "Level 1")
    """
    if not text:
        return None, None

    match = re.search(r"from\s+(.+?)\s+to\s+(.+?)\.", str(text), re.IGNORECASE)

    if not match:
        return None, None

    return match.group(1).strip(), match.group(2).strip()


def merge_route_floor_steps(direction_steps, route_floor_steps):
    """
    Safer indoor direction merge.

    We do NOT trust every pathway-derived level change because SourceOID
    lookup can produce noisy floor changes.

    We only use transition-derived floor changes because they are reliable:
    Continue on Level 2.
    Take the stairs from Level 2 to Level 1.
    Continue on Level 1.
    """
    if not route_floor_steps:
        return direction_steps

    transition_steps = [
        step for step in route_floor_steps
        if step.get("type") == "transition"
    ]

    if not transition_steps:
        return direction_steps

    cleaned = []

    for step in direction_steps:
        text = step.get("text", "")
        lower = text.lower()

        # Remove raw ArcGIS level text because it can be wrong.
        if lower.startswith("continue on "):
            continue

        # Remove older placeholder text.
        if "vertical transition" in lower:
            continue

        # Remove old injected transition text before rebuilding.
        if (
            lower.startswith("take the elevator") or
            lower.startswith("take the stairs") or
            lower.startswith("take the escalator") or
            lower.startswith("take the ramp") or
            lower.startswith("take the moving walkway")
        ):
            continue

        cleaned.append(step)

    arrive_index = None

    for i, step in enumerate(cleaned):
        if step.get("text", "").lower().startswith("arrive"):
            arrive_index = i
            break

    if arrive_index is None:
        arrive_index = len(cleaned)

    inserted_steps = []
    seen = set()

    for transition_step in transition_steps:
        transition_text = transition_step.get("text", "")
        from_level, to_level = parse_transition_levels_from_text(transition_text)

        if from_level:
            before_text = f"Continue on {from_level}."
            key = before_text.lower()

            if key not in seen:
                inserted_steps.append({
                    "text": before_text,
                    "type": "level"
                })
                seen.add(key)

        key = transition_text.lower()

        if key not in seen:
            inserted_steps.append(transition_step)
            seen.add(key)

        if to_level:
            after_text = f"Continue on {to_level}."
            key = after_text.lower()

            if key not in seen:
                inserted_steps.append({
                    "text": after_text,
                    "type": "level"
                })
                seen.add(key)

    return cleaned[:arrive_index] + inserted_steps + cleaned[arrive_index:]

def merge_transition_steps(direction_steps, transition_steps):
    """
    Inserts real transition instructions before arrival and removes generic
    vertical transition messages.
    """
    if not transition_steps:
        return direction_steps

    cleaned = [
        step for step in direction_steps
        if "vertical transition" not in step.get("text", "").lower()
    ]

    arrive_index = None

    for i, step in enumerate(cleaned):
        text = step.get("text", "").lower()

        if text.startswith("arrive"):
            arrive_index = i
            break

    if arrive_index is None:
        return cleaned + transition_steps

    return cleaned[:arrive_index] + transition_steps + cleaned[arrive_index:]


def build_indoor_directions(raw_directions, destination_name):
    """
    Converts raw ArcGIS directions into cleaner indoor-style directions.
    Stage 1:
    - Remove excessive turn-only clutter
    - Detect level changes
    - Add vertical transition messages
    """
    raw_steps = []

    if isinstance(raw_directions, list):
        for route_dir in raw_directions:
            features = route_dir.get("features", []) if isinstance(route_dir, dict) else []

            for step in features:
                attrs = step.get("attributes", {}) if isinstance(step, dict) else {}

                text = attrs.get("text") or attrs.get("Text")
                length = attrs.get("length") or attrs.get("Length")
                time_val = attrs.get("time") or attrs.get("Time")
                maneuver = attrs.get("maneuverType") or attrs.get("ManeuverType")

                if text:
                    raw_steps.append({
                        "text": str(text),
                        "length": length,
                        "time": time_val,
                        "maneuver": maneuver
                    })

    if not raw_steps:
        return []

    cleaned_steps = []
    current_level = None
    last_added_text = None

    for step in raw_steps:
        text = step.get("text", "")
        lower_text = text.lower()
        level = extract_level_from_direction_text(text)

        # Start
        if "start" in lower_text:
            new_text = "Start from the selected location."
            if new_text != last_added_text:
                cleaned_steps.append({
                    "text": new_text,
                    "type": "start"
                })
                last_added_text = new_text
            continue

        # Finish
        if "finish" in lower_text or "arrive" in lower_text:
            new_text = f"Arrive at {destination_name}."
            if new_text != last_added_text:
                cleaned_steps.append({
                    "text": new_text,
                    "type": "arrive"
                })
                last_added_text = new_text
            continue

        # Level-aware movement
        if level:
            friendly_level = friendly_direction_level(level)

            if current_level and current_level != level:
                transition_text = (
                    f"Use a vertical transition from "
                    f"{friendly_direction_level(current_level)} to {friendly_level}."
                )

                if transition_text != last_added_text:
                    cleaned_steps.append({
                        "text": transition_text,
                        "type": "transition"
                    })
                    last_added_text = transition_text

            current_level = level

            new_text = f"Continue on {friendly_level}."
            if new_text != last_added_text:
                cleaned_steps.append({
                    "text": new_text,
                    "type": "level"
                })
                last_added_text = new_text

            continue

        # Keep useful turn guidance, but avoid too many tiny repetitive turns
        useful_turn = (
            "turn left" in lower_text or
            "turn right" in lower_text or
            "go " in lower_text or
            "make a" in lower_text
        )

        if useful_turn:
            new_text = text.strip()

            if new_text != last_added_text:
                cleaned_steps.append({
                    "text": new_text,
                    "type": "turn"
                })
                last_added_text = new_text

    # Remove extra turn clutter if there are too many steps
    final_steps = []
    turn_buffer = []

    for step in cleaned_steps:
        if step.get("type") == "turn":
            turn_buffer.append(step)
        else:
            if len(turn_buffer) > 0:
                # Keep only the first two consecutive turn instructions
                final_steps.extend(turn_buffer[:2])
                turn_buffer = []

            final_steps.append(step)

    if len(turn_buffer) > 0:
        final_steps.extend(turn_buffer[:2])

    return final_steps

# ------------------------------------------------------------
# Level-aware Z handling for indoor routing
# ------------------------------------------------------------

LEVEL_Z_LOOKUP = {
    "L0": 0.0,
    "L01": 6.5532,
    "L1": 6.5532,
    "L02": 16.3068,
    "L2": 16.3068
}


def normalize_short_level_id(level_id):
    """
    Converts full Indoors LEVEL_ID values into short level keys:
    OtherBuildings.DignityHealthArena.L01 -> L01
    OtherBuildings.DignityHealthArena.L0  -> L0
    """
    if not level_id:
        return None

    text = str(level_id).strip()

    if text.endswith(".L0") or text.endswith("_L0"):
        return "L0"

    if text.endswith(".L01") or text.endswith("_L01"):
        return "L01"

    if text.endswith(".L1") or text.endswith("_L1"):
        return "L1"

    if text.endswith(".L02") or text.endswith("_L02"):
        return "L02"

    if text.endswith(".L2") or text.endswith("_L2"):
        return "L2"

    # Already short
    if text in LEVEL_Z_LOOKUP:
        return text

    return None

# Show categories / use types list in the chat before choosing one to show more details
def get_unique_values_from_fields(df, fields):
    values = set()

    if df is None or df.empty:
        return []

    for field in fields:
        if field in df.columns:
            series = df[field].dropna().astype(str).str.strip()

            for value in series:
                clean_value = value.strip()

                if not clean_value:
                    continue

                if clean_value.upper() in ["NAN", "NONE", "NULL"]:
                    continue

                values.add(clean_value.upper())

    return sorted(values)


def format_symbol_category_list(categories, label):
    if not categories:
        return f"I could not find any {label} symbol categories."

    lines = [f"I found these {label} symbol categories:"]
    lines.append("")

    for i, category in enumerate(categories, start=1):
        lines.append(f"{i}. {category}")

    lines.append("")
    lines.append(f"Which {label} category would you like me to show?")

    return "\n".join(lines)


def format_use_type_list(use_types, label, show_all=False):
    if not use_types:
        return f"I could not find any {label} use types."

    max_display = 40
    total = len(use_types)

    if show_all:
        shown = use_types
        lines = [f"I found {total} {label} use types:"]
    else:
        shown = use_types[:max_display]
        lines = [f"I found {total} {label} use types."]
        lines.append("")
        lines.append(f"Here are the first {len(shown)}:")

    lines.append("")

    for i, use_type in enumerate(shown, start=1):
        lines.append(f"{i}. {use_type}")

    if not show_all and total > max_display:
        lines.append("")
        lines.append("Do you want me to show all of them?")

    if show_all or total <= max_display:
        lines.append("")
        lines.append(f"You can type any {label} use type to show matching records.")

    return "\n".join(lines)

def get_level_id_from_attributes(attrs):
    """
    Reads LEVEL_ID from selected feature or candidate attributes.
    """
    if not attrs:
        return None

    return (
        attrs.get("LEVEL_ID") or
        attrs.get("Level_ID") or
        attrs.get("level_id") or
        attrs.get("levelId") or
        attrs.get("LEVELID")
    )


def z_from_level_id(level_id):
    """
    Returns a Z value for the level.
    """
    short_level = normalize_short_level_id(level_id)

    if not short_level:
        return None

    return LEVEL_Z_LOOKUP.get(short_level)

def to_point_geometry(geometry: dict, level_id=None):
    """
    Converts point, polygon, or polyline geometry to a clean point geometry.
    If geometry has no Z, uses LEVEL_ID to add a floor-aware Z value.
    """
    if not geometry:
        return None

    spatial_ref = geometry.get("spatialReference") or {"wkid": 102100}

    def clean_point(x, y, z=None):
        point = {
            "x": x,
            "y": y,
            "spatialReference": spatial_ref
        }

        # If Z exists in geometry, use it.
        # Otherwise, use LEVEL_ID-based fallback.
        final_z = z

        if final_z is None and level_id:
            final_z = z_from_level_id(level_id)

        if final_z is not None:
            point["z"] = final_z

        return point

    # Already a point
    if "x" in geometry and "y" in geometry:
        return clean_point(
            geometry.get("x"),
            geometry.get("y"),
            geometry.get("z")
        )

    # Polygon rings: use simple centroid from vertices
    if "rings" in geometry and geometry.get("rings"):
        xs = []
        ys = []
        zs = []

        for ring in geometry["rings"]:
            for vertex in ring:
                if len(vertex) >= 2:
                    xs.append(vertex[0])
                    ys.append(vertex[1])

                if len(vertex) >= 3 and vertex[2] is not None:
                    zs.append(vertex[2])

        if xs and ys:
            z = sum(zs) / len(zs) if zs else None

            return clean_point(
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                z
            )

    # Polyline paths: use middle vertex
    if "paths" in geometry and geometry.get("paths"):
        vertices = []

        for path in geometry["paths"]:
            for vertex in path:
                if len(vertex) >= 2:
                    vertices.append(vertex)

        if vertices:
            mid = vertices[len(vertices) // 2]
            z = mid[2] if len(mid) >= 3 else None

            return clean_point(mid[0], mid[1], z)

    return None


@app.post("/nearest")
def nearest_facility(payload: dict):
    origin = payload.get("origin")
    facilities = payload.get("facilities", [])
    destination_category = payload.get("destination_category", "facility")
    travel_mode = payload.get("travel_mode", "Walking")

    if not origin:
        return {
            "error": "Missing origin geometry.",
            "answer": "Please select a starting location first."
        }

    if not facilities:
        return {
            "error": "Missing destination facilities.",
            "answer": f"I could not find any {destination_category} candidates to route to."
        }

    origin_attrs = payload.get("origin_attributes", {}) or {}
    origin_level_id = get_level_id_from_attributes(origin_attrs)

    origin_point = to_point_geometry(
        origin,
        level_id=origin_level_id
    )

    if not origin_point:
        return {
            "error": "Invalid origin geometry.",
            "answer": "The selected origin does not have a valid route point."
        }

    facility_features = []

    for i, facility in enumerate(facilities):
        geom = facility.get("geometry")
        attrs = facility.get("attributes", {})

        facility_level_id = get_level_id_from_attributes(attrs)

        point = to_point_geometry(
            geom,
            level_id=facility_level_id
        )

        if not point:
            continue

        name = (
            attrs.get("NAME") or
            attrs.get("Name") or
            attrs.get("ROOM_NUMBER") or
            attrs.get("ROOM_NAME") or
            attrs.get("UNIT_ID") or
            attrs.get("POI_ID") or
            f"{destination_category}_{i + 1}"
        )

        use_type = (
            attrs.get("USE_TYPE") or
            attrs.get("Use_Type") or
            attrs.get("use_type") or
            ""
        )

        display_name = str(name)

        if use_type:
            display_name = f"{display_name} ({use_type})"

        # ArcGIS Closest Facility maps the Facilities class Name field.
        # Do not send extra custom fields here; they create REST warning messages.
        facility_features.append({
            "geometry": point,
            "attributes": {
                "Name": display_name
            }
        })

    if not facility_features:
        return {
            "error": "No valid facility geometries.",
            "answer": f"I found {len(facilities)} {destination_category} candidates, but none had valid geometry."
        }

    incidents = {
        "features": [
            {
                "geometry": origin_point,
                "attributes": {
                    "Name": "Selected Origin"
                }
            }
        ],
        "spatialReference": origin_point.get("spatialReference", {"wkid": 102100})
    }

    facilities_feature_set = {
        "features": facility_features,
        "spatialReference": origin_point.get("spatialReference", {"wkid": 102100})
    }

    params = {
        "f": "json",
        "incidents": json.dumps(incidents),
        "facilities": json.dumps(facilities_feature_set),

        # Closest Facility REST parameters
        "returnCFRoutes": "true",
        "returnFacilities": "true",
        "returnIncidents": "true",
        "generateDirections": "true",
        "returnDirections": "true",
        "returnTraversedEdges": "true",
        "returnTraversedJunctions": "true",
        "directionsLanguage": "en",
        "directionsOutputType": "esriDOTComplete",

        "defaultTargetFacilityCount": 1,
        "travelDirection": "esriNATravelDirectionToFacility",
        "outputLines": "esriNAOutputLineTrueShapeWithMeasure",

        # Keep snap local. Increase only if valid routes fail to solve.
        "searchTolerance": 5,
        "searchToleranceUnits": "esriMeters",
        "ignoreInvalidLocations": "true",

        "outSR": 102100
    }

    try:
        response = requests.post(CLOSEST_FACILITY_SOLVE_URL, data=params, timeout=60)
        result = response.json()
        
    except Exception as ex:
        return {
            "error": str(ex),
            "answer": "The closest facility service request failed."
        }

    if result.get("error"):
        return {
            "error": result.get("error"),
            "answer": "The closest facility service returned an error.",
            "debug": {
                "facility_candidate_count": len(facility_features),
                "origin_point": origin_point,
                "messages": result.get("messages"),
                "raw_keys": list(result.keys())
            }
        }

    routes = (
        result.get("routes", {}).get("features", []) or
        result.get("cfRoutes", {}).get("features", [])
    )

    directions = result.get("directions", [])

    if not routes:
        return {
            "answer": f"I could not find a route to the nearest {destination_category}.",
            "debug": {
                "facility_candidate_count": len(facility_features),
                "origin_point": origin_point,
                "messages": result.get("messages"),
                "solve_succeeded": result.get("solveSucceeded"),
                "raw_keys": list(result.keys())
            },
            "raw_result": result
        }

    route = routes[0]
    route_attrs = route.get("attributes", {})

    total_time = (
        route_attrs.get("Total_WalkTime") or
        route_attrs.get("Total_Minutes") or
        route_attrs.get("Total_TravelTime")
    )

    total_length = (
        route_attrs.get("Total_Length") or
        route_attrs.get("Shape_Length")
    )

    nearest_name_raw = (
        route_attrs.get("FacilityName") or
        route_attrs.get("Name") or
        route_attrs.get("FacilityID") or
        "nearest facility"
    )

    nearest_name = clean_route_name(nearest_name_raw)

    # ArcGIS often names CF routes like "Selected Origin - 024 (RESTROOM)".
    # For the chat answer, show only the destination side.
    if isinstance(nearest_name, str) and " - " in nearest_name:
        nearest_name = nearest_name.split(" - ")[-1].strip()

    destination_label = str(destination_category or "destination").strip()

    answer_parts = [f"Nearest {destination_label}: {nearest_name}."]

    if total_time is not None:
        answer_parts.append(f"Walk time: about {round(float(total_time), 1)} minutes.")

    if total_length is not None:
        answer_parts.append(f"Distance: {round(float(total_length))} meters.")

    direction_steps = build_indoor_directions(
        raw_directions=directions,
        destination_name=nearest_name
    )

    route_floor_steps = build_route_floor_steps_from_traversed_edges(
        result.get("traversedEdges")
    )

    direction_steps = merge_route_floor_steps(
        direction_steps=direction_steps,
        route_floor_steps=route_floor_steps
    )

    return {
        "answer": " ".join(answer_parts),
        "nearest": {
            "name": nearest_name,
            "destination_category": destination_category,
            "walk_time_minutes": total_time,
            "length_meters": total_length
        },
        "route": route,
        "directions": direction_steps,
        "raw_route_attributes": route_attrs,
        "messages": result.get("messages", []),
        "map_action": {
            "type": "draw_route"
        }
    }

def extract_chart_limit(question: str, default_limit: int = 20):
    q = question.lower()

    match = re.search(r"\btop\s+(\d+)\b", q)

    if match:
        try:
            return max(1, min(int(match.group(1)), 50))
        except Exception:
            return default_limit

    return default_limit


def extract_chart_level(question: str):
    """
    Reads the requested chart level.

    Important:
    If a follow-up question contains more than one level phrase
    like "basement level 2", use the LAST level mentioned.
    """
    q = question.lower()

    matches = []

    patterns = [
        (r"\bbasement\b", "L0"),
        (r"\blevel\s+0\b", "L0"),
        (r"\bl0\b", "L0"),

        (r"\blevel\s+1\b", "L01"),
        (r"\blevel\s+one\b", "L01"),
        (r"\bl1\b", "L01"),

        (r"\blevel\s+2\b", "L02"),
        (r"\blevel\s+two\b", "L02"),
        (r"\bl2\b", "L02")
    ]

    for pattern, level_id in patterns:
        for match in re.finditer(pattern, q):
            matches.append({
                "start": match.start(),
                "level_id": level_id
            })

    if not matches:
        return None

    matches = sorted(matches, key=lambda x: x["start"])

    return matches[-1]["level_id"]


def chart_level_label(level_id):
    if not level_id:
        return "All Levels"

    labels = {
        "L0": "Basement",
        "L01": "Level 1",
        "L1": "Level 1",
        "L02": "Level 2",
        "L2": "Level 2"
    }

    return labels.get(str(level_id), friendly_level_label(level_id))

# ------------------------------------------------------------
# Arena summary cards
# ------------------------------------------------------------

def load_optional_csv(filename: str):
    """
    Loads a CSV if available. Returns an empty DataFrame if missing.
    This keeps the summary endpoint from failing if seats/sections files
    are named differently or not available yet.
    """
    try:
        return load_csv(filename).copy()
    except Exception:
        return pd.DataFrame()


def format_number(value):
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "0"


def format_area(value):
    try:
        return f"{int(round(float(value))):,} sq ft"
    except Exception:
        return "0 sq ft"


def level_sort_key(level_id):
    text = str(level_id or "").strip().upper()

    if text.endswith(".L0") or text.endswith("_L0") or text == "L0":
        return 0
    if text.endswith(".L01") or text.endswith("_L01") or text == "L01" or text == "L1":
        return 1
    if text.endswith(".L02") or text.endswith("_L02") or text == "L02" or text == "L2":
        return 2

    return 99


def extract_section_from_question(text: str):
    q = str(text or "").upper()

    match = re.search(r"\bSECTION\s*[:#-]?\s*([A-Z0-9]+)\b", q)

    if not match:
        match = re.search(r"\bSEC\s*[:#-]?\s*([A-Z0-9]+)\b", q)

    if match:
        return normalize_section_name(match.group(1))

    return None


def count_seat_status_by_section(section: str, status_value: str = "Sold"):
    seats = load_csv("poi_seats.csv").copy()

    normalized_section = normalize_section_name(section)

    if "NAME" not in seats.columns:
        return {
            "section": normalized_section,
            "status": status_value,
            "count": 0,
            "error": "NAME field was not found in poi_seats.csv."
        }

    if "STATUS" not in seats.columns:
        return {
            "section": normalized_section,
            "status": status_value,
            "count": 0,
            "error": "STATUS field was not found in poi_seats.csv."
        }

    # Seat names are like 103-A-01, 215-K-21, etc.
    section_seats = seats[
        seats["NAME"]
        .astype(str)
        .str.upper()
        .str.startswith(f"{normalized_section}-", na=False)
    ].copy()

    status_text = section_seats["STATUS"].astype(str).str.upper().str.strip()

    target_status = str(status_value or "").upper().strip()

    matching = section_seats[status_text == target_status]

    return {
        "section": normalized_section,
        "status": status_value,
        "count": int(len(matching)),
        "total_section_seats": int(len(section_seats))
    }


def count_matching_records(df, search_terms, fields=None):
    """
    Counts records where any search term appears in selected fields.
    Used for amenity summary counts like restrooms, concessions, exits, first aid.
    """
    if df is None or df.empty:
        return 0

    if fields is None:
        fields = ["SYMBOL_CAT", "USE_TYPE", "NAME"]

    mask = pd.Series(False, index=df.index)

    for field in fields:
        if field not in df.columns:
            continue

        series = df[field].fillna("").astype(str).str.upper()

        for term in search_terms:
            mask = mask | series.str.contains(str(term).upper(), na=False)

    return int(mask.sum())


def get_seats_dataframe():
    """
    Tries common seat CSV names. Your project mainly uses poi_seats.csv,
    but this supports fallback names too.
    """
    possible_files = [
        "poi_seats.csv",
        "POI_Seats.csv",
        "seats.csv",
        "Seats.csv"
    ]

    for filename in possible_files:
        df = load_optional_csv(filename)
        if not df.empty:
            return df

    return pd.DataFrame()


def build_arena_summary():
    units = load_optional_csv("units.csv")
    poi = load_optional_csv("poi.csv")
    seats = get_seats_dataframe()

    # -----------------------------
    # Basic counts
    # -----------------------------
    total_units = int(len(units)) if not units.empty else 0
    total_poi = int(len(poi)) if not poi.empty else 0
    total_seats = int(len(seats)) if not seats.empty else 0

    # -----------------------------
    # Gross area
    # -----------------------------
    gross_area = 0

    if not units.empty and "AREA_GROSS" in units.columns:
        units["AREA_GROSS_NUM"] = pd.to_numeric(units["AREA_GROSS"], errors="coerce")
        gross_area = float(units["AREA_GROSS_NUM"].fillna(0).sum())

    # -----------------------------
    # Levels
    # -----------------------------
    level_values = []

    if not units.empty and "LEVEL_ID" in units.columns:
        level_values = units["LEVEL_ID"].dropna().astype(str).unique().tolist()

    level_values = sorted(level_values, key=level_sort_key)
    level_labels = [friendly_level_label(level) for level in level_values]
    level_label_text = ", ".join(level_labels) if level_labels else "Not available"

    # -----------------------------
    # ADA seats
    # -----------------------------
    ada_seats = 0

    if not seats.empty and "NAME" in seats.columns:
        ada_seats = int(seats["NAME"].astype(str).str.contains("'", na=False).sum())

    # -----------------------------
    # Amenity counts
    # -----------------------------
    # Important:
    # Restrooms and concessions are counted from Units only to avoid
    # double-counting matching POIs placed inside the same spaces.
    # Exits and first aid are counted from POIs because they are point-based amenities.

    restroom_count = count_matching_records(
        units,
        ["RESTROOM", "BATHROOM", "TOILET"],
        fields=["SYMBOL_CAT", "USE_TYPE", "NAME"]
    )

    concession_count = count_matching_records(
        units,
        ["CONCESSION", "CONCESSIONS"],
        fields=["SYMBOL_CAT", "USE_TYPE", "NAME"]
    )

    exit_count = count_matching_records(
        poi,
        ["EXIT", "EXITS"],
        fields=["SYMBOL_CAT", "USE_TYPE", "NAME"]
    )

    first_aid_count = count_matching_records(
        poi,
        ["FIRST AID", "FIRSTAID"],
        fields=["SYMBOL_CAT", "USE_TYPE", "NAME"]
    )

    # -----------------------------
    # Largest level by gross area
    # -----------------------------
    largest_level_label = "Not available"
    largest_level_area = 0

    if (
        not units.empty and
        "LEVEL_ID" in units.columns and
        "AREA_GROSS_NUM" in units.columns
    ):
        level_area = (
            units.groupby("LEVEL_ID", dropna=False)["AREA_GROSS_NUM"]
            .sum()
            .reset_index()
            .sort_values("AREA_GROSS_NUM", ascending=False)
        )

        if len(level_area) > 0:
            largest_level_id = level_area.iloc[0]["LEVEL_ID"]
            largest_level_area = float(level_area.iloc[0]["AREA_GROSS_NUM"])
            largest_level_label = friendly_level_label(largest_level_id)

    # -----------------------------
    # Largest category by gross area
    # -----------------------------
    largest_category = "Not available"
    largest_category_area = 0

    if (
        not units.empty and
        "SYMBOL_CAT" in units.columns and
        "AREA_GROSS_NUM" in units.columns
    ):
        category_area = (
            units.groupby("SYMBOL_CAT", dropna=False)["AREA_GROSS_NUM"]
            .sum()
            .reset_index()
            .sort_values("AREA_GROSS_NUM", ascending=False)
        )

        if len(category_area) > 0:
            largest_category = str(category_area.iloc[0]["SYMBOL_CAT"] or "UNKNOWN")
            largest_category_area = float(category_area.iloc[0]["AREA_GROSS_NUM"])

    cards = [
        {
            "label": "Total Units",
            "value": format_number(total_units),
            "subtext": "Mapped indoor spaces",
            "icon": "▦"
        },
        {
            "label": "Total POIs",
            "value": format_number(total_poi),
            "subtext": "Indoor points of interest",
            "icon": "⌖"
        },
        {
            "label": "Gross Area",
            "value": format_area(gross_area),
            "subtext": "Total mapped indoor area",
            "icon": "▣"
        },
        {
            "label": "Levels",
            "value": format_number(len(level_values)),
            "subtext": level_label_text,
            "icon": "↕"
        },
        {
            "label": "Total Seats",
            "value": format_number(total_seats),
            "subtext": "Mapped seat locations",
            "icon": "◉"
        },
        {
            "label": "ADA Seats",
            "value": format_number(ada_seats),
            "subtext": "Accessible seating locations",
            "icon": "♿"
        },
        {
            "label": "Restrooms",
            "value": format_number(restroom_count),
            "subtext": "Restroom spaces",
            "icon": "🚻"
        },
        {
            "label": "Concessions",
            "value": format_number(concession_count),
            "subtext": "Concession spaces",
            "icon": "🍿"
        },
        {
            "label": "Exits",
            "value": format_number(exit_count),
            "subtext": "Mapped exit points",
            "icon": "↗"
        },
        {
            "label": "First Aid",
            "value": format_number(first_aid_count),
            "subtext": "Mapped first aid points",
            "icon": "+"
        },
        {
            "label": "Largest Level",
            "value": largest_level_label,
            "subtext": format_area(largest_level_area),
            "icon": "⬢"
        },
        {
            "label": "Largest Category",
            "value": largest_category,
            "subtext": format_area(largest_category_area),
            "icon": "■"
        }
    ]

    highlights = [
        f"{largest_level_label} has the largest mapped gross area with {format_area(largest_level_area)}.",
        f"{largest_category} is the largest unit category by gross area.",
        f"The dataset includes {format_number(restroom_count)} restroom spaces, {format_number(concession_count)} concession spaces, and {format_number(exit_count)} mapped exits."
    ]

    return {
        "title": "Dignity Health Arena Summary",
        "subtitle": "High-level indoor data snapshot from the local Arena dataset",
        "cards": cards,
        "highlights": highlights
    }


@app.get("/arena/summary")
def arena_summary():
    summary = build_arena_summary()

    return {
        "answer": "Here is the Dignity Health Arena indoor data summary.",
        "summary": summary,
        "data": None,
        "map_action": {
            "type": "zoom_to_arena"
        }
    }



# ------------------------------------------------------------
# Flexible summary intent detection
# ------------------------------------------------------------

def normalize_question_for_intent(text: str):
    q = str(text or "").lower().strip()

    # Remove punctuation
    q = re.sub(r"[^a-z0-9\s]", " ", q)

    # Fix common typo variations
    typo_map = {
        "sumamry": "summary",
        "sumary": "summary",
        "summery": "summary",
        "summar": "summary",
        "summry": "summary",
        "stat": "stats",
        "statistic": "statistics",
        "statics": "statistics",
        "dashbord": "dashboard",
        "dashbaord": "dashboard",
        "areaa": "arena",
        "arina": "arena"
    }

    words = q.split()
    words = [typo_map.get(word, word) for word in words]

    q = " ".join(words)
    q = re.sub(r"\s+", " ", q).strip()

    return q


def is_arena_summary_request(question: str):
    q = normalize_question_for_intent(question)

    # Very short direct commands
    direct_commands = {
        "summary",
        "show summary",
        "give summary",
        "give me summary",
        "show me summary",
        "overview",
        "show overview",
        "stats",
        "show stats",
        "statistics",
        "dashboard",
        "dashboard summary"
    }

    if q in direct_commands:
        return True

    summary_words = [
        "summary",
        "overview",
        "stats",
        "statistics",
        "snapshot",
        "dashboard",
        "report",
        "recap",
        "high level",
        "highlevel"
    ]

    arena_words = [
        "arena",
        "dignity",
        "health",
        "indoor",
        "indoors",
        "building",
        "facility",
        "venue",
        "data",
        "dashboard"
    ]

    action_words = [
        "show",
        "give",
        "get",
        "display",
        "open",
        "create",
        "generate",
        "pull",
        "bring",
        "tell",
        "what"
    ]

    has_summary_word = any(word in q for word in summary_words)
    has_arena_context = any(word in q for word in arena_words)
    has_action_word = any(q.startswith(word + " ") or q == word for word in action_words)

    # Examples this catches:
    # show arena summary
    # give arena summary
    # give me arena stats
    # show dashboard
    # arena overview
    # indoor data snapshot
    # venue statistics
    if has_summary_word and (has_arena_context or has_action_word):
        return True

    # More conversational examples:
    conversational_patterns = [
        "what do we have",
        "what all do we have",
        "give me the big picture",
        "show me the big picture",
        "overall arena",
        "overall indoor",
        "overall data",
        "high level view",
        "high level summary"
    ]

    if any(pattern in q for pattern in conversational_patterns):
        return True

    return False
# ------------------------------------------------------------
# Phase 1 Indoor AI Agent ask endpoint
# ------------------------------------------------------------

@app.post("/ask")
async def ask_agent(payload: dict):
    question = payload.get("question", "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "intent": None,
            "data": None,
            "map_action": {
                "type": "zoom_to_arena"
            }
        }

    q = question.lower().strip()

        # ------------------------------------------------------------
    # Direct shortcut: sold/unsold seats by section
    # Must run before generic search.
    # ------------------------------------------------------------
    if "seat" in q and "section" in q and ("sold" in q or "unsold" in q):
        section = extract_section_from_question(question)

        if not section:
            return {
                "answer": "Which section do you want me to check?",
                "intent": {
                    "intent": "count_seat_status_by_section",
                    "target_layer": "Seats",
                    "response_type": "count"
                },
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        status_value = "Unsold" if "unsold" in q else "Sold"
        result = count_seat_status_by_section(section, status_value)

        if result.get("error"):
            answer = result["error"]
        else:
            answer = (
                f"Section {result['section']} has {result['count']} "
                f"{status_value.lower()} seats out of {result['total_section_seats']} total mapped seats."
            )

        return {
            "answer": answer,
            "intent": {
                "intent": "count_seat_status_by_section",
                "target_layer": "Seats",
                "response_type": "count",
                "params": {
                    "section": section,
                    "status": status_value
                }
            },
            "data": result,
            "map_action": {
                "type": "none"
            }
        }
    
        # ------------------------------------------------------------
    # Direct shortcut: total seats by section
    # ------------------------------------------------------------
    if "seat" in q and "section" in q:
        section = extract_section_from_question(question)

        if section:
            result = count_seats_by_section(
                section=section,
                level_id=None,
                ada_only=False
            )

            return {
                "answer": f"Section {result['section']} has {result['count']} total mapped seats.",
                "intent": {
                    "intent": "count_seats_by_section",
                    "target_layer": "Seats",
                    "response_type": "count",
                    "params": {
                        "section": section
                    }
                },
                "data": result,
                "map_action": {
                    "type": "none"
                }
            }
        # ------------------------------------------------------------
    # Flexible arena summary shortcut
    # ------------------------------------------------------------
    if is_arena_summary_request(question):
        summary = build_arena_summary()

        return {
            "answer": "Here is the Dignity Health Arena indoor data summary.",
            "intent": {
                "intent": "arena_summary",
                "target_layer": "Mixed",
                "response_type": "summary",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for an arena summary."
                }
            },
            "summary": summary,
            "data": None,
            "map_action": {
                "type": "zoom_to_arena"
            }
        }

# ------------------------------------------------------------
# Direct analytics shortcut: gross area by level / floor
# This prevents "show gross area by level" from falling into generic search.
# ------------------------------------------------------------
    if (
        ("gross area" in q or "area" in q)
        and
        ("by level" in q or "by floor" in q or "per level" in q or "per floor" in q)
    ):
        chart_limit = extract_chart_limit(question, default_limit=20)

        data = units_area_by_level(
            limit=chart_limit
        )

        return {
            "answer": "Here is the gross area distribution by level.",
            "intent": {
                "intent": "chart_units_area_by_level",
                "target_layer": "Units",
                "response_type": "chart",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for a gross area by level chart."
                }
            },
            "data": data,
            "chart": {
                "type": "column",
                "title": "Gross Area by Level",
                "subtitle": "Sum of gross area by floor level",
                "xField": "level",
                "yField": "total_area_gross",
                "unit": "sq ft"
            },
            "map_action": {
                "type": "none"
            }
        }

    wants_all = any(phrase in q for phrase in [
        "show all",
        "list all",
        "all of them",
        "show all of them",
        "complete list",
        "full list"
    ])

    # ------------------------------------------------------------
    # List unit / POI symbol categories
    # Categories = SYMBOL_CAT only
    # ------------------------------------------------------------
    is_use_type_question = (
        "use type" in q or
        "use types" in q or
        "usetype" in q or
        "usetypes" in q
    )

    is_visual_chart_question = (
        "chart" in q or
        "graph" in q or
        "plot" in q or
        "visual" in q or
        "visualize" in q
    )

    is_analytics_chart_question = (
        (
            "gross area" in q or
            "area" in q or
            "sq ft" in q or
            "square feet" in q
        )
        and
        (
            "by level" in q or
            "by floor" in q or
            "by category" in q or
            "by categories" in q or
            "by unit category" in q or
            "by symbol category" in q
        )
    ) or (
        (
            "count" in q or
            "number of" in q or
            "how many" in q
        )
        and
        (
            "by category" in q or
            "by categories" in q or
            "by unit category" in q or
            "by poi category" in q or
            "by level" in q or
            "by floor" in q
        )
    )

    is_chart_question = is_visual_chart_question or is_analytics_chart_question

    is_category_question = (
        "category" in q or
        "categories" in q or
        "symbol category" in q or
        "symbol categories" in q
    )

    if is_category_question and not is_use_type_question and not is_chart_question:
        if "poi" in q or "pois" in q:
            poi = load_csv("poi.csv").copy()

            poi_categories = get_unique_values_from_fields(
                poi,
                ["SYMBOL_CAT", "SYMBOL_CATEGORY"]
            )

            return {
                "answer": format_symbol_category_list(poi_categories, "POI"),
                "intent": {
                    "intent": "list_poi_symbol_categories",
                    "target_layer": "POI",
                    "response_type": "message",
                    "validation": {
                        "is_valid": True,
                        "needs_clarification": True,
                        "message": "User asked for available POI symbol categories."
                    }
                },
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        units = load_csv("units.csv").copy()

        unit_categories = get_unique_values_from_fields(
            units,
            ["SYMBOL_CAT", "SYMBOL_CATEGORY"]
        )

        return {
            "answer": format_symbol_category_list(unit_categories, "unit"),
            "intent": {
                "intent": "list_unit_symbol_categories",
                "target_layer": "Units",
                "response_type": "message",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": True,
                    "message": "User asked for available unit symbol categories."
                }
            },
            "data": None,
            "map_action": {
                "type": "none"
            }
        }

    # ------------------------------------------------------------
    # List unit / POI use types
    # Use types = USE_TYPE only
    # ------------------------------------------------------------
    if is_use_type_question:
        if "poi" in q or "pois" in q:
            poi = load_csv("poi.csv").copy()

            poi_use_types = get_unique_values_from_fields(
                poi,
                ["USE_TYPE"]
            )

            return {
                "answer": format_use_type_list(
                    poi_use_types,
                    "POI",
                    show_all=wants_all
                ),
                "intent": {
                    "intent": "list_poi_use_types",
                    "target_layer": "POI",
                    "response_type": "message",
                    "follow_up": {
                        "type": "show_all_poi_use_types"
                    } if not wants_all else None,
                    "validation": {
                        "is_valid": True,
                        "needs_clarification": not wants_all,
                        "message": "User asked for available POI use types."
                    }
                },
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        units = load_csv("units.csv").copy()

        unit_use_types = get_unique_values_from_fields(
            units,
            ["USE_TYPE"]
        )

        return {
            "answer": format_use_type_list(
                unit_use_types,
                "unit",
                show_all=wants_all
            ),
            "intent": {
                "intent": "list_unit_use_types",
                "target_layer": "Units",
                "response_type": "message",
                "follow_up": {
                    "type": "show_all_unit_use_types"
                } if not wants_all else None,
                "validation": {
                    "is_valid": True,
                    "needs_clarification": not wants_all,
                    "message": "User asked for available unit use types."
                }
            },
            "data": None,
            "map_action": {
                "type": "none"
            }
        }

        # ------------------------------------------------------------
    # Chart: gross area by unit symbol category
    # ------------------------------------------------------------
        # ------------------------------------------------------------
    # Charts / analytics
    # ------------------------------------------------------------
    is_chart_question = (
        "chart" in q or
        "graph" in q or
        "plot" in q or
        "visual" in q or
        "visualize" in q
    )

    chart_limit = extract_chart_limit(question, default_limit=20)
    chart_level_id = extract_chart_level(question)

    mentions_area = (
        "gross area" in q or
        "area" in q or
        "sq ft" in q or
        "square feet" in q
    )

    mentions_count = (
        "count" in q or
        "number of" in q or
        "how many" in q
    )

    mentions_category = (
        "category" in q or
        "categories" in q or
        "symbol category" in q or
        "symbol categories" in q
    )

    mentions_by_level = (
        "by level" in q or
        "by floor" in q or
        "per level" in q or
        "per floor" in q or
        "level chart" in q or
        "floor chart" in q
    )

    mentions_poi = (
        "poi" in q or
        "pois" in q or
        "point of interest" in q or
        "points of interest" in q
    )

    # Chart 1: Gross area by level
    # Important: this should only trigger for "by level" / "by floor",
    # not for "level 1 only".
    if is_chart_question and mentions_area and mentions_by_level:
        data = units_area_by_level(
            limit=chart_limit
        )

        return {
            "answer": "Here is the gross area distribution by level.",
            "intent": {
                "intent": "chart_units_area_by_level",
                "target_layer": "Units",
                "response_type": "chart",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for a gross area by level chart."
                }
            },
            "data": data,
            "chart": {
                "type": "column",
                "title": "Gross Area by Level",
                "subtitle": "Sum of gross area by floor level",
                "xField": "level",
                "yField": "total_area_gross",
                "unit": "sq ft"
            },
            "map_action": {
                "type": "none"
            }
        }

    # Chart 2: Gross area by unit category
    # This allows level filters like "level 1 only".
    if is_chart_question and mentions_area and mentions_category:
        data = units_area_by_category(
            level_id=chart_level_id,
            limit=chart_limit
        )

        level_text = chart_level_label(chart_level_id)

        return {
            "answer": f"Here is the gross area distribution by unit category for {level_text}.",
            "intent": {
                "intent": "chart_units_area_by_category",
                "target_layer": "Units",
                "response_type": "chart",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for a gross area by category chart."
                }
            },
            "data": data,
            "chart": {
                "type": "column",
                "title": "Gross Area by Unit Category",
                "subtitle": f"Sum of gross area by SYMBOL_CAT · {level_text}",
                "xField": "category",
                "yField": "total_area_gross",
                "unit": "sq ft"
            },
            "map_action": {
                "type": "none"
            }
        }

    # Chart 3: POI count by category
    if is_chart_question and mentions_poi and mentions_category:
        data = poi_count_by_category(
            level_id=chart_level_id,
            limit=chart_limit
        )

        level_text = chart_level_label(chart_level_id)

        return {
            "answer": f"Here is the POI count by category for {level_text}.",
            "intent": {
                "intent": "chart_poi_count_by_category",
                "target_layer": "POI",
                "response_type": "chart",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for a POI count by category chart."
                }
            },
            "data": data,
            "chart": {
                "type": "column",
                "title": "POI Count by Category",
                "subtitle": f"Count of POIs by SYMBOL_CAT · {level_text}",
                "xField": "category",
                "yField": "poi_count",
                "unit": "POIs"
            },
            "map_action": {
                "type": "none"
            }
        }

    # Chart 4: Count by unit category
    if is_chart_question and mentions_count and mentions_category and not mentions_poi:
        data = units_count_by_category(
            level_id=chart_level_id,
            limit=chart_limit
        )

        level_text = chart_level_label(chart_level_id)

        return {
            "answer": f"Here is the unit count by category for {level_text}.",
            "intent": {
                "intent": "chart_units_count_by_category",
                "target_layer": "Units",
                "response_type": "chart",
                "validation": {
                    "is_valid": True,
                    "needs_clarification": False,
                    "message": "User asked for a unit count by category chart."
                }
            },
            "data": data,
            "chart": {
                "type": "column",
                "title": "Unit Count by Category",
                "subtitle": f"Count of units by SYMBOL_CAT · {level_text}",
                "xField": "category",
                "yField": "unit_count",
                "unit": "units"
            },
            "map_action": {
                "type": "none"
            }
        }

    intent_json = parse_intent(question)
    validation = intent_json.get("validation", {})

    if not validation.get("is_valid"):
        return {
            "answer": validation.get("message"),
            "intent": intent_json,
            "data": None,
            "map_action": {
                "type": "none"
            }
        }

    intent = intent_json.get("intent")
    params = intent_json.get("parameters", {})

    limit = params.get("limit") or 10
    level_id = params.get("level_id")
    category = params.get("category")

    data = None
    answer = "Intent matched successfully."

    # ------------------------------------------------------------
    # Route: top largest units / rooms
    # ------------------------------------------------------------
    if intent == "rank_units_by_area":
    # For category-based ranking, use the existing by-category function first
    # because it already handles category normalization.
        if category:
            category_records = get_units_by_category(
                category=category,
                level_id=level_id,
                limit=500
            )

            category_records = sorted(
                category_records,
                key=lambda x: float(x.get("AREA_GROSS") or 0),
                reverse=True
            )

            data = category_records[:limit]

        else:
            data = top_largest_rooms(
                limit=limit,
                level_id=level_id,
                category=None
            )
        if category:
            answer = f"Here are the top {limit} largest {category} spaces."
        else:
            answer = f"Here are the top {limit} largest units by gross area."
    
    # ------------------------------------------------------------
    # Route: units by category
    # ------------------------------------------------------------
    
        
    elif intent == "filter_units_by_category":
        if not category:
            return {
                "answer": "Which unit category do you want me to show?",
                "intent": intent_json,
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        data = get_units_by_category(
            category=category,
            level_id=level_id,
            limit=500
        )

        answer = f"I found {len(data)} {category} unit records."

        # ------------------------------------------------------------
    # Route: POI by category
    # ------------------------------------------------------------
    elif intent == "filter_poi_by_category":
        if not category:
            return {
                "answer": "Which POI category do you want me to show?",
                "intent": intent_json,
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        data = get_poi_by_category(
            category=category,
            level_id=level_id,
            limit=500
        )

        answer = f"I found {len(data)} {category} POI records."


    # ------------------------------------------------------------
    # Placeholder until we connect remaining endpoints
    # ------------------------------------------------------------
        # ------------------------------------------------------------
    # Route: total seat count
    # ------------------------------------------------------------
    elif intent == "count_total_seats":
        result = count_seats(
            level_id=level_id,
            ada_only=False
        )

        data = result
        answer = f"Total seats: {result['count']}"

    # ------------------------------------------------------------
    # Route: seats by section
    # ------------------------------------------------------------
    elif intent == "count_seats_by_section":
        section = params.get("section")

        result = count_seats_by_section(
            section=section,
            level_id=level_id,
            ada_only=False
        )

        data = result
        answer = f"Section {result['section']} has {result['count']} seats."

    # ------------------------------------------------------------
    # Route: seats by row
    # ------------------------------------------------------------
    elif intent == "count_seats_by_row":
        section = params.get("section")
        row = params.get("row")

        # For now, assume section is required for row queries
        if not section:
            return {
                "answer": "Please provide a section for row-based seat count.",
                "intent": intent_json,
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        result = count_seats_by_row(
            section=section,
            row=row,
            level_id=level_id,
            ada_only=False
        )

        data = result
        answer = f"Row {result['row']} in section {result['section']} has {result['count']} seats."
    
        # ------------------------------------------------------------
    # Route: seats by level
    # ------------------------------------------------------------
    elif intent == "count_seats_by_level":
        data = count_seats_by_level()
        total = sum(item.get("seat_count", 0) for item in data)
        answer = f"There are {total} seats across {len(data)} levels."

    # ------------------------------------------------------------
    # Route: ADA seats by level
    # ------------------------------------------------------------
    elif intent == "count_ada_seats_by_level":
        data = count_ada_seats_by_level()
        total = sum(item.get("ada_seat_count", 0) for item in data)
        answer = f"There are {total} ADA seats across {len(data)} levels."

    # ------------------------------------------------------------
    # Route: sections by level
    # ------------------------------------------------------------
    elif intent == "count_sections_by_level":
        data = count_sections_by_level()
        total = sum(item.get("section_count", 0) for item in data)
        answer = f"There are {total} sections across {len(data)} levels."



        # ------------------------------------------------------------
    # Route: restroom area comparison
    # ------------------------------------------------------------
    elif intent == "compare_restroom_area":
        data = compare_restroom_area(
            level_id=level_id
        )

        answer = "Here is the restroom area comparison."
            # ------------------------------------------------------------
    # Placeholder until we connect remaining endpoints
    # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
    # Route: generic category search across Units and POI
    # ------------------------------------------------------------
        # ------------------------------------------------------------
    # Route: generic category search across Units and POI
    # ------------------------------------------------------------
    elif intent == "generic_category_search":
        search_term = params.get("search_term")
        generic_action = params.get("generic_action")

        if not search_term:
            return {
                "answer": "What do you want me to search for?",
                "intent": intent_json,
                "data": None,
                "map_action": {
                    "type": "none"
                }
            }

        result = search_category(
            term=search_term,
            level_id=level_id,
            limit=500
        )

        # 🔥 CASE 1: COUNT REQUEST
        if generic_action == "count":
            data = {
                "term": search_term,
                "unit_count": result.get("unit_count"),
                "poi_count": result.get("poi_count"),
                "total_count": result.get("total_count")
            }

            answer = (
                f"I found {data['total_count']} records matching '{search_term}': "
                f"{data['unit_count']} Units and {data['poi_count']} POIs."
            )

            intent_json["map_action"] = {"type": "none"}

        # 🔥 CASE 2: SHOW REQUEST
        else:
            data = result.get("records", [])

            intent_json["target_layer"] = result.get("target_layer")

            if not data:
                answer = f"I could not find any records matching '{search_term}'."
                intent_json["map_action"] = {"type": "none"}
            else:
                answer = (
                    f"I found {result.get('total_count')} records matching '{search_term}': "
                    f"{result.get('unit_count')} Units and {result.get('poi_count')} POIs."
                )
    else:
        return {
            "answer": "Intent matched successfully. This endpoint will be connected next.",
            "intent": intent_json,
            "data": None,
            "map_action": intent_json.get("map_action")
        }

    return {
        "answer": answer,
        "intent": intent_json,
        "data": data,
        "map_action": intent_json.get("map_action")
    }

