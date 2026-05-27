import re
from typing import Dict, Any, Optional


# ------------------------------------------------------------
# 1. Synonym dictionary
# ------------------------------------------------------------

SYNONYMS = {
    "unit": ["unit", "units", "room", "rooms", "space", "spaces", "area", "areas"],
    "poi": ["poi", "pois", "point of interest", "points of interest"],
    "seat": ["seat", "seats", "chair", "chairs"],
    "section": ["section", "sections"],
    "level": ["level", "levels", "floor", "floors", "story", "stories"],
    "restroom": ["restroom", "restrooms", "bathroom", "bathrooms", "toilet", "toilets", "washroom", "washrooms"],
    "office": ["office", "offices"],
    "concession": ["concession", "concessions", "food", "food stand", "food stands"],
    "elevator": ["elevator", "elevators", "lift", "lifts"],
    "stair": ["stair", "stairs", "staircase", "staircases", "stairway", "stairways"],
    "exit": ["exit", "exits", "entrance", "entrances", "door", "doors"],
    "ada": ["ada", "accessible", "accessibility", "wheelchair"],
    "largest": ["largest", "biggest", "top", "highest area", "most area", "largest area", "biggest area"],
    "count": ["count", "counts", "how many", "number of", "total", "totals"],
    "compare": ["compare", "comparison", "versus", "vs"],
    "row": ["row", "rows"],
    "nearest": ["nearest", "closest", "nearby", "near me"],
}


# ------------------------------------------------------------
# 2. Intent catalog
# ------------------------------------------------------------

INTENT_CATALOG = {
    "rank_units_by_area": {
        "endpoint": "/units/top-largest",
        "target_layer": "Units",
        "required_concepts": ["unit", "largest"],
        "optional_concepts": ["level", "office", "restroom"],
        "map_action": "highlight_and_zoom",
        "response_type": "list",
    },
    "filter_units_by_category": {
        "endpoint": "/units/by-category",
        "target_layer": "Units",
        "required_concepts": ["unit"],
        "optional_concepts": ["office", "restroom"],
        "map_action": "highlight_and_zoom",
        "response_type": "list",
    },
    "filter_poi_by_category": {
        "endpoint": "/poi/by-category",
        "target_layer": "POI",
        "required_concepts": ["poi"],
        "optional_concepts": ["concession", "restroom", "elevator", "stair", "exit"],
        "map_action": "highlight_and_zoom",
        "response_type": "list",
    },
    "count_total_seats": {
        "endpoint": "/seats/count",
        "target_layer": "POI",
        "required_concepts": ["seat", "count"],
        "optional_concepts": [],
        "map_action": "none",
        "response_type": "summary",
    },
    "count_seats_by_section": {
        "endpoint": "/seats/count-by-section",
        "target_layer": "POI",
        "required_concepts": ["seat", "section", "count"],
        "optional_concepts": [],
        "map_action": "highlight",
        "response_type": "summary",
    },
    "count_seats_by_row": {
        "endpoint": "/seats/count-by-row",
        "target_layer": "POI",
        "required_concepts": ["seat", "row", "count"],
        "optional_concepts": [],
        "map_action": "highlight",
        "response_type": "summary",
    },
    "count_first_row_seats": {
        "endpoint": "/seats/count-first-row",
        "target_layer": "POI",
        "required_concepts": ["seat", "row", "count"],
        "optional_concepts": ["first"],
        "map_action": "highlight",
        "response_type": "summary",
    },
    "count_sections_by_level": {
        "endpoint": "/sections/count-by-level",
        "target_layer": "Sections",
        "required_concepts": ["section", "level", "count"],
        "optional_concepts": [],
        "map_action": "none",
        "response_type": "summary_table",
    },
    "count_seats_by_level": {
        "endpoint": "/seats/count-by-level",
        "target_layer": "POI",
        "required_concepts": ["seat", "level", "count"],
        "optional_concepts": [],
        "map_action": "none",
        "response_type": "summary_table",
    },
    "count_ada_seats_by_level": {
        "endpoint": "/seats/ada/count-by-level",
        "target_layer": "POI",
        "required_concepts": ["seat", "ada", "level", "count"],
        "optional_concepts": [],
        "map_action": "none",
        "response_type": "summary_table",
    },
    "compare_restroom_area": {
        "endpoint": "/units/restrooms/area-compare",
        "target_layer": "Units",
        "required_concepts": ["restroom", "compare"],
        "optional_concepts": ["unit", "level"],
        "map_action": "highlight_and_zoom",
        "response_type": "summary_table",
    },

    # Planned for later, not routed yet
    "nearest_facility": {
        "endpoint": None,
        "target_layer": None,
        "required_concepts": ["nearest"],
        "optional_concepts": ["restroom", "exit", "stair", "elevator"],
        "map_action": "route_and_highlight",
        "response_type": "route",
        "status": "planned",
    },
}


# ------------------------------------------------------------
# 3. Text normalization
# ------------------------------------------------------------

def normalize_question(question: str) -> str:
    """
    Cleans user question so different wording becomes easier to match.
    """
    text = question.lower().strip()

    # Remove punctuation except useful characters
    text = re.sub(r"[?.,!]", " ", text)

    # Normalize common phrases
    replacements = {
        "bathrooms": "restrooms",
        "bathroom": "restroom",
        "toilets": "restrooms",
        "toilet": "restroom",
        "washrooms": "restrooms",
        "washroom": "restroom",
        "floors": "levels",
        "floor": "level",
        "stories": "levels",
        "story": "level",
        "biggest": "largest",
        "closest": "nearest",
        "nearby": "nearest",
        "accessible": "ada",
        "wheelchair": "ada",
        "number of": "count",
        "how many": "count",
        "total number of": "count",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# 4. Concept detection
# ------------------------------------------------------------

def detect_concepts(normalized_question: str) -> Dict[str, bool]:
    """
    Detects which known concepts are present in the question.
    """
    concepts = {}

    for concept, words in SYNONYMS.items():
        concepts[concept] = False
        for word in words:
            # Word boundary check for safer matching
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, normalized_question):
                concepts[concept] = True
                break

    # Special case for "first row"
    concepts["first"] = bool(re.search(r"\bfirst\b", normalized_question))

    return concepts


# ------------------------------------------------------------
# 5. Parameter extraction
# ------------------------------------------------------------

def extract_limit(text: str) -> Optional[int]:
    """
    Extracts top N / first N / largest N style limits.
    """
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\blargest\s+(\d+)\b",
        r"\bshow\s+(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return None


def extract_level(text: str) -> Optional[str]:
    """
    Extracts level/floor values and normalizes to L format where possible.
    Examples:
    level 1 -> L1
    l2 -> L2
    floor 3 -> L3
    """
    patterns = [
        r"\blevel\s*([a-zA-Z0-9]+)\b",
        r"\bl\s*([0-9]+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).upper()
            if not value.startswith("L"):
                value = f"L{value}"
            return value

    return None


def extract_section(text: str) -> Optional[str]:
    """
    Extracts section number/name.
    Example:
    section 115 -> 115
    """
    match = re.search(r"\bsection\s+([a-zA-Z0-9\-]+)\b", text)
    if match:
        return match.group(1).upper()

    return None


def extract_row(text: str) -> Optional[str]:
    """
    Extracts row value.
    Example:
    row A -> A
    """
    match = re.search(r"\brow\s+([a-zA-Z0-9\-]+)\b", text)
    if match:
        return match.group(1).upper()

    return None


def extract_category(text: str, concepts: Dict[str, bool]) -> Optional[str]:
    """
    Extracts the likely category the user is asking about.
    """
    category_priority = [
        "restroom",
        "office",
        "concession",
        "elevator",
        "stair",
        "exit",
        "ada",
    ]

    for category in category_priority:
        if concepts.get(category):
            return category

    return None


def extract_generic_search_term(text: str) -> Optional[str]:
    """
    Extracts a generic term from simple search questions.
    Examples:
    show lockers -> lockers
    find mechanical rooms -> mechanical rooms
    list fire extinguishers -> fire extinguishers
    """
    patterns = [
        r"\bshow\s+all\s+(.+)$",
        r"\bshow\s+(.+)$",
        r"\bfind\s+all\s+(.+)$",
        r"\bfind\s+(.+)$",
        r"\blist\s+all\s+(.+)$",
        r"\blist\s+(.+)$",
        r"\bcount\s+all\s+(.+)$",
        r"\bcount\s+(.+)$",
    ]

    cleanup_words = [
        "on level",
        "in level",
        "on floor",
        "in floor",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            term = match.group(1).strip()

            for marker in cleanup_words:
                if marker in term:
                    term = term.split(marker)[0].strip()

            term = re.sub(r"\b(on|in)\s+l\d+\b", "", term).strip()
            term = re.sub(r"\b(on|in)\s+level\s+\d+\b", "", term).strip()

            return term

    return None


def extract_parameters(text: str, concepts: Dict[str, bool]) -> Dict[str, Any]:
    """
    Extracts all supported parameters.
    """
    search_term = extract_generic_search_term(text)

    generic_action = None
    if search_term:
        if concepts.get("count"):
            generic_action = "count"
        else:
            generic_action = "show"

    return {
        "limit": extract_limit(text) or 10,
        "level_id": extract_level(text),
        "section": extract_section(text),
        "row": extract_row(text),
        "category": extract_category(text, concepts),
        "search_term": search_term,
        "generic_action": generic_action,
    }
# ------------------------------------------------------------
# 6. Intent scoring
# ------------------------------------------------------------

def score_intent(intent_def: Dict[str, Any], concepts: Dict[str, bool]) -> float:
    """
    Scores an intent based on required and optional concept matches.
    More specific intents should beat generic intents.
    """
    required = intent_def.get("required_concepts", [])
    optional = intent_def.get("optional_concepts", [])

    if not required:
        return 0.0

    required_matches = sum(1 for c in required if concepts.get(c))
    optional_matches = sum(1 for c in optional if concepts.get(c))

    # Require ALL required concepts for active intents.
    # This prevents generic intents from winning too easily.
    if required_matches < len(required):
        return 0.0

    base_score = 0.80

    # Reward optional matches
    optional_score = 0.0
    if optional:
        optional_score = (optional_matches / len(optional)) * 0.10

    # Reward specificity. More required concepts = more specific intent.
    specificity_bonus = min(len(required) * 0.03, 0.12)

    final_score = base_score + optional_score + specificity_bonus

    return round(min(final_score, 0.99), 3)


def match_intent(concepts: Dict[str, bool]) -> Dict[str, Any]:
    """
    Finds the best matching intent.
    Includes fallback logic for simple category search questions.
    """

    # ------------------------------------------------------------
    # Fallback 1: simple unit category search
    # Examples:
    # show restrooms
    # show offices
    # ------------------------------------------------------------

        # ------------------------------------------------------------
    # Priority: seat count by row
    # Examples:
    # count seats in row A section 115
    # how many seats are in row B in section 210
    # ------------------------------------------------------------
    if concepts.get("seat") and concepts.get("row") and concepts.get("count"):
        return {
            "intent": "count_seats_by_row",
            "confidence": 0.93,
            "endpoint": "/seats/count-by-row",
            "target_layer": "POI",
            "map_action": "highlight",
            "response_type": "summary",
            "status": "active",
        }
        # ------------------------------------------------------------
    # Fallback 0: largest units by category
    # Examples:
    # top 5 biggest offices
    # largest restrooms
    # biggest office spaces
    # ------------------------------------------------------------
    if concepts.get("largest") and (concepts.get("office") or concepts.get("restroom")):
        return {
            "intent": "rank_units_by_area",
            "confidence": 0.88,
            "endpoint": "/units/top-largest",
            "target_layer": "Units",
            "map_action": "highlight_and_zoom",
            "response_type": "list",
            "status": "active",
        }
    
    if concepts.get("restroom") or concepts.get("office"):
        if not concepts.get("largest") and not concepts.get("compare") and not concepts.get("nearest"):
            return {
                "intent": "filter_units_by_category",
                "confidence": 0.82,
                "endpoint": "/units/by-category",
                "target_layer": "Units",
                "map_action": "highlight_and_zoom",
                "response_type": "list",
                "status": "active",
            }

    # ------------------------------------------------------------
    # Fallback 2: simple POI category search
    # Examples:
    # show concessions
    # show elevators
    # show stairs
    # show exits
    # ------------------------------------------------------------
    if concepts.get("concession") or concepts.get("elevator") or concepts.get("stair") or concepts.get("exit"):
        if not concepts.get("nearest"):
            return {
                "intent": "filter_poi_by_category",
                "confidence": 0.82,
                "endpoint": "/poi/by-category",
                "target_layer": "POI",
                "map_action": "highlight_and_zoom",
                "response_type": "list",
                "status": "active",
            }

    scored = []

    for intent_name, intent_def in INTENT_CATALOG.items():
        score = score_intent(intent_def, concepts)
        scored.append((intent_name, intent_def, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    best_name, best_def, best_score = scored[0]

    if best_score < 0.45:
        return {
            "intent": "unknown",
            "confidence": best_score,
            "endpoint": None,
            "target_layer": None,
            "map_action": "none",
            "response_type": "message",
        }

    return {
        "intent": best_name,
        "confidence": best_score,
        "endpoint": best_def.get("endpoint"),
        "target_layer": best_def.get("target_layer"),
        "map_action": best_def.get("map_action"),
        "response_type": best_def.get("response_type"),
        "status": best_def.get("status", "active"),
    }


# ------------------------------------------------------------
# 7. Validation
# ------------------------------------------------------------

def validate_intent(intent_result: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds clarification/failure messages for missing required parameters
    or planned features.
    """
    intent = intent_result.get("intent")

    if intent == "unknown":
        return {
            "is_valid": False,
            "needs_clarification": False,
            "message": (
                "I could not understand that yet. Try asking about largest rooms, "
                "restrooms, POIs, seats, sections, levels, or ADA seats."
            ),
        }

    if intent_result.get("status") == "planned":
        return {
            "is_valid": False,
            "needs_clarification": False,
            "message": (
                "Nearest facility routing is planned, but not connected yet. "
                "We will add this after the basic intent system is stable."
            ),
        }

    if intent == "count_seats_by_section" and not parameters.get("section"):
        return {
            "is_valid": False,
            "needs_clarification": True,
            "message": "Which section do you want me to check?",
        }

    if intent == "count_seats_by_row" and not parameters.get("row"):
        return {
            "is_valid": False,
            "needs_clarification": True,
            "message": "Which row do you want me to check?",
        }

    return {
        "is_valid": True,
        "needs_clarification": False,
        "message": "Intent matched successfully.",
    }


# ------------------------------------------------------------
# 8. Main parser function
# ------------------------------------------------------------

def parse_intent(question: str) -> Dict[str, Any]:
    """
    Main function used by FastAPI.
    """
    normalized_question = normalize_question(question)
    concepts = detect_concepts(normalized_question)
    parameters = extract_parameters(normalized_question, concepts)
    intent_result = match_intent(concepts)

    if parameters.get("search_term") and not parameters.get("category"):
        intent_result = {
            "intent": "generic_category_search",
            "confidence": 0.70,
            "endpoint": "/search/category",
            "target_layer": None,
            "map_action": "highlight_and_zoom",
            "response_type": "list",
            "status": "active",
        }

    validation = validate_intent(intent_result, parameters)

    return {
        "query": question,
        "normalized_query": normalized_question,
        "intent": intent_result.get("intent"),
        "confidence": intent_result.get("confidence"),
        "target_layer": intent_result.get("target_layer"),
        "endpoint": intent_result.get("endpoint"),
        "parameters": parameters,
        "response_type": intent_result.get("response_type"),
        "map_action": {
            "type": intent_result.get("map_action")
        },
        "validation": validation,
    }