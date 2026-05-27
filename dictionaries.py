UNIT_CATEGORY_MAP = {
    "restroom": ["RESTROOMS"],
    "restrooms": ["RESTROOMS"],
    "toilet": ["RESTROOMS"],
    "toilets": ["RESTROOMS"],

    "concession": ["CONCESSIONS", "FOOD AND SERVICES"],
    "concessions": ["CONCESSIONS", "FOOD AND SERVICES"],
    "food": ["CONCESSIONS", "FOOD AND SERVICES"],
    "food area": ["CONCESSIONS", "FOOD AND SERVICES"],
    "food areas": ["CONCESSIONS", "FOOD AND SERVICES"],

    "ticket booth": ["TICKET BOOTH"],
    "ticket booths": ["TICKET BOOTH"],
    "ticket office": ["TICKET BOOTH"],
    "ticket offices": ["TICKET BOOTH"],

    "office": ["OFFICE"],
    "offices": ["OFFICE"],

    "storage": ["STORAGE"],
    "storages": ["STORAGE"],

    "suite": ["SUITES"],
    "suites": ["SUITES"],

    "club": ["CLUB"],
    "clubs": ["CLUB"],
}

POI_CATEGORY_MAP = {
    "exit": ["EXITS"],
    "exits": ["EXITS"],

    "entrance": ["ENTRANCES"],
    "entrances": ["ENTRANCES"],

    "stair": ["STAIRS"],
    "stairs": ["STAIRS"],

    "elevator": ["ELEVATOR"],
    "elevators": ["ELEVATOR"],

    "concession": ["CONCESSIONS"],
    "concessions": ["CONCESSIONS"],

    "restroom": ["RESTROOMS"],
    "restrooms": ["RESTROOMS"],

    "parking": ["PARKING"],
}

def normalize_unit_categories(category_input: str):
    if not category_input:
        return None

    category_input = category_input.strip().lower()

    if category_input in UNIT_CATEGORY_MAP:
        return UNIT_CATEGORY_MAP[category_input]

    return [category_input.upper()]

def normalize_poi_categories(category_input: str):
    if not category_input:
        return None

    category_input = category_input.strip().lower()

    if category_input in POI_CATEGORY_MAP:
        return POI_CATEGORY_MAP[category_input]

    return [category_input.upper()]