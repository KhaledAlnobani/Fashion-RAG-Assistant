# src/data_loader.py
import joblib
from . import config


def load_products_data(path: str = config.PRODUCTS_DATA_PATH) -> list[dict]:
    """Loads the raw product records from the joblib file."""
    return joblib.load(path)


def load_faq_data(path: str = config.FAQ_DATA_PATH) -> list[dict]:
    """Loads the raw FAQ records from the joblib file."""
    return joblib.load(path)


def clean_record(record: dict) -> dict:
    """
    Normalizes a single raw product record before inserting into Weaviate:
    - blank/None/'nan' text fields become None
    - price/product_id are safely cast, defaulting to None on failure

    Uses .get() everywhere so a missing key doesn't raise KeyError and
    kill the whole ingestion batch over one bad record.
    """
    cleaned = {}

    text_fields = [
        "gender", "masterCategory", "subCategory", "articleType",
        "usage", "season", "productDisplayName", "year", "baseColour",
    ]

    for field in text_fields:
        val = record.get(field)
        if val is None or str(val).strip() == "" or str(val).lower() == "nan":
            cleaned[field] = None
        else:
            cleaned[field] = str(val).strip()

    price_val = record.get("price")
    try:
        cleaned["price"] = None if price_val is None or str(price_val).strip() == "" else float(price_val)
    except (ValueError, TypeError):
        cleaned["price"] = None

    id_val = record.get("product_id")
    try:
        cleaned["product_id"] = None if id_val is None or str(id_val).strip() == "" else int(id_val)
    except (ValueError, TypeError):
        cleaned["product_id"] = None

    return cleaned


def clean_products_data(products: list[dict]) -> list[dict]:
    """Applies clean_record to a full list of raw product records."""
    return [clean_record(p) for p in products]



def get_allowed_field_values(products: list[dict]) -> dict[str, list[str]]:
    """
    Computes the real set of allowed values per categorical field from the
    actual product catalogue. Used to constrain LLM metadata extraction via
    JSON schema enums, so the model can't invent values that don't exist
    in the data (e.g. 'masterCategory: Dresses' when the real value is
    'Apparel', or 'Blue Dresses' as an articleType that was never in the data).
    """
    fields = ("gender", "masterCategory", "articleType", "baseColour", "usage", "season")
    values: dict[str, set] = {field: set() for field in fields}

    for record in products:
        for field in fields:
            val = record.get(field)
            if val is not None and str(val).strip() != "" and str(val).lower() != "nan":
                values[field].add(str(val).strip())

    return {field: sorted(v) for field, v in values.items()}