# src/generation.py

import ollama
from . import config
from .classification import decide_task_nature
from .metadata_extraction import generate_metadata_from_query, parse_json_output
from .retrieval import get_filter_by_metadata, get_relevant_products_from_query


def get_params_for_task(task: str) -> dict:
    """Maps a task label to generation parameters. Falls back to 'technical' (safer default)."""
    return config.GENERATION_PARAMS.get(task, config.GENERATION_PARAMS["technical"])


def generate_items_context(results: list) -> str:
    """Formats retrieved product objects into a compact text block for the LLM prompt."""
    lines = []
    for obj in results:
        p = obj.properties
        lines.append(
            f"Product ID: {p.get('product_id')}, "
            f"Product: {p.get('productDisplayName')}, "
            f"Price: {p.get('price')}, "
            f"Color: {p.get('baseColour')}, "
            f"Category: {p.get('articleType')}, "
            f"Gender: {p.get('gender')}, "
            f"Season: {p.get('season')}."
        )
    return "\n".join(lines)


def query_on_products(collection, metadata_schema: dict, query: str) -> str:
    """
    Full product-query pipeline: classify task nature -> pick generation params
    -> extract metadata filters -> retrieve -> generate answer.

    `collection` is passed in explicitly (from retrieval.ensure_products_collection),
    not fetched globally here — keeps this function testable with a mock collection.
    """
    task_label = decide_task_nature(query)
    params = get_params_for_task(task_label)

    raw_metadata = generate_metadata_from_query(query, metadata_schema)
    metadata = parse_json_output(raw_metadata)
    filters = get_filter_by_metadata(metadata)

    relevant_products = get_relevant_products_from_query(collection, query, filters)
    context = generate_items_context(relevant_products)

    system_prompt = (
        "Given the available set of cloth products, answer the question that follows, "
        "providing the item ID in your answers. Other information might be provided but not "
        "necessarily all of them; pick only the relevant ones for the given query and avoid "
        "being too long when describing the items' features. If no number of products is "
        "mentioned in the query, select at most five to show. Act as a helpful fashion assistant."
    )
    user_prompt = f"CLOTH PRODUCTS AVAILABLE:\n{context}\n\nQUERY: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = ollama.chat(
        model=config.GENERATION_MODEL,
        messages=messages,
        options={
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_predict": 300,
        },
    )
    return response.message.content.strip()