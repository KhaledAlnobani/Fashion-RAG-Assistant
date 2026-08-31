# src/retrieval.py

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from . import config


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Opens and returns a Weaviate client connection.
    Caller is responsible for closing it (client.close()) when done —
    use as a context manager where possible.
    """
    return weaviate.connect_to_local(
        port=config.WEAVIATE_PORT,
        grpc_port=config.WEAVIATE_GRPC_PORT,
    )


def ensure_products_collection(client: weaviate.WeaviateClient):
    """
    Creates the 'products' collection if it doesn't exist, otherwise
    returns the existing one. Schema definition lives here, not in a notebook.
    """
    if client.collections.exists(config.PRODUCTS_COLLECTION_NAME):
        return client.collections.get(config.PRODUCTS_COLLECTION_NAME)

    return client.collections.create(
        name=config.PRODUCTS_COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            model=config.EMBEDDING_MODEL,
            api_endpoint=config.OLLAMA_ENDPOINT,
            vectorize_collection_name=False,
        ),
        properties=[
            Property(name="productDisplayName", data_type=DataType.TEXT),
            Property(name="articleType", data_type=DataType.TEXT),
            Property(name="baseColour", data_type=DataType.TEXT),
            Property(name="usage", data_type=DataType.TEXT),
            Property(name="season", data_type=DataType.TEXT),
            Property(name="price", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="product_id", data_type=DataType.INT, skip_vectorization=True),
            Property(name="year", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="gender", data_type=DataType.TEXT, skip_vectorization=True),
        ],
    )


def get_filter_by_metadata(json_output: dict | None):
    """Builds a list of Weaviate Filter objects from extracted metadata."""
    if json_output is None:
        return None

    valid_keys = ("gender", "masterCategory", "articleType", "baseColour", "price", "usage", "season")
    filters = []

    for key, value in json_output.items():
        if key not in valid_keys:
            continue

        if key == "price":
            if not isinstance(value, dict):
                continue
            min_price, max_price = value.get("min"), value.get("max")
            if min_price is None or max_price is None:
                continue
            if min_price > 0:
                filters.append(Filter.by_property(key).greater_or_equal(min_price))
            if max_price != "inf":
                filters.append(Filter.by_property(key).less_or_equal(max_price))
        else:
            if not value:  
                continue
            filters.append(Filter.by_property(key).contains_any(value))

    return filters


def get_relevant_products_from_query(collection, query: str, filters: list | None, limit: int = 20):
    """
    Runs near_text search with progressive filter relaxation if too few results.
    `collection` and `filters` are passed in — no globals, no hidden dependency
    on metadata_extraction.py (keeps retrieval.py testable on its own).
    """
    if not filters:
        return collection.query.near_text(query=query, limit=limit).objects

    response = collection.query.near_text(query=query, limit=limit, filters=Filter.all_of(filters)).objects
    if len(response) >= 10:
        return response

    importance_order = ["gender", "masterCategory", "articleType", "price", "baseColour", "season", "usage"]

    for i in range(1, len(importance_order)):
        keys_to_keep = importance_order[:-i]
        current_filters = [f for f in filters if f.target in keys_to_keep]
        if not current_filters:
            break
        response = collection.query.near_text(query=query, limit=limit, filters=Filter.all_of(current_filters)).objects
        if len(response) >= 5:
            return response

    return collection.query.near_text(query=query, limit=limit).objects