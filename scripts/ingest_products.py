# scripts/ingest_products.py

"""
One-time (or on-data-change) script to build the Weaviate 'products' collection
and populate it from the raw joblib dataset. This is intentionally separate
from the app's runtime logic (src/) — you run this manually, not on every
app startup.

Usage:
    python -m scripts.ingest_products
"""

import tqdm

from src.data_loader import load_products_data, clean_products_data
from src.retrieval import get_weaviate_client, ensure_products_collection


def main():
    client = get_weaviate_client()
    try:
        collection = ensure_products_collection(client)

        raw_products = load_products_data()
        cleaned_products = clean_products_data(raw_products)

        print(f"Ingesting {len(cleaned_products)} products...")

        with collection.batch.dynamic() as batch:
            for record in tqdm.tqdm(cleaned_products):
                batch.add_object(properties=record)

        failed = collection.batch.failed_objects
        if failed:
            print(f"{len(failed)} objects failed to ingest.")
            print("First failure:", failed[0].message)
        else:
            print("Ingestion completed successfully.")

        print(f"Total objects in collection: {len(collection)}")

    finally:
        client.close()


if __name__ == "__main__":
    main()