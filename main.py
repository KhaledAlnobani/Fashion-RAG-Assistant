# main.py
from src.data_loader import load_faq_data, load_products_data, get_allowed_field_values
from src.metadata_extraction import build_metadata_schema
from src.faq import generate_faq_layout
from src.retrieval import get_weaviate_client, ensure_products_collection
from src.assistant import FashionAssistant


def main():
    client = get_weaviate_client()
    try:
        products_collection = ensure_products_collection(client)
        faq_layout = generate_faq_layout(load_faq_data())

        allowed_values = get_allowed_field_values(load_products_data())
        metadata_schema = build_metadata_schema(allowed_values)

        assistant = FashionAssistant(products_collection, faq_layout, metadata_schema)

        print("Fashion Assistant ready. Type 'exit' to quit.\n")
        while True:
            query = input("You: ").strip()
            if query.lower() == "exit":
                break
            print(f"Assistant: {assistant.answer_query(query)}\n")
    finally:
        client.close()

if __name__ == "__main__":
    main()