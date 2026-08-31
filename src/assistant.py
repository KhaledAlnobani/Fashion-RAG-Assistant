# src/assistant.py

import ollama
from . import config
from .classification import check_if_faq_or_product
from .faq import query_on_faq
from .generation import query_on_products


class FashionAssistant:
    """
    Orchestrates the full pipeline: classify query -> route to FAQ / Product /
    OTHER handler -> maintain short chat history.

    Dependencies (Weaviate collection, FAQ layout) are injected via __init__
    rather than built internally or read from globals — keeps this class
    testable with mocks and makes startup wiring explicit in main.py.
    """

    def __init__(self, products_collection, faq_layout: str, metadata_schema: dict):
        self.products_collection = products_collection
        self.faq_layout = faq_layout
        self.metadata_schema = metadata_schema
        self.chat_memory: list[dict] = []

    def _answer_other(self, query: str) -> str:
        system_prompt = (
            "You are a helpful assistant. The user provided a question that does not fit "
            "FAQ or Product related questions. Answer it based on the context you already have so far."
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.chat_memory)
        messages.append({"role": "user", "content": f"user question {query}"})

        response = ollama.chat(
            model=config.GENERATION_MODEL,
            messages=messages,
            options={"temperature": 0.7, "num_predict": 300},
        )
        return response.message.content.strip()

    def answer_query(self, query: str) -> str:
        label = check_if_faq_or_product(query)

        if label == "FAQ":
            final_response = query_on_faq(query, self.faq_layout)

        elif label == "Product":
            try:
                final_response = query_on_products(self.products_collection, self.metadata_schema, query)
            except Exception as e:
                print(f"query_on_products failed: {e}")  
                system_prompt = (
                    "User provided a question that broke the querying system. Instruct them "
                    "to rephrase it. Answer it based on the context you already have so far."
                )
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.chat_memory)
                messages.append({"role": "user", "content": f"User question {query}"})

                response = ollama.chat(
                    model=config.GENERATION_MODEL,
                    messages=messages,
                    options={"temperature": 0.7, "num_predict": 300},
                )
                final_response = response.message.content.strip()

        else:
            final_response = self._answer_other(query)

        self.chat_memory.append({"role": "user", "content": query})
        self.chat_memory.append({"role": "assistant", "content": final_response})
        if len(self.chat_memory) > config.MAX_CHAT_MEMORY_TURNS:
            self.chat_memory = self.chat_memory[-config.MAX_CHAT_MEMORY_TURNS:]

        return final_response