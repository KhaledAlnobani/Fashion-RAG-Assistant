# src/faq.py

import ollama
from . import config

from .tracing import trace_span
from opentelemetry import trace

def generate_faq_layout(faq_data: list[dict]) -> str:
    """
    Flattens all FAQ entries into one text block to inject as context.

    NOTE — known limitation: this dumps *all* FAQs into every prompt rather
    than doing real semantic retrieval (unlike products.py, which uses
    Weaviate). Fine at ~25 entries; if the FAQ set grows meaningfully, this
    should move to the same vector-search pattern as products.
    """
    lines = [
        f'question: {f["question"]} Answer: {f["answer"]} Type: {f["type"]}'
        for f in faq_data
    ]
    return "\n".join(lines)


@trace_span("query_on_faq", {"openinference.span.kind": "LLM", "llm.model_name": config.GENERATION_MODEL})
def query_on_faq(query: str, faq_layout: str) -> str:
    """
    Answers a query using the full FAQ context.
    """
    system_prompt = (
        "You are a professional FAQ question-answering assistant.\n\n"
        "Your job is to answer the user's question using the FAQ context."
    )

    user_prompt = f"""
Here is the FAQ context:

{faq_layout}

Here is the user's question:

{query}

Answer the user's question using only the FAQ context.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = ollama.chat(
        model=config.GENERATION_MODEL,
        messages=messages,
        options={
            "temperature": 0.0,
            "num_predict": 200,
        },
    )

    res = response.message.content.strip()
    current_span = trace.get_current_span()
    current_span.set_attributes({
        "input.value": query,
        "llm.token_count.prompt": response.get("prompt_eval_count", 0),
        "llm.token_count.completion": response.get("eval_count", 0),
        "output.value": res,
    })
    return res