# src/classification.py

import ollama
from . import config

VALID_QUERY_LABELS = ("FAQ", "Product", "OTHER")
VALID_TASK_LABELS = ("creative", "technical")


def check_if_faq_or_product(query: str) -> str:
    """
    Classifies a query into FAQ / Product / OTHER.
    Falls back to 'OTHER' if the model returns anything outside the
    three allowed labels — never trust raw LLM text as a control-flow value.
    """
    system_prompt = (
        'You are a strict text classification API. Classify the user\'s query into '
        'exactly one of these three classes: "FAQ", "Product", or "OTHER".\n\n'
        'You must output ONLY a single word representing the class and nothing else.\n\n'
        '- Use "FAQ" for policies, instructions, support, or general system questions.\n'
        '- Use "Product" for item availability, prices, recommendations, comparisons, '
        'or styling/outfit requests — including creative outfit or "look" requests.\n'
        '- Use "OTHER" for greetings, general conversation, chit-chat, or topics unrelated '
        'to fashion products and company policies.'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = ollama.chat(
        model=config.INTENT_CLASSIFICATION_MODEL,
        messages=messages,
        # think=False,  
        options={
            "temperature": 0.1,
            "num_predict": 700, 
        },
    )

    label = response.message.content.strip()
    return label if label in VALID_QUERY_LABELS else "OTHER"


def decide_task_nature(query: str) -> str:
    """
    Classifies a query as 'creative' or 'technical', to drive generation
    parameters downstream. Defaults to 'technical' on any unexpected output —
    the safer/cheaper failure mode (lower temperature, more deterministic).
    """
    system_prompt = """Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices, etc.).
Label it as creative or technical.

Examples:
Query: Give me suggestions on a nice look for a nightclub.
Label: creative

Query: What are the blue dresses you have available?
Label: technical

Query: Give me three T-shirts for summer.
Label: technical

Query: Give me a look for attending a wedding party.
Label: creative

Only output one token: the label."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"question:\n\n{query}"},
    ]

    response = ollama.chat(
        model=config.INTENT_CLASSIFICATION_MODEL,
        messages=messages,
        # think=False,
        options={
            "temperature": 0.1,
            "num_predict": 700,
        },
    )

    label = response.message.content.strip().lower()
    return label if label in VALID_TASK_LABELS else "technical"