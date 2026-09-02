# src/metadata_extraction.py

import json
import ollama
from . import config

from .tracing import trace_span
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

def build_metadata_schema(allowed_values: dict[str, list[str]]) -> dict:
    """
    Builds a JSON schema for metadata extraction, constraining each
    categorical field to only the values that actually exist in the
    product catalogue (via enum) — the model can no longer invent
    values like 'masterCategory: Dresses' that don't exist in the data.
    """
    def enum_array(field: str) -> dict:
        return {
            "type": "array",
            "items": {"type": "string", "enum": allowed_values[field]},
        }

    return {
        "type": "object",
        "properties": {
            "gender": enum_array("gender"),
            "masterCategory": enum_array("masterCategory"),
            "articleType": enum_array("articleType"),
            "baseColour": enum_array("baseColour"),
            "usage": enum_array("usage"),
            "season": enum_array("season"),
            "price": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": ["number", "string"]},
                },
                "required": ["min", "max"],
            },
        },
        "required": ["gender", "masterCategory", "articleType", "baseColour", "usage", "season", "price"],
    }


@trace_span("generate_metadata_from_query", {"openinference.span.kind": "LLM", "llm.model_name": config.METADATA_EXTRACTION_MODEL})
def generate_metadata_from_query(query: str, metadata_schema: dict) -> str:
    """
    Extracts structured product filters from a free-text query.
    `metadata_schema` is built once at startup (via build_metadata_schema)
    from the real catalogue values, and passed in — not rebuilt per call.
    """
    system_prompt = """Given a user query, extract product filters using ONLY the exact values allowed by the schema (the enum lists) — never invent a value that isn't in the allowed list.
- Gender: target audience.
- Master Category: broad classification.
- Article Type: exact product type.
- Base Colour: main color.
- Season: intended season.
- Usage: intended occasion.
- Price: cost range. If no price is mentioned, use min=0 and max="inf".

Only include values the query actually implies — don't guess."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"query:\n{query}"},
    ]

    response = ollama.chat(
        model=config.METADATA_EXTRACTION_MODEL,
        messages=messages,
        format=metadata_schema,
        options={
            "temperature": 0.1,
            "num_predict": 300,
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

@trace_span("parse_json_output", {"openinference.span.kind": "UTILITY"})
def parse_json_output(llm_output: str) -> dict | None:
    """Parses the LLM's JSON output — no string-replacement hacks needed
    since the schema enforces valid structure and values at the model level."""
    current_span = trace.get_current_span()
    current_span.set_attributes({"input.value": llm_output})
    try:
        current_span.set_attributes({"output.value": llm_output})
        return json.loads(llm_output)
    except json.JSONDecodeError as e:
        current_span.record_exception(e)
        current_span.set_status(Status(StatusCode.ERROR, str(e)))
        current_span.set_attributes({"error": str(e)})
        return None
