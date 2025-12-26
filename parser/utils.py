"""
Utility functions for parsing LLM responses.
"""

import json

def responses_text(response) -> str:
    # If some wrapper returns a plain string
    if isinstance(response, str):
        return response

    # Pydantic model -> dict
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        # last resort
        data = getattr(response, "__dict__", {})

    texts = []
    for item in data.get("output", []) or []:
        # Most common: {"type":"message","content":[{"type":"output_text","text":"..."}]}
        for c in item.get("content", []) or []:
            t = c.get("text")
            if t:
                texts.append(t)

        # Sometimes text is nested differently; add more cases if needed.

    return "\n".join(texts).strip()
