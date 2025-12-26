def responses_text(response) -> str:
    if isinstance(response, str):
        return response

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        data = getattr(response, "__dict__", {})

    texts = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            t = c.get("text")
            if t:
                texts.append(t)

    return "\n".join(texts).strip()
