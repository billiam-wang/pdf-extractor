"""
Utility functions for parsing LLM responses.
"""

import json


def extract_first_text_response(content):
    """
    Extract the first non-reasoning text response from LLM output.

    Follows strict order:
    1. Filter out reasoning types
    2. Take first entry
    3. Validate it's a text entry
    4. Return only the text field

    Args:
        content: Raw LLM response content (string)

    Returns:
        Extracted text content or original content if parsing fails
    """
    try:
        parsed = json.loads(content)

        # Handle list of response items
        if isinstance(parsed, list):
            # Step 1: Filter out all reasoning type items
            non_reasoning_items = [
                item for item in parsed
                if isinstance(item, dict) and item.get('type') != 'reasoning'
            ]

            # Step 2: Take the first entry (if exists)
            if non_reasoning_items:
                first_entry = non_reasoning_items[0]

                # Step 3: Validate it's a text entry
                if first_entry.get('type') == 'text':
                    # Step 4: Return only the text field
                    return first_entry.get('text', '')

        # Handle single dict response (already filtered if not reasoning)
        elif isinstance(parsed, dict):
            if parsed.get('type') == 'text':
                return parsed.get('text', content)

        # If we couldn't parse structured format, return original
        return content
    except (json.JSONDecodeError, TypeError):
        # Not JSON, return as-is
        return content
