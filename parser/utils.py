"""
Utility functions for parsing LLM responses.
"""

import json


def extract_first_text_response(content):
    if isinstance(content, list):
        non_reasoning_items = [
            item for item in content
            if isinstance(item, dict) and item.get('type') != 'reasoning'
        ]

        if non_reasoning_items:
            first_entry = non_reasoning_items[0]

            if first_entry.get('type') == 'text':
                # Step 4: Return only the text field
                return first_entry.get('text', '')

    elif isinstance(content, dict):
        if content.get('type') == 'text':
            return content.get('text', content)

    return f"Error extracting content from: {content}"
