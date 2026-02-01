import json
from typing import Any


def maybe_parse_json(text: str) -> Any | None:
    """Parse JSON if possible.

    Args:
        text: Input string to parse.

    Returns:
        Decoded JSON value if valid, otherwise None.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from text.

    Args:
        text: Input JSON text.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the parsed JSON is not an object.
    """
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object (dict).")
    return value


def parse_int(text: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse an integer with optional bounds.

    Args:
        text: Input text containing an integer.
        minimum: Optional minimum allowed value.
        maximum: Optional maximum allowed value.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If the value violates bounds.
    """
    value = int(text.strip())
    if minimum is not None and value < minimum:
        raise ValueError(f"Expected int >= {minimum}; got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Expected int <= {maximum}; got {value}")
    return value
