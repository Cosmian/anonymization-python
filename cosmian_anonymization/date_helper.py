from datetime import timezone

from dateutil import parser as date_parser

DURATION_IN_SECONDS = {
    "Second": 1,
    "Minute": 60,
    "Hour": 3600,
    "Day": 86400,
    "Month": 2_628_000,
    "Year": 31_536_000,
}


def date_to_rfc3339(date_str: str) -> str:
    """
    Converts a date string to ISO format with timezone (RFC 3339).

    Args:
        date_str (str): The input date string.

    Returns:
        str: The date string in RFC3339 format.
    """
    dt = date_parser.parse(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
