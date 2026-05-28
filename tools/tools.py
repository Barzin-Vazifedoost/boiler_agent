"""Tool definitions for the agent(s).

Add your own @tool functions below.  The get_tools() helper returns the full
list so that worker and supervisor agents can import it in one place.
"""

import json
import os
import urllib.parse
import urllib.request

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information about a query.

    This is a lightweight example using DuckDuckGo Instant Answer API.
    Replace or extend with a proper search integration as needed.

    Args:
        query: The search query string.

    Returns:
        A short answer or abstract from the search results.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        answer = data.get("AbstractText") or data.get("Answer") or "No answer found."
        return answer
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO-8601 string.

    Returns:
        Current UTC datetime string.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a local file.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        File contents as a string, or an error message.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        return f"Could not read file: {exc}"


def get_tools():
    """Return the list of tools available to agents."""
    return [search_web, get_current_datetime, read_file]
