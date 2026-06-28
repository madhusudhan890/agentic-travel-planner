"""
app/tools/wikipedia_tool.py
────────────────────────────
Wikipedia tool for the travel agent.

The LLM reads the @tool docstring and decides when to call this.
No manual configuration needed — if the question is about a destination,
the agent will call this to get real, up-to-date info.
"""

import wikipedia
from langchain_core.tools import tool

@tool
def get_destination_info(destination: str) -> str:
    """Search Wikipedia for information about a travel destination.

    Use this tool when:
    - The user asks about a specific city, country, or landmark
    - You need accurate facts about local culture, history, or famous attractions
    - You want to give the traveller real context about where they are going

    Args:
        destination: The city, country, or landmark to search (e.g. "Kyoto Japan", "Eiffel Tower")

    Returns:
        A summary of key information about the destination from Wikipedia.
    """
    try:
        # Use the wikipedia python package directly
        result = wikipedia.summary(destination, sentences=5)
        if not result or result.strip() == "":
            return f"No Wikipedia information found for '{destination}'. Proceeding with built-in knowledge."
        return result
    except Exception as exc:
        return f"Wikipedia search failed for '{destination}': {exc}. Proceeding with built-in knowledge."
