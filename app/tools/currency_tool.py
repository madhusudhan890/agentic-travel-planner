"""
app/tools/currency_tool.py
───────────────────────────
Currency exchange rate tool for the travel agent.

Uses the Frankfurter API — completely FREE, no API key required.
https://www.frankfurter.app/

The LLM reads the @tool docstring and calls this automatically
when the user mentions a budget in a foreign currency context.
"""

import json
import urllib.request

from langchain_core.tools import tool

# Frankfurter API — free, no key needed
_FRANKFURTER_URL = "https://api.frankfurter.app/latest"


@tool
def get_exchange_rate(base_currency: str, target_currency: str, amount: float = 1.0) -> str:
    """Get the current exchange rate between two currencies to help with budget planning.

    Use this tool when:
    - The user mentions a budget (e.g. "$2000", "£500", "₹50000")
    - You need to convert the budget into the destination's local currency
    - The user explicitly asks about exchange rates or money conversion

    Args:
        base_currency: The currency the traveller has (e.g. "USD", "EUR", "GBP", "INR")
        target_currency: The currency of the destination (e.g. "JPY", "THB", "EUR", "USD")
        amount: The amount to convert (default is 1.0 for the base rate)

    Returns:
        A string showing the current exchange rate and converted amount.

    Examples:
        get_exchange_rate("USD", "JPY", 2000.0)
        → "$2000 USD = ¥298,400 JPY (rate: 1 USD = 149.20 JPY)"
    """
    try:
        url = f"{_FRANKFURTER_URL}?from={base_currency.upper()}&to={target_currency.upper()}&amount={amount}"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 AgenticTravelPlanner/1.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        rate = data["rates"][target_currency.upper()]
        return (
            f"{amount} {base_currency.upper()} = {rate:.2f} {target_currency.upper()} "
            f"(rate: 1 {base_currency.upper()} = {rate / amount:.4f} {target_currency.upper()})"
        )
    except Exception as exc:
        return (
            f"Could not fetch exchange rate for {base_currency} → {target_currency}: {exc}. "
            "Please check xe.com manually for current rates."
        )
