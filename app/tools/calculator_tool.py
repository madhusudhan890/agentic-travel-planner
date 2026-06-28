"""
app/tools/calculator_tool.py
─────────────────────────────
Budget calculator — pure Python, no API needed.

WHY A CALCULATOR TOOL EXISTS:
  LLMs are notoriously bad at math. Giving them a calculator tool
  ensures precise budget calculations rather than "approximately $X" guesses.

  This is the simplest example of function calling:
  the LLM decides to call the calculator with numbers it extracted,
  gets back exact results, and quotes them accurately.

  Production pattern: Always provide a calculator for numeric operations.
  Never let the LLM "reason" through arithmetic — it makes errors.
"""

from langchain_core.tools import tool


@tool
def calculate_trip_budget(
    total_budget: float,
    num_travelers: int,
    num_days: int,
    flight_cost: float = 0.0,
    hotel_cost_per_night: float = 0.0,
    food_cost_per_day: float = 0.0,
    activities_per_day: float = 0.0,
    transport_per_day: float = 0.0,
    emergency_buffer_pct: float = 10.0,
) -> str:
    """Calculate a detailed trip budget breakdown and daily allowance.

    Use this tool to compute exact budget figures when the user provides
    specific amounts. Always call this to ensure budget arithmetic is correct.

    Args:
        total_budget: Total budget in user's currency
        num_travelers: Number of travelers
        num_days: Trip duration in days
        flight_cost: Total flight cost for all travelers (0 if not flying)
        hotel_cost_per_night: Per-room cost per night
        food_cost_per_day: Daily food budget per person
        activities_per_day: Daily activities/entrance fees per person
        transport_per_day: Daily local transport per person
        emergency_buffer_pct: Emergency fund % (recommended: 10-15%)

    Returns:
        Detailed budget breakdown with daily allowance and analysis.
    """
    try:
        # Core calculations
        accommodation_total = hotel_cost_per_night * num_days
        food_total = food_cost_per_day * num_travelers * num_days
        activities_total = activities_per_day * num_travelers * num_days
        transport_total = transport_per_day * num_travelers * num_days

        subtotal = flight_cost + accommodation_total + food_total + activities_total + transport_total
        emergency_fund = (subtotal * emergency_buffer_pct) / 100
        grand_total = subtotal + emergency_fund

        remaining = total_budget - grand_total
        is_within_budget = remaining >= 0

        budget_per_person = total_budget / num_travelers if num_travelers > 0 else total_budget
        daily_total = (subtotal / num_days) if num_days > 0 else 0

        # Per-person costs
        flight_pp = flight_cost / num_travelers if num_travelers > 0 else flight_cost
        accommodation_pp = accommodation_total / num_travelers if num_travelers > 0 else accommodation_total

        status_emoji = "✅" if is_within_budget else "⚠️"
        surplus_deficit = f"+{remaining:.0f} surplus" if remaining >= 0 else f"{remaining:.0f} OVER BUDGET"

        result = f"""## 💰 Trip Budget Analysis

**Total Budget:** ${total_budget:,.0f} | **Per Person:** ${budget_per_person:,.0f}
**Travelers:** {num_travelers} | **Duration:** {num_days} days
**Status:** {status_emoji} {surplus_deficit}

### Cost Breakdown (All Travelers)
| Category | Total | Per Person | Per Day |
|----------|-------|-----------|---------|
| ✈️ Flights | ${flight_cost:,.0f} | ${flight_pp:,.0f} | — |
| 🏨 Accommodation | ${accommodation_total:,.0f} | ${accommodation_pp:,.0f} | ${hotel_cost_per_night:,.0f}/night |
| 🍜 Food | ${food_total:,.0f} | ${food_cost_per_day * num_days:,.0f} | ${food_cost_per_day:,.0f}/person |
| 🎯 Activities | ${activities_total:,.0f} | ${activities_per_day * num_days:,.0f} | ${activities_per_day:,.0f}/person |
| 🚌 Transport | ${transport_total:,.0f} | ${transport_per_day * num_days:,.0f} | ${transport_per_day:,.0f}/person |
| 🆘 Emergency ({emergency_buffer_pct:.0f}%) | ${emergency_fund:,.0f} | ${emergency_fund/num_travelers:,.0f} | — |
| **GRAND TOTAL** | **${grand_total:,.0f}** | **${grand_total/num_travelers:,.0f}** | **${daily_total:,.0f}** |

### Daily Budget Guide
- **Daily spending money per person:** ${(food_cost_per_day + activities_per_day + transport_per_day):,.0f}
- **Total daily cost (all):** ${daily_total:,.0f}
- **Non-flight budget:** ${total_budget - flight_cost:,.0f}
"""

        if not is_within_budget:
            over = abs(remaining)
            result += f"""
### ⚠️ Budget Adjustment Needed (${over:,.0f} over)
Suggestions to reduce costs:
1. **Accommodation:** Switch to budget/hostel options (save ${accommodation_total * 0.4:,.0f})
2. **Food:** More street food, fewer restaurant meals (save ${food_total * 0.3:,.0f})
3. **Activities:** Focus on free attractions — parks, markets, temples
4. **Transport:** Use public transit instead of taxis (save ${transport_total * 0.5:,.0f})
"""
        else:
            result += f"\n### ✅ Budget Status: ${remaining:,.0f} remaining — comfortable margin!\n"

        return result

    except ZeroDivisionError:
        return "⚠️ Calculator error: num_travelers cannot be 0"
    except Exception as exc:
        return f"⚠️ Budget calculation error: {exc}"


@tool
def convert_currency_amount(amount: float, from_currency: str, rate_to_target: float, target_currency: str) -> str:
    """Convert a currency amount using a provided exchange rate.

    Use this AFTER getting the exchange rate from get_exchange_rate tool.
    The LLM calls get_exchange_rate first, then uses this for calculations.

    Args:
        amount: Amount in source currency
        from_currency: Source currency code (e.g., "USD")
        rate_to_target: Exchange rate (units of target per 1 source)
        target_currency: Target currency code (e.g., "JPY")

    Returns:
        Formatted conversion result.
    """
    converted = amount * rate_to_target
    return (
        f"{amount:,.2f} {from_currency} = {converted:,.2f} {target_currency} "
        f"(at rate: 1 {from_currency} = {rate_to_target:.4f} {target_currency})"
    )
