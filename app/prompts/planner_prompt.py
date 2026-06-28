"""
app/prompts/planner_prompt.py
──────────────────────────────
Planner node prompt — extracts structured trip intent from user input.

WHY THIS PROMPT EXISTS:
  The planner is the "brain" that converts a free-form request
  ("I want to go to Japan in spring with $3000 for 10 days, I love sushi")
  into a structured TripPlan that all other agents can use.

  Without this extraction step:
  - "5 nights" vs "5 days" would confuse every downstream agent
  - "$3000 total" vs "$3000/person" would cause budget errors
  - Implicit interests wouldn't be surfaced to relevant agents

STRUCTURED OUTPUTS CONCEPT:
  We use Pydantic schemas + LangChain's `with_structured_output()` here.
  The LLM is forced to return JSON matching our schema — not free text.
  This eliminates parsing errors and makes downstream code predictable.

  Production alternatives:
  - OpenAI: function_call parameter with JSON schema
  - Gemini: response_schema parameter
  - Anthropic: tool_use API
  All do the same thing: constrain LLM output to a specific JSON schema.
"""

PLANNER_SYSTEM_PROMPT = """You are a travel planning data extractor.
Your ONLY job is to extract structured trip information from user input.

Extract exactly what the user says. Do not add assumptions.
If information is missing, use the provided defaults.

Be precise about:
- Budget: distinguish total vs per-person, extract currency symbol
- Duration: convert "a week" → 7 days, "long weekend" → 3 days
- Interests: extract as a list of specific topics
- Travel style: map to one of [backpacker, budget, moderate, luxury, ultra-luxury]
"""

PLANNER_HUMAN_PROMPT = """Extract structured trip plan from this request:

Destination: {destination}
Days: {days}
Month: {month}
Budget: {budget}
Interests: {interests}
Travel Style: {travel_style}
Number of Travelers: {num_travelers}
Departure City: {departure_city}

Determine which specialist agents to activate:
- "flight": if departure_city is provided OR days > 1 and destination is international
- "hotel": always (unless day trip)
- "weather": always (affects activity planning)
- "restaurant": if interests include food/dining OR any trip
- "visa": if international travel is likely
- "currency": if destination uses a different currency
- "budget": always (budget breakdown is always useful)

Return which agents to activate and why.
"""
