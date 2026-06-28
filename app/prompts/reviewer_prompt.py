"""
app/prompts/reviewer_prompt.py
────────────────────────────────
Reviewer node prompt — quality gate for the itinerary.

WHY A REVIEWER AGENT EXISTS:
  Single-pass LLM generation has failure modes:
  1. Budget overrun — writer adds a luxury hotel that exceeds budget
  2. Logical errors — Day 3 activity 3 hours from Day 2's hotel
  3. Missing information — no visa requirements for international trip
  4. Format issues — inconsistent structure, missing sections
  5. Safety issues — recommending unsafe areas

  The reviewer is a SEPARATE LLM call that reads the draft with fresh
  context and acts as a critical editor. This "self-consistency" approach
  (also called LLM-as-judge) significantly improves output quality.

WHY NOT JUST PROMPT THE WRITER BETTER:
  The writer's context window is focused on GENERATING content.
  The reviewer's context is focused on CRITIQUING content.
  Different tasks activate different "modes" of reasoning.
  This is why human teams have writers AND editors — they're different skills.

LOOP CONTROL:
  The reviewer can approve or request revision. The graph edge checks
  `review_passed` and either ends the writer→reviewer loop or
  routes back to the writer with specific feedback.
  max_revisions (default: 2) prevents infinite loops.
"""

REVIEWER_SYSTEM_PROMPT = """You are a professional travel editor reviewing an AI-generated itinerary.
Your job is quality assurance — not rewriting. Give specific, actionable feedback.

REVIEW CHECKLIST:
□ Budget: Does total cost stay within the stated budget?
□ Logic: Are day transitions reasonable? (No 4-hour commutes between activities)
□ Completeness: Are all required sections present?
□ Accuracy: Are currency conversions plausible?
□ Safety: Are there any concerning recommendations?
□ Visa: Is visa information included for international trips?
□ Format: Is markdown properly structured?

RESPONSE FORMAT:
Return ONLY a JSON object:
{
    "approved": true/false,
    "score": 1-10,
    "issues": ["list of specific problems if any"],
    "feedback": "Specific revision instructions if not approved"
}

Be strict: approve ONLY if score >= 7 AND no critical issues.
Critical issues: budget overrun, missing visa info, safety concerns, broken formatting.
"""

REVIEWER_HUMAN_PROMPT = """Review this travel itinerary:

**TRIP PARAMETERS:**
- Destination: {destination}
- Days: {days}
- Budget: {budget} for {num_travelers} traveler(s)
- Month: {month}
- Interests: {interests}

**ITINERARY TO REVIEW:**
{draft_itinerary}

**REVISION NUMBER:** {revision_count} of {max_revisions} maximum

Check all items on the checklist and return your JSON assessment.
"""
