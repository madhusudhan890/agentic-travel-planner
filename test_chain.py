import asyncio
from app.models.request import TripRequest
from app.chain.gemini_chain import GeminiTravelChain

async def test_agent():
    print("Testing GeminiTravelChain initialization...")
    chain = GeminiTravelChain()
    
    print("Creating mock request...")
    request = TripRequest(
        destination="Tokyo, Japan",
        days=3,
        month="October",
        budget="1000 USD",
        interests="food, technology",
        travel_style="fast-paced"
    )
    
    print("Running plan_trip (this will trigger tools and LLM)...")
    try:
        # Give it a max of 45 seconds for testing
        result = await asyncio.wait_for(chain.plan_trip(request), timeout=45.0)
        print("\n=== SUCCESS ===")
        print(f"Result starts with: {result[:200]}...")
        print("===============")
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Failed: {type(e).__name__}: {e}")
        print("=============")

if __name__ == "__main__":
    asyncio.run(test_agent())
