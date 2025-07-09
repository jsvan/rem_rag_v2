"""Tests for the LLM client"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.openai_client import LLMClient, StructuredLLMClient


async def test_llm_basic():
    """Test basic LLM operations"""
    client = LLMClient()
    
    # Test token counting
    text = "The Cuban Missile Crisis was a pivotal moment."
    tokens = client.count_tokens(text)
    assert tokens > 0
    print(f"Token count for '{text}': {tokens}")
    
    # Test single generation
    response = await client.generate(
        "What year was the Cuban Missile Crisis? Reply with just the year.",
        temperature=0
    )
    print(f"Single generation: {response}")
    assert "1962" in response
    
    # Test batch generation
    prompts = [
        "Name one key figure in the Cuban Missile Crisis. Just the name.",
        "What superpower was involved besides the US? Just the country name."
    ]
    responses = await client.batch_generate(prompts, temperature=0, max_concurrent=2)
    print(f"Batch generation: {responses}")
    assert len(responses) == 2
    
    # Test stats
    stats = client.get_stats()
    print(f"Usage stats: {stats}")
    assert stats["total_requests"] >= 3


async def test_structured_extraction():
    """Test structured entity extraction"""
    client = StructuredLLMClient()
    
    text = """
    In October 1962, Khrushchev made the fateful decision to place nuclear missiles 
    in Cuba. This brought the Soviet Union and United States to the brink of war.
    President Kennedy responded with a naval blockade.
    """
    
    entities = await client.extract_entities(
        text,
        "Extract key entities and what we learn about them from this text."
    )
    
    print(f"Extracted entities: {entities}")
    
    # Check we got some entities
    assert len(entities) > 0
    assert all("entity" in e and "learning" in e for e in entities)
    
    # Should find at least Khrushchev or Kennedy
    entity_names = [e["entity"] for e in entities]
    assert any(name in str(entity_names) for name in ["Khrushchev", "Kennedy", "Soviet", "Cuba"])


async def main():
    """Run all tests"""
    print("Testing basic LLM functionality...")
    await test_llm_basic()
    
    print("\nTesting structured extraction...")
    await test_structured_extraction()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())