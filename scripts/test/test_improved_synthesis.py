#!/usr/bin/env python3
"""
Test improved synthesis generation that focuses on insights rather than meta-analysis.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import LLMClient
from src.config import LLM_MODEL


# Current problematic prompt
CURRENT_SYNTHESIS_PROMPT = """New information: {new_info}

Existing knowledge: {existing_knowledge}

Write exactly 1 short paragraph (50-100 words) explaining how this new information relates to existing knowledge. Focus on:
- Key similarities or contradictions
- How it extends or challenges what we know
- The most important new insight (if any)

Be concise and specific. Keep your response to a single, impactful paragraph.

If this adds nothing new or is simply a repetition of what we already know, respond with exactly: NOTHING"""


# Improved prompt that focuses on the actual insight
IMPROVED_SYNTHESIS_PROMPT = """New information: {new_info}

Existing knowledge: {existing_knowledge}

Based on comparing these passages, state the key insight or pattern that emerges. Write exactly 1 paragraph (50-100 words) that:

- States the actual pattern/insight directly (not how passages "relate")
- Uses specific terms and concepts from the passages
- Focuses on what we learn about the topic itself
- Avoids meta-language like "reinforces", "extends", "highlights"

Example good synthesis: "Humanitarian interventions create moral hazard by encouraging weaker parties to escalate conflicts expecting foreign support, while simultaneously nations hesitate to deploy ground troops, resulting in delayed interventions that fail to prevent atrocities like Rwanda."

If this adds nothing new or is simply repetition, respond with exactly: NOTHING"""


# Alternative prompt focusing on contradiction/tension
ALTERNATIVE_SYNTHESIS_PROMPT = """New information: {new_info}

Existing knowledge: {existing_knowledge}

Synthesize what these passages reveal about the topic. In exactly 1 paragraph (50-100 words):

- State the core insight, pattern, or contradiction discovered
- Use the specific language and terms from the passages
- Focus on the subject matter, not the relationship between texts
- Write as if explaining the pattern to someone unfamiliar with these sources

Begin directly with the insight (e.g., "Economic sanctions often...", "Nuclear deterrence requires...", "Great powers maintain influence through...")

If this adds nothing new, respond with exactly: NOTHING"""


async def test_synthesis_approaches():
    """Test different synthesis approaches with real examples."""
    
    # Initialize LLM client
    llm_client = LLMClient(model=LLM_MODEL)
    
    # Test case: Humanitarian intervention
    new_info = """The text highlights a debate among scholars about the effectiveness of 
    humanitarian interventions, with some arguing that such actions can prolong conflicts 
    rather than resolve them."""
    
    existing_knowledge = """Humanitarian problems are rarely only humanitarian problems; 
    the taking of life or withholding of food is almost always a political act. If the 
    United States is not prepared to address the underlying political conflict and to know 
    whose side it is on, the military may end up separating warring parties for an 
    indefinite period."""
    
    print("🔬 Testing Synthesis Approaches")
    print("=" * 80)
    print("\nTest Case: Humanitarian Intervention")
    print("-" * 40)
    print(f"New Info: {new_info}")
    print(f"\nExisting: {existing_knowledge}")
    
    # Test current approach
    print("\n\n📝 CURRENT APPROACH")
    print("-" * 40)
    current_synthesis = await llm_client.generate(
        prompt=CURRENT_SYNTHESIS_PROMPT.format(
            new_info=new_info,
            existing_knowledge=existing_knowledge
        ),
        temperature=0.7,
        max_tokens=200
    )
    print(current_synthesis)
    
    # Test improved approach
    print("\n\n✨ IMPROVED APPROACH")
    print("-" * 40)
    improved_synthesis = await llm_client.generate(
        prompt=IMPROVED_SYNTHESIS_PROMPT.format(
            new_info=new_info,
            existing_knowledge=existing_knowledge
        ),
        temperature=0.7,
        max_tokens=200
    )
    print(improved_synthesis)
    
    # Test alternative approach
    print("\n\n🔄 ALTERNATIVE APPROACH")
    print("-" * 40)
    alternative_synthesis = await llm_client.generate(
        prompt=ALTERNATIVE_SYNTHESIS_PROMPT.format(
            new_info=new_info,
            existing_knowledge=existing_knowledge
        ),
        temperature=0.7,
        max_tokens=200
    )
    print(alternative_synthesis)
    
    # Test case 2: Economic/financial
    print("\n\n" + "=" * 80)
    print("Test Case 2: Financial Markets")
    print("-" * 40)
    
    new_info2 = """Hedge funds and speculative investments are often blamed for destabilizing 
    emerging economies, but they are not the primary cause of the volatility in global capital flows."""
    
    existing_knowledge2 = """International bank lending to emerging markets has often been more 
    volatile than portfolio investments in equities and bonds -- because the vast majority of 
    bank lending has taken the form of short-term loans between banks, not long-term project financing."""
    
    print(f"New Info: {new_info2}")
    print(f"\nExisting: {existing_knowledge2}")
    
    print("\n\n📝 CURRENT APPROACH")
    print("-" * 40)
    current_synthesis2 = await llm_client.generate(
        prompt=CURRENT_SYNTHESIS_PROMPT.format(
            new_info=new_info2,
            existing_knowledge=existing_knowledge2
        ),
        temperature=0.7,
        max_tokens=200
    )
    print(current_synthesis2)
    
    print("\n\n✨ IMPROVED APPROACH")  
    print("-" * 40)
    improved_synthesis2 = await llm_client.generate(
        prompt=IMPROVED_SYNTHESIS_PROMPT.format(
            new_info=new_info2,
            existing_knowledge=existing_knowledge2
        ),
        temperature=0.7,
        max_tokens=200
    )
    print(improved_synthesis2)
    
    # Analyze keyword presence
    print("\n\n📊 KEYWORD ANALYSIS")
    print("=" * 80)
    
    def count_keywords(text, keywords):
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword.lower() in text_lower)
    
    meta_keywords = ["reinforces", "extends", "highlights", "new information", "existing knowledge", 
                     "relates to", "builds upon", "aligns with", "suggests that"]
    
    topic_keywords1 = ["humanitarian", "intervention", "political", "conflict", "military", "prolong"]
    topic_keywords2 = ["hedge funds", "bank", "lending", "volatile", "capital", "emerging markets"]
    
    print("\nSynthesis 1 Analysis:")
    print(f"Current - Meta words: {count_keywords(current_synthesis, meta_keywords)}, Topic words: {count_keywords(current_synthesis, topic_keywords1)}")
    print(f"Improved - Meta words: {count_keywords(improved_synthesis, meta_keywords)}, Topic words: {count_keywords(improved_synthesis, topic_keywords1)}")
    print(f"Alternative - Meta words: {count_keywords(alternative_synthesis, meta_keywords)}, Topic words: {count_keywords(alternative_synthesis, topic_keywords1)}")
    
    print("\nSynthesis 2 Analysis:")
    print(f"Current - Meta words: {count_keywords(current_synthesis2, meta_keywords)}, Topic words: {count_keywords(current_synthesis2, topic_keywords2)}")
    print(f"Improved - Meta words: {count_keywords(improved_synthesis2, meta_keywords)}, Topic words: {count_keywords(improved_synthesis2, topic_keywords2)}")


if __name__ == "__main__":
    asyncio.run(test_synthesis_approaches())