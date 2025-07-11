#!/usr/bin/env python3
"""
Test the updated prompts for article summaries and REM syntheses.
"""

import os
import sys
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.llm import LLMClient
from rem_rag_backend.src.config import ARTICLE_SUMMARY_PROMPT, OPENAI_API_KEY


async def test_article_summary():
    """Test the updated article summary prompt"""
    print("🧪 Testing Updated Article Summary Prompt")
    print("=" * 60)
    
    # Sample article text
    article_text = """The Cuban Missile Crisis of October 1962 represents perhaps the closest the world has ever come to nuclear war. The crisis began when American U-2 spy planes discovered Soviet medium-range ballistic missiles under construction in Cuba, just 90 miles from the Florida coast. These missiles, capable of carrying nuclear warheads, could strike most of the continental United States within minutes of launch.

President John F. Kennedy faced an extraordinary dilemma. Military advisors pushed for immediate air strikes or a full-scale invasion of Cuba. However, Kennedy understood that such actions could trigger Soviet retaliation in Berlin or elsewhere, potentially escalating to nuclear war. Instead, he chose a middle path: a naval blockade of Cuba.

The resolution came through secret negotiations. Kennedy agreed to publicly pledge not to invade Cuba and privately promised to remove American Jupiter missiles from Turkey. In exchange, Khrushchev would withdraw the Soviet missiles from Cuba."""
    
    llm = LLMClient()
    
    prompt = ARTICLE_SUMMARY_PROMPT.format(article_text=article_text)
    print("\n📝 Generating article summary...")
    
    summary = await llm.generate(prompt, max_tokens=300)
    
    print("\n📄 Article Summary:")
    print("-" * 60)
    print(summary)
    print("-" * 60)
    
    # Count words
    word_count = len(summary.split())
    print(f"\n📊 Word count: {word_count} words")
    
    # Count paragraphs
    paragraphs = [p.strip() for p in summary.split('\n') if p.strip()]
    print(f"📊 Paragraphs: {len(paragraphs)}")


async def test_rem_synthesis():
    """Test the updated REM synthesis prompt"""
    print("\n\n🧪 Testing Updated REM Synthesis Prompt")
    print("=" * 60)
    
    # Sample REM question and context
    question = "In what ways do these passages reflect the persistent tension between powerful central authorities and the challenges posed by internal dissent and external pressures?"
    
    context = f"""Implicit Question: {question}

Source Materials:

Source 1 (1962):
The Cuban Missile Crisis demonstrated how superpowers could be pushed to the brink by miscalculation and misunderstanding. Kennedy's advisors were divided between hawks demanding immediate military action and doves seeking diplomatic solutions. (Year: 1962)

Source 2 (1989):
The fall of the Berlin Wall showed how even the most seemingly permanent political structures could crumble rapidly when popular movements gained momentum. East German authorities lost control as citizens took matters into their own hands. (Year: 1989)

Source 3 (1990):
China's economic reforms under Deng Xiaoping represented a careful balance between maintaining Communist Party control and unleashing market forces. The state retained ultimate authority while allowing controlled experimentation in Special Economic Zones. (Year: 1990)"""
    
    llm = LLMClient()
    
    # Format the prompt as it would be in the REM cycle
    prompt = f"""Given the following implicit question and source materials, 
generate a synthesis that reveals hidden patterns and connections:

{context}

Write a concise synthesis in exactly 2 short paragraphs (150-200 words total) that directly addresses the core issue WITHOUT restating or referencing "the question" or "the implicit question".

Start your response by integrating the key concept from the question naturally into your opening sentence. For example, if the question is about "stability of states in crisis", begin with something like "Stability of states in the face of crisis reveals..."

Be direct and specific. Reveal non-obvious patterns across the time periods and offer insights that emerge from the juxtaposition.

Synthesis:"""
    
    print("\n🌙 Generating REM synthesis...")
    
    synthesis = await llm.generate(prompt, max_tokens=300)
    
    print("\n✨ REM Synthesis:")
    print("-" * 60)
    print(synthesis)
    print("-" * 60)
    
    # Check if it starts with the boilerplate
    if synthesis.lower().startswith("the implicit question") or "implicit question" in synthesis[:100].lower():
        print("\n⚠️  WARNING: Synthesis still contains boilerplate reference to 'implicit question'")
    else:
        print("\n✅ Good: Synthesis integrates the concept naturally without boilerplate")
    
    # Count words
    word_count = len(synthesis.split())
    print(f"\n📊 Word count: {word_count} words")


async def main():
    """Run all tests"""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    await test_article_summary()
    await test_rem_synthesis()
    
    print("\n\n✅ All prompt tests completed!")


if __name__ == "__main__":
    asyncio.run(main())