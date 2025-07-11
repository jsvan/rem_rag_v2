#!/usr/bin/env python3
"""
Test processing a single article with the new async writer service.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle import ReadingCycle

async def main():
    # Initialize components
    vector_store = REMVectorStore()
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    chunker = SentenceAwareChunker()
    
    reading_cycle = ReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=vector_store,
        chunker=chunker
    )
    
    # Create a longer test article
    test_article = {
        'text': """The Cuban Missile Crisis of October 1962 represented the closest the world has come to nuclear war during the Cold War period. The crisis began when American U-2 spy planes photographed Soviet medium-range ballistic missiles under construction in Cuba. President John F. Kennedy was informed of the missiles on October 16, 1962, setting off a thirteen-day confrontation between the United States and the Soviet Union.

The Soviet Union's decision to place nuclear missiles in Cuba was driven by multiple factors. Premier Nikita Khrushchev saw it as a way to defend Cuba against future American invasion attempts after the failed Bay of Pigs invasion in 1961. Additionally, the Soviets sought to address the strategic imbalance created by American Jupiter missiles in Turkey and Italy, which could strike the Soviet Union.

President Kennedy convened the Executive Committee of the National Security Council (ExComm) to discuss America's response. The options ranged from diplomatic pressure to a full-scale invasion of Cuba. Military leaders, including Air Force Chief of Staff Curtis LeMay, advocated for immediate air strikes. However, Kennedy was concerned that military action could escalate into nuclear war.

After intense deliberations, Kennedy decided on a naval blockade, which he termed a "quarantine" to avoid the implications of an act of war under international law. On October 22, Kennedy announced the discovery of the missiles to the American public in a televised address, demanding their removal and announcing the naval quarantine of Cuba.

The world held its breath as Soviet ships approached the quarantine line. On October 24, Soviet ships stopped short of the blockade, providing the first sign that escalation might be avoided. However, work on the missile sites continued, and tensions remained extremely high.

The crisis reached its peak on October 27 when a Soviet surface-to-air missile shot down Major Rudolf Anderson Jr.'s U-2 over Cuba. The same day, another U-2 accidentally strayed into Soviet airspace over Siberia. These incidents brought the superpowers dangerously close to war.

Behind the scenes, both leaders were searching for a face-saving solution. Khrushchev sent two letters to Kennedy - the first offering to remove the missiles in exchange for a U.S. pledge not to invade Cuba, and the second demanding the removal of Jupiter missiles from Turkey. Kennedy publicly accepted the first offer while secretly agreeing to remove the missiles from Turkey after a face-saving delay.

On October 28, Khrushchev announced that he would dismantle the missiles and return them to the Soviet Union. The crisis was over, but its impact was profound. Both leaders had stared into the nuclear abyss and pulled back.

The Cuban Missile Crisis led to several important developments in superpower relations. The Moscow-Washington hotline was established to enable direct communication between leaders during crises. Both nations also became more cautious about nuclear brinkmanship, leading to future arms control agreements.

For Cuba's Fidel Castro, the crisis was a bitter disappointment. He felt betrayed by Soviet willingness to negotiate without consulting him and remained convinced that Cuba should have been prepared to sacrifice itself for the socialist cause.

The crisis revealed important lessons about nuclear deterrence, crisis management, and the importance of communication channels between adversaries. It demonstrated that even rational leaders could find themselves on a path to nuclear war through miscalculation and misunderstanding. The world had come within a hair's breadth of nuclear catastrophe, and both superpowers recognized the need for more stable mechanisms to manage their rivalry.""",
        'year': 1962,
        'article_id': 'test_article_1962',
        'title': 'Test Article: Cuban Missile Crisis - A Detailed Analysis'
    }
    
    print("Processing single test article...")
    
    # Get initial stats
    initial_stats = vector_store.get_stats()
    initial_count = initial_stats['total_documents']
    print(f"\nInitial document count: {initial_count}")
    
    # Process with async mode
    results = await reading_cycle.process_articles_async([test_article])
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total requests: {results.get('total_requests', 0)}")
    print(f"Entities extracted: {results.get('entities_extracted', 0)}")
    print(f"Learnings stored: {results.get('learnings_stored', 0)}")
    print(f"Syntheses stored: {results.get('syntheses_stored', 0)}")
    print(f"Summaries stored: {results.get('summaries_stored', 0)}")
    print(f"Database writes: {results.get('database_writes', 0)}")
    
    # Check final stats
    final_stats = vector_store.get_stats()
    final_count = final_stats['total_documents']
    print(f"\nFinal document count: {final_count}")
    print(f"Documents added: {final_count - initial_count}")

if __name__ == "__main__":
    asyncio.run(main())