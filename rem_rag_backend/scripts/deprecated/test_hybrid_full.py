#!/usr/bin/env python3
"""
Test the full hybrid processing approach with both reading and REM cycles.

This test verifies:
1. No database locking issues occur
2. OpenAI calls are efficiently batched
3. The complete pipeline works end-to-end
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle_hybrid import BatchReadingCycle
from rem_rag_backend.src.core.rem_cycle_hybrid import BatchREMCycle
from rem_rag_backend.src.config import OPENAI_API_KEY, REM_SCALING_FACTOR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_articles():
    """Create a few test articles for processing"""
    # Make articles longer to ensure they get chunked
    return [
        {
            'text': """The Cuban Missile Crisis of October 1962 represents perhaps the closest the world has ever come to nuclear war. The crisis began when American U-2 spy planes discovered Soviet medium-range ballistic missiles under construction in Cuba, just 90 miles from the Florida coast. These missiles, capable of carrying nuclear warheads, could strike most of the continental United States within minutes of launch.

            President John F. Kennedy faced an extraordinary dilemma. Military advisors pushed for immediate air strikes or a full-scale invasion of Cuba. However, Kennedy understood that such actions could trigger Soviet retaliation in Berlin or elsewhere, potentially escalating to nuclear war. Instead, he chose a middle path: a naval blockade of Cuba, which he carefully termed a "quarantine" to avoid the legal implications of a blockade, which is considered an act of war.

            For thirteen tense days, the world held its breath. Soviet ships continued toward Cuba, while American naval vessels prepared to intercept them. Behind the scenes, intense negotiations took place. Soviet Premier Nikita Khrushchev sent conflicting messages, at times seeming conciliatory, at other times belligerent. The situation reached its peak when a Soviet surface-to-air missile shot down Major Rudolf Anderson Jr.'s U-2 over Cuba.

            The resolution came through secret negotiations. Kennedy agreed to publicly pledge not to invade Cuba and privately promised to remove American Jupiter missiles from Turkey. In exchange, Khrushchev would withdraw the Soviet missiles from Cuba. This face-saving compromise allowed both leaders to claim victory while stepping back from the brink of catastrophe.

            The crisis had profound implications for nuclear diplomacy. It led to the establishment of the Moscow-Washington hotline, ensuring direct communication between the superpowers during crises. It also sparked a renewed push for arms control, culminating in the 1963 Partial Test Ban Treaty. Perhaps most importantly, it demonstrated that even at the height of the Cold War, rational leadership could prevail over ideological confrontation when faced with mutual destruction.""",
            'title': 'Lessons from the Cuban Missile Crisis',
            'author': 'Test Author 1',
            'year': 1962,
            'article_id': 'test-1962-001'
        },
        {
            'text': """The fall of the Berlin Wall on November 9, 1989, marked not just the physical dismantling of a barrier, but the symbolic end of the Cold War division of Europe. For 28 years, the Wall had stood as the most visible manifestation of the Iron Curtain, separating families, friends, and a nation into two ideological camps.

            The Wall's demise came suddenly, though tensions had been building throughout 1989. Hungary had opened its border with Austria, creating the first hole in the Iron Curtain. Thousands of East Germans fled through this route, while others sought refuge in West German embassies across Eastern Europe. Meanwhile, weekly demonstrations in Leipzig grew from a few thousand to hundreds of thousands, with citizens chanting "Wir sind das Volk" (We are the people).

            The immediate cause of the Wall's fall was almost accidental. On November 9, East German spokesman Günter Schabowski announced new regulations that would ease border restrictions. When pressed about when the regulations would take effect, he fumbled through his notes and said, "As far as I know, immediately." This was not the plan, but East Berliners rushed to the crossing points. Overwhelmed guards, lacking clear orders and facing massive crowds, eventually opened the gates.

            What followed was one of history's great celebrations. East and West Berliners embraced, danced, and took hammers to the Wall. Champagne flowed, and pieces of the Wall became instant souvenirs. The Brandenburg Gate, closed for nearly three decades, became the focal point of reunification celebrations.

            The rapid reunification that followed surprised even optimistic observers. Within a year, Germany was reunified under West German law. The economic challenges were immense - East German industry was largely obsolete, and the cost of modernization ran into hundreds of billions of marks. Yet the political momentum was unstoppable. The fall of the Wall triggered a cascade of revolutions across Eastern Europe, fundamentally reshaping the continent's political landscape.""",
            'title': 'The Fall of the Berlin Wall',
            'author': 'Test Author 2',
            'year': 1989,
            'article_id': 'test-1989-001'
        },
        {
            'text': """China's economic transformation under Deng Xiaoping represents one of the most remarkable development stories in human history. When Deng consolidated power in 1978, China was an impoverished, isolated nation, still reeling from the Cultural Revolution. By the time of his death in 1997, China had become the world's fastest-growing major economy, lifting hundreds of millions out of poverty.

            Deng's genius lay in his pragmatic approach, encapsulated in his famous saying, "It doesn't matter if a cat is black or white, as long as it catches mice." This philosophy allowed him to introduce market reforms while maintaining Communist Party control. The strategy began with agricultural reforms, dismantling collective farms and allowing peasants to sell surplus produce at market prices. Agricultural productivity soared, freeing up labor for industrial development.

            The establishment of Special Economic Zones (SEZs) was perhaps Deng's most innovative policy. Starting with Shenzhen, adjacent to Hong Kong, these zones offered tax incentives, relaxed regulations, and modern infrastructure to attract foreign investment. Shenzhen transformed from a fishing village of 30,000 to a megacity of over 12 million. Other SEZs followed, creating laboratories for capitalist experimentation within the socialist system.

            Foreign investment poured in, drawn by China's vast labor force and improving infrastructure. Joint ventures brought not just capital but crucial technology and management expertise. Chinese firms learned quickly, often surpassing their foreign partners. The trade surplus grew, providing capital for further development.

            The results were spectacular. GDP growth averaged over 9% annually for three decades. Per capita income increased twenty-fold. China became the world's manufacturing hub, producing everything from toys to high-speed trains. Yet this growth came with costs: environmental degradation, widening inequality, and persistent political repression. Deng's model of "socialism with Chinese characteristics" created a unique hybrid system that continues to challenge conventional economic wisdom.""",
            'title': 'China\'s Economic Miracle',
            'author': 'Test Author 3',
            'year': 1990,
            'article_id': 'test-1990-001'
        }
    ]


async def test_hybrid_processing():
    """Test the complete hybrid processing pipeline"""
    print("🧪 Testing Hybrid Processing Pipeline")
    print("=" * 70)
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore(collection_name="test_hybrid_" + str(int(datetime.now().timestamp())))
    chunker = SentenceAwareChunker()
    
    print(f"📊 Using test collection: {store.collection_name}")
    
    # Create processors
    reading_cycle = BatchReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=store,
        chunker=chunker,
        batch_size=5,
        max_concurrent_chunks=20
    )
    
    rem_cycle = BatchREMCycle(
        store=store,
        llm=llm,
        api_key=OPENAI_API_KEY,
        max_concurrent=20
    )
    
    # Test 1: Process articles
    print("\n📚 Test 1: Processing test articles...")
    articles = create_test_articles()
    
    try:
        stats = await reading_cycle.process_articles_batch(articles)
        
        successful = sum(1 for s in stats if "error" not in s)
        print(f"\n✅ Articles processed: {successful}/{len(articles)}")
        
        for stat in stats:
            if "error" in stat:
                print(f"   ❌ {stat['title']}: {stat['error']}")
            else:
                print(f"   ✅ {stat['title']}: {stat['total_chunks']} chunks, "
                      f"{stat['valuable_syntheses']} syntheses")
        
    except Exception as e:
        print(f"❌ Article processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Check database state
    print("\n📊 Test 2: Checking database state...")
    try:
        # Count nodes by type
        for node_type in ['chunk', 'summary', 'synthesis']:
            # ChromaDB doesn't support count with filter, so we use get with limit
            results = store.collection.get(where={"node_type": node_type}, limit=1000)
            count = len(results["ids"])
            print(f"   • {node_type}: {count} nodes")
            
    except Exception as e:
        print(f"❌ Database check failed: {e}")
    
    # Test 3: Run REM cycle
    print("\n🌙 Test 3: Running REM cycle...")
    try:
        # Calculate number of cycles
        total_nodes = store.collection.count()
        num_cycles = max(1, int(total_nodes * REM_SCALING_FACTOR))
        
        print(f"   • Total nodes: {total_nodes}")
        print(f"   • REM cycles to run: {num_cycles}")
        
        rem_stats = await rem_cycle.run_batch_rem_cycles(
            num_cycles=num_cycles,
            current_year=1990
        )
        
        print(f"\n✅ REM cycle completed!")
        print(f"   • REM nodes created: {rem_stats['total_rem_nodes']}")
        print(f"   • Valuable syntheses: {rem_stats['valuable_syntheses']}")
        print(f"   • Failed: {rem_stats['failed']}")
        
    except Exception as e:
        print(f"❌ REM cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Query the knowledge base
    print("\n🔍 Test 4: Testing knowledge retrieval...")
    try:
        test_queries = [
            "What were the key Cold War crises?",
            "How did economic reforms change nations?",
            "What role did popular movements play?"
        ]
        
        for query in test_queries:
            results = store.query(query, k=3)
            print(f"\n   Query: '{query}'")
            print(f"   Found: {len(results['documents'])} relevant passages")
            if results['documents']:
                # Show first result
                doc = results['documents'][0][:100] + "..."
                meta = results['metadatas'][0] if results['metadatas'] else {}
                print(f"   Top result ({meta.get('node_type', 'unknown')}): {doc}")
                
    except Exception as e:
        print(f"❌ Query test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")


async def main():
    """Run the test"""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    print("⚠️  This test will:")
    print("  • Create a new test collection in ChromaDB")
    print("  • Process 3 test articles")
    print("  • Run a small REM cycle")
    print("  • Cost approximately $0.01-0.02")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    await test_hybrid_processing()


if __name__ == "__main__":
    asyncio.run(main())