"""
Test REM RAG with mock Foreign Affairs articles to verify the system works
"""

import os
import sys
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.data_processing.chunker import SmartChunker
from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore


def create_mock_articles():
    """Create mock 1962 Foreign Affairs articles for testing"""
    return [
        {
            "article_id": "1962-10-001",
            "title": "The Lessons of Cuba",
            "text": """The Cuban Missile Crisis of October 1962 brought the world closer to nuclear war than 
            ever before. Soviet Premier Khrushchev's decision to place intermediate-range missiles in Cuba 
            represented a bold gambit to alter the strategic balance. President Kennedy's response - a naval 
            quarantine rather than immediate military action - demonstrated both firmness and restraint. 
            The crisis revealed critical lessons about nuclear brinkmanship, the importance of communication 
            channels between superpowers, and the role of miscalculation in international relations.""",
            "url": "www.foreignaffairs.com/articles/cuba/1962-10-15/lessons-of-cuba",
            "year": 1962,
            "date": "1962-10-15"
        },
        {
            "article_id": "1962-07-001", 
            "title": "Berlin: Test of Western Will",
            "text": """The Berlin Wall, erected in August 1961, continues to test Western resolve. 
            The division of Berlin represents more than a physical barrier - it symbolizes the ideological 
            divide between East and West. NATO's commitment to West Berlin's defense remains firm, but 
            questions persist about escalation scenarios. The presence of Western troops in Berlin serves 
            as both a tripwire and a guarantee. Any Soviet move against West Berlin would trigger an 
            automatic Western response, potentially escalating to nuclear confrontation.""",
            "url": "www.foreignaffairs.com/articles/germany/1962-07-01/berlin-test-western-will",
            "year": 1962,
            "date": "1962-07-01"
        },
        {
            "article_id": "1962-04-001",
            "title": "The Sino-Soviet Split: Implications for the West",
            "text": """The growing rift between Moscow and Beijing presents both opportunities and dangers 
            for Western policy. Ideological disputes over the nature of revolution and coexistence with 
            capitalism have evolved into a fundamental split in the Communist world. China's criticism of 
            Soviet 'revisionism' and Moscow's concern over Chinese 'adventurism' create new dynamics in 
            the Cold War. The West must carefully consider how to respond to this division without 
            precipitating a reconciliation based on shared opposition to Western policies.""",
            "url": "www.foreignaffairs.com/articles/china/1962-04-01/sino-soviet-split",
            "year": 1962,
            "date": "1962-04-01"
        }
    ]


async def test_system():
    """Test the REM RAG system with mock data"""
    print("🧪 Testing REM RAG with Mock 1962 Articles")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore(collection_name="test_rem_rag_mock")
    chunker = SmartChunker()
    reading_cycle = ReadingCycle(llm, structured_llm, store, chunker)
    rem_cycle = REMCycle(llm, store)
    
    # Create mock articles
    articles = create_mock_articles()
    print(f"\n📚 Created {len(articles)} mock articles")
    
    # Process through READING cycle
    print("\n📖 READING Cycle:")
    print("-" * 40)
    for article in articles:
        print(f"\nProcessing: {article['title']}")
        try:
            stats = await reading_cycle.process_article(article)
            print(f"  ✅ Processed {stats['total_chunks']} chunks")
            print(f"  ✅ Found {stats['total_entities']} entities")
            print(f"  ✅ Created {stats['valuable_syntheses']} valuable syntheses")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Run a small REM cycle
    print("\n\n🌙 REM Cycle:")
    print("-" * 40)
    try:
        rem_nodes = rem_cycle.run_cycle(num_dreams=5, current_year=1962)
        print(f"\n✅ Created {len(rem_nodes)} REM dream nodes")
        
        # Query for insights
        print("\n💡 Sample REM Insights on 'nuclear crisis':")
        insights = rem_cycle.query_rem_insights("nuclear crisis", top_k=2)
        for i, insight in enumerate(insights, 1):
            print(f"\n  Insight {i}:")
            print(f"  Question: {insight['question']}")
            print(f"  Years: {insight['source_years']}")
            print(f"  Synthesis: {insight['synthesis'][:200]}...")
            
    except Exception as e:
        print(f"❌ REM cycle error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n✨ Test complete!")
    print("\n📝 Next steps:")
    print("  1. Rescrape Foreign Affairs data with proper metadata")
    print("  2. Include year, author, issue number in dataset")
    print("  3. Add better error handling for API timeouts")
    print("  4. Consider batching API calls for efficiency")


def main():
    """Run the test"""
    asyncio.run(test_system())


if __name__ == "__main__":
    main()