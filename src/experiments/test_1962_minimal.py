"""
Minimal test of 1962 experiment - process just 1 article to test the pipeline
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.data_processing.fa_loader import ForeignAffairsLoader
from src.data_processing.chunker import SmartChunker
from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
import asyncio


def test_minimal():
    """Test with just one article"""
    print("🧪 Minimal 1962 Test")
    print("=" * 50)
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore(collection_name="test_1962_minimal")
    loader = ForeignAffairsLoader()
    chunker = SmartChunker()
    reading_cycle = ReadingCycle(llm, structured_llm, store, chunker)
    rem_cycle = REMCycle(llm, store)
    
    # Load dataset
    print("\n📚 Loading dataset...")
    all_data = loader.load_dataset()
    
    # Convert to list
    if hasattr(all_data, 'to_dict'):
        all_data = all_data.to_dict('records')
    
    # Find 1962 articles
    articles_1962 = []
    for item in all_data:
        if isinstance(item, dict):
            year = item.get("year")
            
            # Try to extract from URL if year is None
            if year is None and 'url' in item:
                import re
                match = re.search(r'/(\d{4})-\d{2}-\d{2}/', item['url'])
                if match:
                    year = int(match.group(1))
            
            if year == 1962:
                item['year'] = year
                articles_1962.append(item)
    
    print(f"Found {len(articles_1962)} articles from 1962")
    
    if not articles_1962:
        print("❌ No 1962 articles found!")
        return
    
    # Process just the first article
    article = articles_1962[0]
    article['article_id'] = "test-1962-001"
    
    print(f"\n📄 Processing article: {article.get('title', 'Unknown')}")
    print(f"   URL: {article.get('url', 'No URL')}")
    print(f"   Text preview: {article.get('text', '')[:200]}...")
    
    try:
        # Process through READING cycle
        print("\n📖 Running READING cycle...")
        result = asyncio.run(reading_cycle.process_article(article))
        print(f"✅ READING cycle complete: {result}")
    except Exception as e:
        print(f"❌ READING cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Try a mini REM cycle
    print("\n🌙 Running mini REM cycle (3 dreams)...")
    try:
        rem_nodes = rem_cycle.run_cycle(num_dreams=3, current_year=1962)
        print(f"✅ Created {len(rem_nodes)} REM nodes")
    except Exception as e:
        print(f"❌ REM cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Test complete!")


if __name__ == "__main__":
    test_minimal()