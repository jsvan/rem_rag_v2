#!/usr/bin/env python3
"""
Test TRUE batch processing with year 1922 data.
This will test synthesis generation since we have existing 1922 data.
"""

import os
import sys
import json
import glob
import asyncio
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle import ReadingCycle
from rem_rag_backend.src.config import OPENAI_API_KEY


async def test_batch_1922():
    """Test the true batch processing with 1922 articles."""
    
    print("🧪 Testing TRUE Batch Processing with Year 1922")
    print("=" * 70)
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore()
    chunker = SentenceAwareChunker()
    
    # Check existing database content
    print("\n📊 Current database state:")
    total = store.collection.count()
    print(f"  • Total nodes: {total}")
    
    # Create batch processor
    batch_processor = ReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=store,
        chunker=chunker,
        api_key=OPENAI_API_KEY
    )
    
    # Load 1922 articles (let's use 3 for testing)
    data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    pattern = os.path.join(data_dir, "1922_*.json")
    
    articles = []
    for filepath in sorted(glob.glob(pattern))[:3]:  # Just 3 articles
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                
                article = {
                    'text': article_data.get('content', ''),
                    'title': article_data.get('title') or 'Unknown',
                    'author': article_data.get('author') or 'Unknown',
                    'year': article_data.get('year') or 1922,
                    'article_id': f"1922-test-{len(articles)+1:03d}"
                }
                
                if article['text']:
                    articles.append(article)
                    print(f"\n📄 Loaded: {article['title'][:60]}...")
                    print(f"   Length: {len(article['text'])} chars")
                    
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
    
    if not articles:
        print("❌ No articles loaded!")
        return
    
    print(f"\n✅ Loaded {len(articles)} test articles from 1922")
    
    # Test batch processing
    print("\n🚀 Starting batch processing...")
    print("💡 Since we have existing data, this should generate synthesis requests")
    print("💰 Using 50% reduced cost via Batch API")
    
    try:
        results = await batch_processor.process_articles_batch(articles)
        
        print("\n📊 Test Results:")
        print(f"  Batch ID: {results.get('batch_id', 'N/A')}")
        print(f"  Total articles: {results.get('total_articles', 0)}")
        print(f"  Total requests: {results.get('total_requests', 0)}")
        print(f"  Syntheses stored: {results.get('syntheses_stored', 0)}")
        print(f"  Summaries stored: {results.get('summaries_stored', 0)}")
        print(f"  Failed requests: {results.get('failed_requests', 0)}")
        print(f"  Processing time: {results.get('processing_time', 0)/60:.1f} minutes")
        
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
        else:
            print("\n✅ Test completed successfully!")
            print("\n🔍 This batch is now processing on OpenAI's servers")
            print(f"   Check status with: python scripts/check_batch_status.py {results.get('batch_id', 'N/A')}")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    print("🚦 Running 1922 test (no user input required)")
    asyncio.run(test_batch_1922())