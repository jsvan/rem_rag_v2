#!/usr/bin/env python3
"""
Test script for TRUE batch processing using OpenAI Batch API.

This script tests the TrueBatchReadingCycle with a small set of articles
to verify the implementation works correctly.
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
from rem_rag_backend.src.core.reading_cycle_true_batch import TrueBatchReadingCycle
from rem_rag_backend.src.config import OPENAI_API_KEY


async def test_true_batch():
    """Test the true batch processing with a few articles."""
    
    print("🧪 Testing TRUE Batch Processing with OpenAI Batch API")
    print("=" * 70)
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore()
    chunker = SentenceAwareChunker()
    
    # Create true batch processor
    batch_processor = TrueBatchReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=store,
        chunker=chunker,
        api_key=OPENAI_API_KEY
    )
    
    # Load a few test articles (let's use 3 from year 2000)
    data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    pattern = os.path.join(data_dir, "2000_*.json")
    
    articles = []
    for filepath in sorted(glob.glob(pattern))[:3]:  # Just 3 articles for testing
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                
                article = {
                    'text': article_data.get('content', ''),
                    'title': article_data.get('title') or 'Unknown',
                    'author': article_data.get('author') or 'Unknown',
                    'year': article_data.get('year') or 2000,
                    'article_id': f"2000-test-{len(articles)+1:03d}"
                }
                
                if article['text']:
                    articles.append(article)
                    print(f"📄 Loaded: {article['title'][:60]}...")
                    
        except Exception as e:
            print(f"⚠️  Error loading article: {e}")
    
    if not articles:
        print("❌ No articles loaded!")
        return
    
    print(f"\n✅ Loaded {len(articles)} test articles")
    
    # Test batch processing
    print("\n🚀 Starting batch processing...")
    print("⚠️  Note: This will submit a real batch to OpenAI and may take several minutes")
    print("💰 Cost: Approximately 50% less than regular API calls")
    
    response = input("\nProceed with test? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
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
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    asyncio.run(test_true_batch())