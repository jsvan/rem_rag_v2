#!/usr/bin/env python3
"""
Simple test of batch processing with just 3 articles.
"""

import os
import sys
import json
import glob
import asyncio
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle_batch import BatchReadingCycle
from rem_rag_backend.src.config import OPENAI_API_KEY


async def test_batch():
    """Test batch processing with 3 articles"""
    print("Starting batch test...")
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    store = REMVectorStore(collection_name="test_1922_batch")
    chunker = SentenceAwareChunker()
    
    batch_reader = BatchReadingCycle(
        llm,
        structured_llm,
        store,
        chunker,
        batch_size=2,
        max_concurrent_chunks=5
    )
    
    # Load just 3 articles
    data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    pattern = os.path.join(data_dir, "1922_*.json")
    
    articles = []
    for filepath in sorted(glob.glob(pattern))[:3]:  # Just first 3
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                
                article = {
                    'text': article_data.get('content', ''),
                    'title': article_data.get('title', 'Unknown'),
                    'year': 1922,
                    'article_id': f"1922-{len(articles)+1:03d}"
                }
                
                if article['text']:
                    articles.append(article)
                    print(f"Loaded: {article['title'][:50]}...")
        except Exception as e:
            print(f"Error loading: {e}")
    
    print(f"\nProcessing {len(articles)} articles...")
    
    start_time = time.time()
    try:
        stats = await batch_reader.process_articles_batch(articles)
        print(f"\nCompleted in {time.time() - start_time:.1f}s")
        
        # Print stats
        for stat in stats:
            if 'error' not in stat:
                print(f"- {stat['title'][:40]}: {stat['total_chunks']} chunks, {stat['total_entities']} entities")
            else:
                print(f"- {stat['title'][:40]}: ERROR - {stat['error']}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
    else:
        asyncio.run(test_batch())