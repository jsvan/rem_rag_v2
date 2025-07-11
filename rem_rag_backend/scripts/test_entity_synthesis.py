#!/usr/bin/env python3
"""
Test entity extraction and synthesis with a small number of articles.
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
from datasets import load_dataset

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
    
    # Load just 3 articles from 1922
    print("Loading Foreign Affairs dataset...")
    dataset = load_dataset("bitsinthesky/foreign_affairs_2024june20")
    
    # Filter for 1922 and take first 3
    articles_1922 = []
    for article in dataset['train']:
        # Extract year from URL (format: .../articles/1922-10-01/...)
        url = article.get('url', '')
        import re
        year_match = re.search(r'/(\d{4})-\d{2}-\d{2}/', url)
        if year_match and year_match.group(1) == '1922':
            articles_1922.append({
                'text': article['text'],
                'year': 1922,
                'article_id': url.split('/')[-1],  # Use last part of URL as ID
                'title': article.get('title', 'Unknown Title')
            })
            if len(articles_1922) >= 3:
                break
    
    print(f"\nProcessing {len(articles_1922)} articles from 1922...")
    
    # Process with async mode
    results = await reading_cycle.process_articles_async(articles_1922)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total requests: {results.get('total_requests', 0)}")
    print(f"Entities extracted: {results.get('entities_extracted', 0)}")
    print(f"Learnings stored: {results.get('learnings_stored', 0)}")
    print(f"Syntheses stored: {results.get('syntheses_stored', 0)}")
    print(f"Summaries stored: {results.get('summaries_stored', 0)}")
    
    # Check node counts
    print("\nChecking node counts...")
    node_counts = vector_store.get_node_counts()
    for node_type, count in sorted(node_counts.items()):
        print(f"  {node_type:20} {count:5}")

if __name__ == "__main__":
    asyncio.run(main())