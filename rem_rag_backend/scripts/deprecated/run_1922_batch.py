#!/usr/bin/env python3
"""
Batch processing version of the 1922 Foreign Affairs experiment.

This version uses the BatchReadingCycle for much faster processing.
"""

import os
import sys
import json
import glob
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle_batch import BatchReadingCycle
from rem_rag_backend.src.config import OPENAI_API_KEY


class Batch1922Experiment:
    """Batch processing version of the 1922 experiment"""
    
    def __init__(self, batch_size: int = 5, chunk_concurrency: int = 10):
        """
        Initialize with configurable batch processing settings.
        
        Args:
            batch_size: Number of articles to process concurrently
            chunk_concurrency: Max chunks to process concurrently per article
        """
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore(collection_name="rem_rag_1922_batch")
        self.chunker = SentenceAwareChunker()
        
        # Initialize batch reader
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.store,
            self.chunker,
            batch_size=batch_size,
            max_concurrent_chunks=chunk_concurrency
        )
        
    async def run(self):
        """Execute the experiment"""
        print("🚀 Starting Batch 1922 Experiment")
        print("=" * 50)
        print(f"Batch size: {self.batch_reader.batch_size}")
        print(f"Chunk concurrency: {self.batch_reader.max_concurrent_chunks}")
        
        # Load articles
        articles = self.load_1922_articles()
        print(f"\n📚 Loaded {len(articles)} articles from 1922")
        
        # Process articles using batch processing
        print("\n📖 Processing articles in batches...")
        start_time = time.time()
        
        try:
            stats = await self.batch_reader.process_articles_batch(articles)
            
            # Print summary statistics
            total_time = time.time() - start_time
            successful = sum(1 for s in stats if 'error' not in s)
            total_chunks = sum(s.get('total_chunks', 0) for s in stats if 'error' not in s)
            total_entities = sum(s.get('total_entities', 0) for s in stats if 'error' not in s)
            total_syntheses = sum(s.get('valuable_syntheses', 0) for s in stats if 'error' not in s)
            
            print(f"\n✅ Processing complete!")
            print(f"   Articles: {successful}/{len(articles)}")
            print(f"   Total chunks: {total_chunks}")
            print(f"   Total entities: {total_entities}")
            print(f"   Valuable syntheses: {total_syntheses}")
            print(f"   Time: {total_time:.1f}s ({total_time/len(articles):.1f}s per article)")
            
            # Calculate estimated speedup vs serial
            serial_estimate = len(articles) * 30  # ~30s per article in serial
            print(f"   Estimated speedup: {serial_estimate/total_time:.1f}x vs serial processing")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze results
        print("\n\n📊 Analyzing Results")
        print("=" * 50)
        await self.analyze_results()
    
    def load_1922_articles(self) -> List[Dict[str, Any]]:
        """Load all 1922 articles from local JSON files"""
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
        pattern = os.path.join(data_dir, "1922_*.json")
        
        articles = []
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Transform to expected format
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title', 'Unknown'),
                        'author': article_data.get('author', 'Unknown'),
                        'year': article_data.get('year', 1922),
                        'volume': article_data.get('volume', 1),
                        'issue': article_data.get('issue', 1),
                        'url': article_data.get('url', ''),
                        'article_id': f"1922-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:  # Only add if we have content
                        articles.append(article)
            except Exception as e:
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    async def analyze_results(self):
        """Analyze what we've stored"""
        # Query for some key themes from 1922
        themes = [
            "League of Nations",
            "Versailles Treaty",
            "Soviet Russia",
            "reparations",
            "popular diplomacy"
        ]
        
        print("\n🔍 Searching for key 1922 themes:")
        for theme in themes:
            try:
                results = self.store.query(theme, k=3, filter={'year': 1922})
                
                if results['documents']:
                    print(f"\n📌 {theme}:")
                    for i, doc in enumerate(results['documents']):
                        metadata = results['metadatas'][i]
                        print(f"  - From: {metadata.get('article_title', 'Unknown')}")
                        print(f"    Preview: {doc[:100]}...")
            except Exception as e:
                print(f"\n❌ Error querying '{theme}': {e}")
        
        # Sample some random content
        print("\n\n🎲 Random sample of stored content:")
        try:
            sample = self.store.sample(n=5, filter={'year': 1922})
            if sample['documents']:
                for i, doc in enumerate(sample['documents']):
                    metadata = sample['metadatas'][i]
                    print(f"\n  [{i+1}] Type: {metadata.get('node_type', 'unknown')}")
                    print(f"      Article: {metadata.get('article_title', 'Unknown')}")
                    print(f"      Preview: {doc[:150]}...")
        except Exception as e:
            print(f"  ❌ Error sampling: {e}")


async def main():
    """Run the batch experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process 1922 Foreign Affairs articles with batch processing")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of articles to process concurrently")
    parser.add_argument("--chunk-concurrency", type=int, default=10, help="Max chunks to process concurrently per article")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    experiment = Batch1922Experiment(
        batch_size=args.batch_size,
        chunk_concurrency=args.chunk_concurrency
    )
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("This experiment will:")
    print("1. Load 1922 Foreign Affairs articles")
    print("2. Process them using batch reading cycle")
    print("3. Extract entities and generate syntheses")
    print("4. Search for key themes from 1922\n")
    
    try:
        await experiment.run()
        print("\n✅ Experiment completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())