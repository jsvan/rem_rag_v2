#!/usr/bin/env python3
"""
Process a full year of Foreign Affairs articles using batch reading cycle.

This script:
1. Loads all articles from a specified year
2. Processes them in batches for better performance
3. Runs REM cycles at the end
4. Provides detailed statistics
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import argparse
from datetime import datetime

# Add the project root to Python path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle_batch import BatchReadingCycle
from rem_rag_backend.src.core.rem_cycle_batch import BatchREMCycle
from rem_rag_backend.src.config import OPENAI_API_KEY, DATA_DIR, REM_SCALING_FACTOR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YearBatchProcessor:
    """Process a full year of articles using batch processing."""
    
    def __init__(self, batch_size: int = 10, chunk_concurrency: int = 20):
        """
        Initialize the processor.
        
        Args:
            batch_size: Number of articles to process concurrently
            chunk_concurrency: Max chunks to process concurrently
        """
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.vector_store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.vector_store,
            self.chunker,
            batch_size=batch_size,
            max_concurrent_chunks=chunk_concurrency
        )
        
        self.rem_cycle = BatchREMCycle(
            self.vector_store,
            self.llm,
            api_key=OPENAI_API_KEY
        )
    
    def load_year_articles(self, year: int) -> List[Dict]:
        """Load all articles from a specific year."""
        # Update this path based on your data location
        data_file = Path(f"/Users/jsv/Projects/foreign_affairs/data/json/foreign_affairs_{year}.json")
        
        if not data_file.exists():
            # Try alternative path
            data_file = Path(f"/Users/jsv/Projects/foreign_affairs/data/foreign_affairs_{year}s.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"No data file found for year {year}")
        
        with open(data_file, 'r') as f:
            year_data = json.load(f)
        
        # Extract articles
        articles = []
        for entry in year_data:
            # Handle different JSON formats
            if isinstance(entry, dict) and 'text' in entry:
                articles.append(entry)
            elif isinstance(entry, dict) and 'articles' in entry:
                articles.extend(entry['articles'])
        
        logger.info(f"Loaded {len(articles)} articles from {year}")
        return articles
    
    async def process_year(self, year: int) -> Dict:
        """
        Process all articles from a year.
        
        Args:
            year: The year to process
            
        Returns:
            Processing statistics
        """
        print(f"\n🚀 Starting Batch Processing for Year {year}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Load articles
        try:
            articles = self.load_year_articles(year)
        except FileNotFoundError as e:
            logger.error(str(e))
            return {"error": str(e)}
        
        print(f"📚 Loaded {len(articles)} articles from {year}")
        print(f"🔧 Batch size: {self.batch_reader.batch_size}")
        print(f"🔧 Chunk concurrency: {self.batch_reader.max_concurrent_chunks}")
        
        # Process articles in batches
        print(f"\n📖 Processing {year} articles...")
        article_stats = await self.batch_reader.process_articles_batch(articles)
        
        # Calculate statistics
        total_chunks = sum(s['total_chunks'] for s in article_stats if 'error' not in s)
        total_entities = sum(s['total_entities'] for s in article_stats if 'error' not in s)
        total_syntheses = sum(s['valuable_syntheses'] for s in article_stats if 'error' not in s)
        errors = sum(1 for s in article_stats if 'error' in s)
        
        reading_time = time.time() - start_time
        
        print(f"\n✅ Reading phase complete!")
        print(f"   Processed: {len(articles) - errors}/{len(articles)} articles")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total entities: {total_entities}")
        print(f"   Valuable syntheses: {total_syntheses}")
        print(f"   Time: {reading_time:.1f}s ({reading_time/len(articles):.1f}s per article)")
        
        # Run REM cycles
        print(f"\n🌙 Running REM cycles...")
        rem_start = time.time()
        
        # Calculate number of REM cycles
        all_nodes = self.vector_store.get_all_nodes(limit=1)
        total_non_rem = len(all_nodes["ids"])
        num_rem_cycles = max(1, int(total_non_rem * REM_SCALING_FACTOR))
        
        print(f"   Total non-REM nodes: {total_non_rem}")
        print(f"   REM cycles to run: {num_rem_cycles}")
        
        # Run batch REM
        rem_stats = await self.rem_cycle.run_batch_rem_cycles(
            num_cycles=num_rem_cycles,
            current_year=year
        )
        
        rem_time = time.time() - rem_start
        
        print(f"\n✅ REM phase complete!")
        print(f"   REM nodes created: {rem_stats['total_rem_nodes']}")
        print(f"   Valuable syntheses: {rem_stats['valuable_syntheses']}")
        print(f"   Time: {rem_time:.1f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        
        return {
            "year": year,
            "total_articles": len(articles),
            "successful_articles": len(articles) - errors,
            "errors": errors,
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "reading_syntheses": total_syntheses,
            "rem_nodes": rem_stats['total_rem_nodes'],
            "rem_syntheses": rem_stats['valuable_syntheses'],
            "reading_time": reading_time,
            "rem_time": rem_time,
            "total_time": total_time
        }
    
    def print_summary(self, stats: Dict):
        """Print a nice summary of processing statistics."""
        print("\n" + "=" * 70)
        print(f"📊 YEAR {stats['year']} PROCESSING COMPLETE")
        print("=" * 70)
        
        print(f"\n📚 Articles:")
        print(f"   Total: {stats['total_articles']}")
        print(f"   Successful: {stats['successful_articles']}")
        print(f"   Errors: {stats['errors']}")
        
        print(f"\n🔍 Nodes Created:")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Entities: {stats['total_entities']}")
        print(f"   Reading syntheses: {stats['reading_syntheses']}")
        print(f"   REM nodes: {stats['rem_nodes']}")
        print(f"   REM syntheses: {stats['rem_syntheses']}")
        
        print(f"\n⏱️  Performance:")
        print(f"   Reading phase: {stats['reading_time']:.1f}s")
        print(f"   REM phase: {stats['rem_time']:.1f}s")
        print(f"   Total time: {stats['total_time']:.1f}s")
        print(f"   Average per article: {stats['total_time']/stats['total_articles']:.1f}s")
        
        # Calculate speedup estimate
        serial_estimate = stats['total_articles'] * 120  # ~2 min per article serial
        speedup = serial_estimate / stats['total_time']
        print(f"   Estimated speedup: {speedup:.1f}x vs serial processing")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process a year of Foreign Affairs articles in batches")
    parser.add_argument("year", type=int, help="Year to process (e.g., 2000)")
    parser.add_argument("--batch-size", type=int, default=10, help="Articles to process concurrently")
    parser.add_argument("--chunk-concurrency", type=int, default=20, help="Chunks to process concurrently")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    processor = YearBatchProcessor(
        batch_size=args.batch_size,
        chunk_concurrency=args.chunk_concurrency
    )
    
    stats = await processor.process_year(args.year)
    
    if "error" not in stats:
        processor.print_summary(stats)


if __name__ == "__main__":
    asyncio.run(main())