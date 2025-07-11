#!/usr/bin/env python3
"""
Batch processing version of the 2000s decade processor.

This script:
1. Processes articles from 2000-2009 using BatchReadingCycle
2. Handles "Error finding id" issues with retry logic
3. Runs REM cycles at the end of each year
4. Provides detailed error tracking and recovery
"""

import os
import sys
import json
import glob
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
import logging
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle_hybrid import BatchReadingCycle
from rem_rag_backend.src.core.rem_cycle_hybrid import BatchREMCycle
from rem_rag_backend.src.config import REM_SCALING_FACTOR, OPENAI_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.vector_store").setLevel(logging.WARNING)


class Decade2000sBatchProcessor:
    """Batch processing version of the 2000s decade processor"""
    
    def __init__(self, batch_size: int = 10, chunk_concurrency: int = 20, checkpoint_file: str = None):
        """
        Initialize the processor with batch settings.
        
        Args:
            batch_size: Number of articles to process concurrently
            chunk_concurrency: Max chunks to process concurrently per article
            checkpoint_file: Path to checkpoint file for resuming
        """
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        
        # Initialize batch processors
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.store,
            self.chunker,
            batch_size=batch_size,
            max_concurrent_chunks=chunk_concurrency
        )
        
        self.rem_cycle = BatchREMCycle(self.store, self.llm, api_key=OPENAI_API_KEY)
        
        self.stats = defaultdict(lambda: defaultdict(int))
        self.checkpoint_file = checkpoint_file or "2000s_checkpoint.json"
        self.completed_years = self.load_checkpoint()
        
    def load_checkpoint(self) -> set:
        """Load checkpoint to resume processing"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed_years', []))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return set()
    
    def save_checkpoint(self, year: int):
        """Save checkpoint after completing a year"""
        self.completed_years.add(year)
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'completed_years': list(self.completed_years),
                    'stats': dict(self.stats),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def run(self):
        """Execute the decade processing"""
        print("🚀 Starting 2000s Decade Batch Processing")
        print("=" * 70)
        print(f"Processing articles from 2000-2009 with yearly REM cycles")
        print(f"Batch size: {self.batch_reader.batch_size} articles")
        print(f"Chunk concurrency: {self.batch_reader.batch_processor.max_concurrent}")
        print(f"REM scaling factor: {REM_SCALING_FACTOR} (n/{int(1/REM_SCALING_FACTOR)} dreams)")
        
        if self.completed_years:
            print(f"Resuming from checkpoint. Already completed: {sorted(self.completed_years)}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Process each year
        years_to_process = [y for y in range(2000, 2010) if y not in self.completed_years]
        
        with tqdm(total=len(years_to_process), desc="Processing years", unit="year") as year_pbar:
            for year in range(2000, 2010):
                if year in self.completed_years:
                    continue
                    
                print(f"\n\n{'='*60}")
                print(f"📅 YEAR {year}")
                print(f"{'='*60}")
                
                try:
                    await self.process_year(year)
                    self.save_checkpoint(year)
                    year_pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to process year {year}: {e}")
                    print(f"❌ Year {year} failed: {e}")
                    year_pbar.update(1)
                    # Continue with next year instead of stopping
                    continue
        
        # Final analysis
        elapsed = time.time() - start_time
        print(f"\n\n{'='*70}")
        print(f"📊 DECADE PROCESSING COMPLETE")
        print(f"{'='*70}")
        self.print_final_stats(elapsed)
    
    async def process_year(self, year: int):
        """Process all articles from a specific year"""
        # Load articles
        articles = self.load_year_articles(year)
        print(f"\n📚 Loaded {len(articles)} articles from {year}")
        
        if not articles:
            print(f"⚠️  No articles found for {year}")
            return
        
        # Process articles in batches
        print(f"\n📖 Processing {year} articles in batches...")
        year_start = time.time()
        
        try:
            # Process all articles using batch reader
            article_stats = await self.batch_reader.process_articles_batch(articles)
            
            # Aggregate statistics
            successful = 0
            failed = 0
            total_chunks = 0
            total_entities = 0
            total_syntheses = 0
            
            for stat in article_stats:
                if 'error' in stat:
                    failed += 1
                    logger.error(f"Article {stat['title']}: {stat['error']}")
                else:
                    successful += 1
                    total_chunks += stat.get('total_chunks', 0)
                    total_entities += stat.get('total_entities', 0)
                    total_syntheses += stat.get('valuable_syntheses', 0)
            
            # Update year stats
            self.stats[year]['articles_processed'] = successful
            self.stats[year]['articles_failed'] = failed
            self.stats[year]['chunks_stored'] = total_chunks
            self.stats[year]['entities_extracted'] = total_entities
            self.stats[year]['syntheses_created'] = total_syntheses
            
            year_time = time.time() - year_start
            print(f"\n✅ Year {year} articles processed!")
            print(f"   Successful: {successful}/{len(articles)}")
            print(f"   Failed: {failed}")
            print(f"   Chunks: {total_chunks}")
            print(f"   Entities: {total_entities}")
            print(f"   Syntheses: {total_syntheses}")
            print(f"   Time: {year_time:.1f}s ({year_time/len(articles):.1f}s per article)")
            
        except Exception as e:
            logger.error(f"Batch processing failed for {year}: {e}")
            raise
        
        # Run REM cycle at end of year
        print(f"\n🌙 Running REM cycle for {year}...")
        try:
            rem_start = time.time()
            
            # Calculate number of REM cycles based on total nodes
            sample_count = self.store.collection.count()
            num_rem_cycles = max(1, int(sample_count * REM_SCALING_FACTOR))
            
            print(f"   Total nodes: {sample_count}")
            print(f"   REM cycles to run: {num_rem_cycles}")
            
            # Run batch REM
            rem_stats = await self.rem_cycle.run_batch_rem_cycles(
                num_cycles=num_rem_cycles,
                current_year=year
            )
            
            self.stats[year]['rem_nodes'] = rem_stats['total_rem_nodes']
            self.stats[year]['rem_syntheses'] = rem_stats['valuable_syntheses']
            
            rem_time = time.time() - rem_start
            print(f"✨ Created {rem_stats['total_rem_nodes']} REM insights in {rem_time:.1f}s")
            
        except Exception as e:
            logger.error(f"REM cycle failed for {year}: {e}")
            print(f"❌ REM cycle failed: {e}")
            self.stats[year]['rem_failed'] = True
    
    def load_year_articles(self, year: int) -> List[Dict[str, Any]]:
        """Load all articles from a specific year"""
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
        pattern = os.path.join(data_dir, f"{year}_*.json")
        
        articles = []
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Transform to expected format
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title') or 'Unknown',
                        'author': article_data.get('author') or 'Unknown',
                        'year': article_data.get('year') or year,
                        'volume': article_data.get('volume') or 0,
                        'issue': article_data.get('issue') or 0,
                        'url': article_data.get('url') or '',
                        'article_id': f"{year}-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:  # Only add if we have content
                        articles.append(article)
            except Exception as e:
                logger.warning(f"Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def print_final_stats(self, elapsed_time: float):
        """Print comprehensive statistics"""
        total_articles = sum(self.stats[y]['articles_processed'] for y in self.stats)
        total_chunks = sum(self.stats[y]['chunks_stored'] for y in self.stats)
        total_entities = sum(self.stats[y]['entities_extracted'] for y in self.stats)
        total_syntheses = sum(self.stats[y]['syntheses_created'] for y in self.stats)
        total_rem = sum(self.stats[y]['rem_nodes'] for y in self.stats)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total articles processed: {total_articles}")
        print(f"  • Total chunks stored: {total_chunks}")
        print(f"  • Total entities extracted: {total_entities}")
        print(f"  • Total syntheses created: {total_syntheses}")
        print(f"  • Total REM insights: {total_rem}")
        print(f"  • Processing time: {elapsed_time/60:.1f} minutes")
        
        print(f"\n📅 Year-by-Year Breakdown:")
        for year in sorted(self.stats.keys()):
            stats = self.stats[year]
            print(f"\n  {year}:")
            print(f"    • Articles: {stats['articles_processed']} processed, {stats.get('articles_failed', 0)} failed")
            print(f"    • Chunks: {stats['chunks_stored']} stored")
            print(f"    • Entities: {stats['entities_extracted']} extracted")
            print(f"    • Syntheses: {stats['syntheses_created']} created")
            print(f"    • REM: {stats['rem_nodes']} insights, {stats.get('rem_syntheses', 0)} syntheses")
        
        # Query database for final counts by node type
        print(f"\n🗄️  Database Node Counts:")
        try:
            for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
                count = self.store.collection.count(where={"node_type": node_type})
                print(f"    • {node_type}: {count} nodes")
        except Exception as e:
            print(f"    ❌ Error querying database: {e}")


async def main():
    """Run the 2000s decade batch processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process 2000s Foreign Affairs articles with batch processing")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of articles to process concurrently (default: 10)")
    parser.add_argument("--chunk-concurrency", type=int, default=20,
                       help="Max chunks to process concurrently per article (default: 20)")
    parser.add_argument("--checkpoint", type=str, default="2000s_checkpoint.json",
                       help="Checkpoint file for resuming (default: 2000s_checkpoint.json)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    processor = Decade2000sBatchProcessor(
        batch_size=args.batch_size,
        chunk_concurrency=args.chunk_concurrency,
        checkpoint_file=args.checkpoint
    )
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("\nThis will process ~1,074 articles from 2000-2009.")
    print("Estimated time: 1-2 hours with batch processing")
    print("Estimated cost: ~$8-12 with GPT-4o-mini")
    print("\nFeatures:")
    print("  - Batch processing for faster performance")
    print("  - Checkpoint support for resuming")
    print("  - Error handling and retry logic")
    print("  - Detailed progress tracking")
    
    if not args.resume and os.path.exists(args.checkpoint):
        print(f"\n⚠️  Checkpoint file exists: {args.checkpoint}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() != 'y':
            response = input("Start fresh? This will overwrite the checkpoint. (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
            # Clear checkpoint
            processor.completed_years = set()
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        await processor.run()
        print("\n✅ Decade processing completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print(f"Progress saved to {processor.checkpoint_file}")
        print("Run with --resume to continue from where you left off")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())