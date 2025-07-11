#!/usr/bin/env python3
"""
Unified script for processing Foreign Affairs articles using the REM RAG system.

By default uses async processing for faster results. Add --batch flag to use:
- ReadingCycle: OpenAI Batch API for 50% cost savings (slower)
- REMCycle: Batch processing for REM dreams

Usage:
    # Process a single year (async by default)
    python scripts/process_articles.py --year 2000
    
    # Process a decade with batch API for cost savings
    python scripts/process_articles.py --decade 2000s --batch
    
    # Process specific articles
    python scripts/process_articles.py --files article1.json article2.json
    
    # Resume from checkpoint
    python scripts/process_articles.py --decade 2000s --resume
"""

import os
import sys
import json
import glob
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import time
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle import ReadingCycle
from rem_rag_backend.src.core.rem_cycle import REMCycle
from rem_rag_backend.src.config import REM_SCALING_FACTOR, OPENAI_API_KEY


class ArticleProcessor:
    """Unified processor for Foreign Affairs articles using batch API."""
    
    def __init__(self, checkpoint_file: str = None, use_batch: bool = False):
        """
        Initialize the processor.
        
        Args:
            checkpoint_file: Path to checkpoint file for resuming
            use_batch: Whether to use batch API (False by default) or async processing (True)
        """
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        self.use_batch = use_batch
        
        # Initialize batch processors
        self.reading_cycle = ReadingCycle(
            self.llm,
            self.structured_llm,
            self.store,
            self.chunker,
            api_key=OPENAI_API_KEY
        )
        
        self.rem_cycle = REMCycle(self.llm, self.store)
        
        self.stats = defaultdict(lambda: defaultdict(int))
        self.checkpoint_file = checkpoint_file or "processing_checkpoint.json"
        self.completed_items = self.load_checkpoint()
        
    def load_checkpoint(self) -> set:
        """Load checkpoint to resume processing."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed_items', []))
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
        return set()
    
    def save_checkpoint(self, item: str):
        """Save checkpoint after completing an item."""
        self.completed_items.add(item)
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'completed_items': list(self.completed_items),
                    'stats': dict(self.stats),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    def load_year_articles(self, year: int) -> List[Dict[str, Any]]:
        """Load all articles from a specific year."""
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
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def load_articles_from_files(self, files: List[str]) -> List[Dict[str, Any]]:
        """Load articles from specific files."""
        articles = []
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Try to extract year from filename or content
                    year = article_data.get('year', 2024)
                    if not year and '_' in os.path.basename(filepath):
                        try:
                            year = int(os.path.basename(filepath).split('_')[0])
                        except:
                            year = 2024
                    
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title') or 'Unknown',
                        'author': article_data.get('author') or 'Unknown',
                        'year': year,
                        'article_id': f"custom-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:
                        articles.append(article)
            except Exception as e:
                print(f"  ⚠️  Error loading {filepath}: {e}")
        
        return articles
    
    async def process_articles(self, articles: List[Dict], context_name: str = "batch") -> Dict:
        """Process a batch of articles."""
        print(f"\n📚 [{datetime.now().strftime('%H:%M:%S')}] Processing {len(articles)} articles")
        start_time = time.time()
        
        try:
            if self.use_batch:
                # Process articles using batch API
                print(f"\n📖 [{datetime.now().strftime('%H:%M:%S')}] Processing articles with Batch API...")
                print("💡 This will submit all requests as a single batch job")
                
                batch_results = await self.reading_cycle.process_articles_batch(articles)
                
                if 'error' in batch_results:
                    print(f"❌ Batch processing failed: {batch_results['error']}")
                    return batch_results
                
                # Update stats
                self.stats[context_name]['articles_processed'] = batch_results.get('total_articles', 0)
                self.stats[context_name]['total_requests'] = batch_results.get('total_requests', 0)
                self.stats[context_name]['syntheses_stored'] = batch_results.get('syntheses_stored', 0)
                self.stats[context_name]['summaries_stored'] = batch_results.get('summaries_stored', 0)
                self.stats[context_name]['batch_id'] = batch_results.get('batch_id', 'N/A')
                
                processing_time = batch_results.get('processing_time', 0)
                print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] Batch processing complete!")
                print(f"   Batch ID: {batch_results.get('batch_id', 'N/A')}")
                print(f"   Articles: {batch_results.get('total_articles', 0)}")
                print(f"   API requests: {batch_results.get('total_requests', 0)}")
                print(f"   Syntheses: {batch_results.get('syntheses_stored', 0)}")
                print(f"   Summaries: {batch_results.get('summaries_stored', 0)}")
                print(f"   Time: {processing_time/60:.1f} minutes")
                print(f"   💰 Cost savings: ~50% compared to regular API")
                
                return batch_results
            else:
                # Process articles using async API
                print(f"\n⚡ [{datetime.now().strftime('%H:%M:%S')}] Processing articles with Async API...")
                print("💡 This will process requests concurrently for faster results")
                
                async_results = await self.reading_cycle.process_articles_async(articles)
                
                if 'error' in async_results:
                    print(f"❌ Async processing failed: {async_results['error']}")
                    return async_results
                
                # Update stats
                self.stats[context_name]['articles_processed'] = async_results.get('total_articles', 0)
                self.stats[context_name]['total_requests'] = async_results.get('total_requests', 0)
                self.stats[context_name]['syntheses_stored'] = async_results.get('syntheses_stored', 0)
                self.stats[context_name]['summaries_stored'] = async_results.get('summaries_stored', 0)
                
                processing_time = async_results.get('processing_time', 0)
                print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] Async processing complete!")
                print(f"   Articles: {async_results.get('total_articles', 0)}")
                print(f"   API requests: {async_results.get('total_requests', 0)}")
                print(f"   Syntheses: {async_results.get('syntheses_stored', 0)}")
                print(f"   Summaries: {async_results.get('summaries_stored', 0)}")
                print(f"   Time: {processing_time/60:.1f} minutes")
                print(f"   ⚡ Processing at full speed (no batch discounts)")
                
                return async_results
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise
    
    async def process_year(self, year: int):
        """Process all articles from a specific year."""
        if f"year_{year}" in self.completed_items:
            print(f"✅ Year {year} already processed, skipping...")
            return
        
        print(f"\n\n{'='*60}")
        print(f"📅 [{datetime.now().strftime('%H:%M:%S')}] YEAR {year}")
        print(f"{'='*60}")
        
        # Load articles
        articles = self.load_year_articles(year)
        print(f"📚 [{datetime.now().strftime('%H:%M:%S')}] Loaded {len(articles)} articles from {year}")
        
        if not articles:
            print(f"⚠️  No articles found for {year}")
            return
        
        # Process articles
        await self.process_articles(articles, f"year_{year}")
        
        # Run REM cycle
        print(f"\n🌙 [{datetime.now().strftime('%H:%M:%S')}] Running REM cycle for {year}...")
        rem_start = time.time()
        
        # Calculate number of REM cycles
        sample_count = self.store.collection.count()
        num_rem_cycles = max(1, int(sample_count * REM_SCALING_FACTOR))
        
        print(f"   Total nodes: {sample_count}")
        print(f"   REM cycles to run: {num_rem_cycles}")
        
        # Run REM cycle (batch or async based on setting)
        if self.use_batch:
            rem_ids = self.rem_cycle.run_cycle(current_year=year)
        else:
            rem_ids = await self.rem_cycle.run_cycle_async(current_year=year)
        
        self.stats[f"year_{year}"]['rem_nodes'] = len(rem_ids)
        
        rem_time = time.time() - rem_start
        print(f"✨ [{datetime.now().strftime('%H:%M:%S')}] Created {len(rem_ids)} REM insights in {rem_time/60:.1f} minutes")
        
        # Save checkpoint
        self.save_checkpoint(f"year_{year}")
    
    async def process_decade(self, decade_start: int):
        """Process a full decade of articles."""
        print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] Starting {decade_start}s Decade Processing")
        print("=" * 70)
        print(f"Processing articles from {decade_start}-{decade_start+9} with yearly REM cycles")
        print(f"REM scaling factor: {REM_SCALING_FACTOR} (n/{int(1/REM_SCALING_FACTOR)} dreams)")
        
        if self.completed_items:
            print(f"Resuming from checkpoint. Already completed: {sorted(self.completed_items)}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Process each year
        years_to_process = [y for y in range(decade_start, decade_start + 10) 
                           if f"year_{y}" not in self.completed_items]
        
        for year in years_to_process:
            try:
                await self.process_year(year)
            except Exception as e:
                print(f"❌ Year {year} failed: {e}")
                # Continue with next year instead of stopping
                continue
        
        # Final analysis
        elapsed = time.time() - start_time
        print(f"\n\n{'='*70}")
        print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] DECADE PROCESSING COMPLETE")
        print(f"{'='*70}")
        self.print_final_stats(elapsed)
    
    def print_final_stats(self, elapsed_time: float):
        """Print comprehensive statistics."""
        total_articles = sum(self.stats[y].get('articles_processed', 0) for y in self.stats)
        total_requests = sum(self.stats[y].get('total_requests', 0) for y in self.stats)
        total_syntheses = sum(self.stats[y].get('syntheses_stored', 0) for y in self.stats)
        total_summaries = sum(self.stats[y].get('summaries_stored', 0) for y in self.stats)
        total_rem = sum(self.stats[y].get('rem_nodes', 0) for y in self.stats)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total articles processed: {total_articles}")
        print(f"  • Total API requests: {total_requests}")
        print(f"  • Total syntheses stored: {total_syntheses}")
        print(f"  • Total summaries stored: {total_summaries}")
        print(f"  • Total REM insights: {total_rem}")
        print(f"  • Processing time: {elapsed_time/60:.1f} minutes")
        print(f"  • 💰 Estimated savings: ~50% compared to regular API")
        
        # Query database for final counts by node type
        print(f"\n🗄️  Database Node Counts:")
        try:
            for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
                count = self.store.collection.count(where={"node_type": node_type})
                print(f"    • {node_type}: {count} nodes")
        except Exception as e:
            print(f"    ❌ Error querying database: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Foreign Affairs articles using REM RAG batch system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single year
  python scripts/process_articles.py --year 2000
  
  # Process a decade
  python scripts/process_articles.py --decade 2000s
  
  # Process specific files
  python scripts/process_articles.py --files data/article1.json data/article2.json
  
  # Resume from checkpoint
  python scripts/process_articles.py --decade 2000s --resume
        """
    )
    
    parser.add_argument("--year", type=int, help="Process a specific year")
    parser.add_argument("--decade", type=str, help="Process a decade (e.g., '2000s')")
    parser.add_argument("--files", nargs="+", help="Process specific article files")
    parser.add_argument("--checkpoint", type=str, help="Custom checkpoint file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API for 50% cost savings (slower)")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    # Validate arguments
    if not any([args.year, args.decade, args.files]):
        parser.error("Must specify --year, --decade, or --files")
    
    # Create processor
    processor = ArticleProcessor(checkpoint_file=args.checkpoint, use_batch=args.batch)
    
    # Handle different processing modes
    if args.year:
        await processor.process_year(args.year)
    
    elif args.decade:
        # Parse decade
        decade_str = args.decade.lower().rstrip('s')
        try:
            decade_start = int(decade_str)
            if decade_start < 1900 or decade_start > 2100:
                raise ValueError("Invalid decade")
        except:
            parser.error(f"Invalid decade format: {args.decade}. Use format like '2000s' or '2000'")
        
        await processor.process_decade(decade_start)
    
    elif args.files:
        # Process specific files
        articles = processor.load_articles_from_files(args.files)
        if articles:
            await processor.process_articles(articles, "custom_batch")
            
            # Run REM cycle
            print(f"\n🌙 [{datetime.now().strftime('%H:%M:%S')}] Running REM cycle...")
            if processor.use_batch:
                rem_ids = processor.rem_cycle.run_cycle()
            else:
                rem_ids = await processor.rem_cycle.run_cycle_async()
            print(f"✨ [{datetime.now().strftime('%H:%M:%S')}] Created {len(rem_ids)} REM insights")
    
    print("\n✅ Processing completed!")


if __name__ == "__main__":
    asyncio.run(main())