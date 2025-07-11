#!/usr/bin/env python3
"""
TRUE BATCH processing version of the 2000s decade processor.

This script uses OpenAI's Batch API for 50% cost savings:
1. Processes entire years as single batch jobs
2. All synthesis and summary generation uses Batch API
3. REM cycles also use Batch API
4. Trades immediate results for significant cost savings
"""

import os
import sys
import json
import glob
import asyncio
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


class Decade2000sTrueBatchProcessor:
    """TRUE batch processing version using OpenAI Batch API"""
    
    def __init__(self, checkpoint_file: str = None):
        """
        Initialize the processor.
        
        Args:
            checkpoint_file: Path to checkpoint file for resuming
        """
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        
        # Initialize batch processors
        self.batch_reader = ReadingCycle(
            self.llm,
            self.structured_llm,
            self.store,
            self.chunker,
            api_key=OPENAI_API_KEY
        )
        
        # REM cycle already uses batch API
        self.rem_cycle = REMCycle(self.llm, self.store)
        
        self.stats = defaultdict(lambda: defaultdict(int))
        self.checkpoint_file = checkpoint_file or "2000s_true_batch_checkpoint.json"
        self.completed_years = self.load_checkpoint()
        
    def load_checkpoint(self) -> set:
        """Load checkpoint to resume processing"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    # Load stats if available
                    if 'stats' in data:
                        for year_str, year_stats in data['stats'].items():
                            self.stats[int(year_str)] = defaultdict(int, year_stats)
                    return set(data.get('completed_years', []))
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
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
            print(f"❌ Failed to save checkpoint: {e}")
    
    async def run(self):
        """Execute the decade processing with TRUE batch API"""
        print("🚀 Starting 2000s Decade TRUE BATCH Processing")
        print("=" * 70)
        print("Using OpenAI Batch API for 50% cost savings")
        print(f"Processing articles from 2000-2009 with yearly REM cycles")
        print(f"REM scaling factor: {REM_SCALING_FACTOR} (n/{int(1/REM_SCALING_FACTOR)} dreams)")
        
        if self.completed_years:
            print(f"Resuming from checkpoint. Already completed: {sorted(self.completed_years)}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Process each year
        years_to_process = [y for y in range(2000, 2010) if y not in self.completed_years]
        
        for year in years_to_process:
            print(f"\n\n{'='*60}")
            print(f"📅 YEAR {year}")
            print(f"{'='*60}")
            
            try:
                await self.process_year(year)
                self.save_checkpoint(year)
            except Exception as e:
                print(f"❌ Year {year} failed: {e}")
                # Continue with next year instead of stopping
                continue
        
        # Final analysis
        elapsed = time.time() - start_time
        print(f"\n\n{'='*70}")
        print(f"📊 DECADE PROCESSING COMPLETE")
        print(f"{'='*70}")
        self.print_final_stats(elapsed)
    
    async def process_year(self, year: int):
        """Process all articles from a specific year using TRUE batch API"""
        # Load articles
        articles = self.load_year_articles(year)
        print(f"\n📚 Loaded {len(articles)} articles from {year}")
        
        if not articles:
            print(f"⚠️  No articles found for {year}")
            return
        
        # Process articles using TRUE batch API
        print(f"\n📖 Processing {year} articles with Batch API...")
        print("💡 This will submit all requests as a single batch job")
        year_start = time.time()
        
        try:
            # Check if we have an existing batch ID for this year
            existing_batch_id = None
            if year in self.stats and 'batch_id' in self.stats[year]:
                existing_batch_id = self.stats[year]['batch_id']
                if existing_batch_id and existing_batch_id != 'N/A':
                    print(f"📋 Found existing batch ID: {existing_batch_id}")
            
            # Process all articles using true batch reader
            batch_results = await self.batch_reader.process_articles_batch(articles, existing_batch_id)
            
            if 'error' in batch_results:
                print(f"❌ Batch processing failed: {batch_results['error']}")
                self.stats[year]['batch_failed'] = True
                return
            
            # Update stats
            self.stats[year]['articles_processed'] = batch_results.get('total_articles', 0)
            self.stats[year]['total_requests'] = batch_results.get('total_requests', 0)
            self.stats[year]['syntheses_stored'] = batch_results.get('syntheses_stored', 0)
            self.stats[year]['summaries_stored'] = batch_results.get('summaries_stored', 0)
            self.stats[year]['batch_id'] = batch_results.get('batch_id', 'N/A')
            
            year_time = batch_results.get('processing_time', 0)
            print(f"\n✅ Year {year} batch processing complete!")
            print(f"   Batch ID: {batch_results.get('batch_id', 'N/A')}")
            print(f"   Articles: {batch_results.get('total_articles', 0)}")
            print(f"   API requests: {batch_results.get('total_requests', 0)}")
            print(f"   Syntheses: {batch_results.get('syntheses_stored', 0)}")
            print(f"   Summaries: {batch_results.get('summaries_stored', 0)}")
            print(f"   Time: {year_time/60:.1f} minutes")
            print(f"   💰 Cost savings: ~50% compared to regular API")
            
        except Exception as e:
            print(f"❌ Batch processing failed for {year}: {e}")
            self.stats[year]['batch_failed'] = True
            raise
        
        # Run REM cycle at end of year (also uses batch API)
        print(f"\n🌙 Running REM cycle for {year}...")
        rem_retry_count = 0
        max_rem_retries = 3
        
        while rem_retry_count < max_rem_retries:
            try:
                rem_start = time.time()
                
                # Calculate number of REM cycles
                sample_count = self.store.collection.count()
                num_rem_cycles = max(1, int(sample_count * REM_SCALING_FACTOR))
                
                print(f"   Total nodes: {sample_count}")
                print(f"   REM cycles to run: {num_rem_cycles}")
                
                # Run batch REM
                rem_ids = self.rem_cycle.run_cycle(current_year=year)
                
                self.stats[year]['rem_nodes'] = len(rem_ids)
                
                rem_time = time.time() - rem_start
                print(f"✨ Created {len(rem_ids)} REM insights in {rem_time/60:.1f} minutes")
                break  # Success, exit retry loop
                
            except Exception as e:
                rem_retry_count += 1
                print(f"❌ REM cycle failed for {year} (attempt {rem_retry_count}/{max_rem_retries}): {e}")
                
                if rem_retry_count < max_rem_retries:
                    wait_time = rem_retry_count * 30  # 30s, 60s, 90s
                    print(f"⏳ Retrying REM cycle in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ REM cycle permanently failed after {max_rem_retries} attempts")
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
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def print_final_stats(self, elapsed_time: float):
        """Print comprehensive statistics"""
        total_articles = sum(self.stats[y]['articles_processed'] for y in self.stats)
        total_requests = sum(self.stats[y]['total_requests'] for y in self.stats)
        total_syntheses = sum(self.stats[y]['syntheses_stored'] for y in self.stats)
        total_summaries = sum(self.stats[y]['summaries_stored'] for y in self.stats)
        total_rem = sum(self.stats[y]['rem_nodes'] for y in self.stats)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total articles processed: {total_articles}")
        print(f"  • Total API requests: {total_requests}")
        print(f"  • Total syntheses stored: {total_syntheses}")
        print(f"  • Total summaries stored: {total_summaries}")
        print(f"  • Total REM insights: {total_rem}")
        print(f"  • Processing time: {elapsed_time/60:.1f} minutes")
        print(f"  • 💰 Estimated savings: ~50% compared to regular API")
        
        print(f"\n📅 Year-by-Year Breakdown:")
        for year in sorted(self.stats.keys()):
            stats = self.stats[year]
            print(f"\n  {year}:")
            print(f"    • Batch ID: {stats.get('batch_id', 'N/A')}")
            print(f"    • Articles: {stats['articles_processed']}")
            print(f"    • Syntheses: {stats['syntheses_stored']}")
            print(f"    • Summaries: {stats['summaries_stored']}")
            print(f"    • REM: {stats['rem_nodes']} insights")
            if stats.get('batch_failed'):
                print(f"    • ❌ Batch processing failed")
        
        # Query database for final counts by node type
        print(f"\n🗄️  Database Node Counts:")
        try:
            for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
                count = self.store.collection.count(where={"node_type": node_type})
                print(f"    • {node_type}: {count} nodes")
        except Exception as e:
            print(f"    ❌ Error querying database: {e}")


async def main():
    """Run the 2000s decade TRUE batch processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process 2000s Foreign Affairs articles with TRUE OpenAI Batch API")
    parser.add_argument("--checkpoint", type=str, default="2000s_true_batch_checkpoint.json",
                       help="Checkpoint file for resuming (default: 2000s_true_batch_checkpoint.json)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    processor = Decade2000sTrueBatchProcessor(
        checkpoint_file=args.checkpoint
    )
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("\nThis will process ~1,074 articles from 2000-2009 using OpenAI Batch API.")
    print("\n💡 Key differences from hybrid batch:")
    print("  - ALL synthesis/summary generation uses Batch API")
    print("  - 50% cost savings on all OpenAI API calls")
    print("  - Each year processed as a single batch job")
    print("  - May take several hours per year (up to 24h max)")
    print("\nEstimated cost: ~$4-6 for the decade (50% savings)")
    
    if not args.resume and os.path.exists(args.checkpoint):
        print(f"\n⚠️  Checkpoint file exists: {args.checkpoint}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() != 'y':
            response = input("Start fresh? This will overwrite the checkpoint. (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
            # Clear checkpoint properly
            processor.completed_years = set()
            processor.stats = defaultdict(lambda: defaultdict(int))
            # Backup old checkpoint
            if os.path.exists(processor.checkpoint_file):
                backup_file = f"{processor.checkpoint_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(processor.checkpoint_file, backup_file)
                print(f"✅ Backed up old checkpoint to: {backup_file}")
    
    response = input("\nProceed with TRUE batch processing? (y/n): ")
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