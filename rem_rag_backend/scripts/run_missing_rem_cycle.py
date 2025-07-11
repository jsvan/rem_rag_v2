#!/usr/bin/env python3
"""
Run missing REM cycle for a year that completed reading but not REM.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from collections import defaultdict

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient
from rem_rag_backend.src.core.rem_cycle import REMCycle
from rem_rag_backend.src.config import REM_SCALING_FACTOR, OPENAI_API_KEY


async def run_missing_rem_cycle(year: int = 2000):
    """Run REM cycle for a specific year."""
    
    print(f"🌙 Running Missing REM Cycle for Year {year}")
    print("=" * 70)
    
    # Initialize components
    llm = LLMClient()
    store = REMVectorStore()
    rem_cycle = REMCycle(llm, store)
    
    # Check current state
    checkpoint_file = "2000s_true_batch_checkpoint.json"
    stats = {}
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            if 'stats' in data and str(year) in data['stats']:
                stats = data['stats'][str(year)]
    
    print(f"\n📊 Current Status for {year}:")
    print(f"   Articles processed: {stats.get('articles_processed', 0)}")
    print(f"   Summaries stored: {stats.get('summaries_stored', 0)}")
    print(f"   REM nodes: {stats.get('rem_nodes', 0)}")
    
    if stats.get('rem_nodes', 0) > 0:
        print(f"\n✅ REM cycle already completed for {year}")
        response = input("Run again anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Calculate number of REM cycles
    sample_count = store.collection.count()
    num_rem_cycles = max(1, int(sample_count * REM_SCALING_FACTOR))
    
    print(f"\n📈 REM Cycle Planning:")
    print(f"   Total nodes in database: {sample_count}")
    print(f"   REM scaling factor: {REM_SCALING_FACTOR}")
    print(f"   REM cycles to run: {num_rem_cycles}")
    
    response = input("\nProceed with REM cycle? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run REM cycle
    print(f"\n🚀 Starting REM cycle for {year}...")
    rem_start = time.time()
    
    try:
        rem_ids = rem_cycle.run_cycle(current_year=year)
        
        rem_time = time.time() - rem_start
        print(f"\n✨ Created {len(rem_ids)} REM insights in {rem_time/60:.1f} minutes")
        
        # Update checkpoint
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Update REM nodes count
            if 'stats' not in checkpoint_data:
                checkpoint_data['stats'] = {}
            if str(year) not in checkpoint_data['stats']:
                checkpoint_data['stats'][str(year)] = {}
            
            checkpoint_data['stats'][str(year)]['rem_nodes'] = len(rem_ids)
            checkpoint_data['last_updated'] = datetime.now().isoformat()
            
            # Save updated checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"✅ Updated checkpoint with {len(rem_ids)} REM nodes")
        
    except Exception as e:
        print(f"❌ REM cycle failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run missing REM cycle for a year")
    parser.add_argument("--year", type=int, default=2000,
                        help="Year to run REM cycle for (default: 2000)")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    await run_missing_rem_cycle(args.year)


if __name__ == "__main__":
    asyncio.run(main())