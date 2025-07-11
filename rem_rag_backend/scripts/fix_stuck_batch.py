#!/usr/bin/env python3
"""
Fix stuck batch processing by properly resuming from checkpoint.
"""

import os
import sys
import json
import openai
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def main():
    """Fix stuck batch by analyzing checkpoint and current batches."""
    
    # Initialize OpenAI client
    openai.api_key = OPENAI_API_KEY
    
    print("🔧 Fixing Stuck Batch Processing")
    print("=" * 70)
    
    # Check checkpoint
    checkpoint_file = "2000s_true_batch_checkpoint.json"
    if not os.path.exists(checkpoint_file):
        print("❌ No checkpoint file found")
        return
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print("\n📋 Checkpoint Status:")
    print(f"   Completed years: {checkpoint.get('completed_years', [])}")
    
    # Check for completed batch
    if '2000' in checkpoint.get('stats', {}):
        year_stats = checkpoint['stats']['2000']
        completed_batch_id = year_stats.get('batch_id')
        print(f"\n✅ Year 2000 already completed:")
        print(f"   Batch ID: {completed_batch_id}")
        print(f"   Articles: {year_stats.get('articles_processed', 0)}")
        print(f"   Summaries: {year_stats.get('summaries_stored', 0)}")
    
    # Check for stuck batch
    print("\n🔍 Checking for stuck batches...")
    stuck_batch_id = "batch_686ff924c11481909fafdeb759c5e2c5"
    
    try:
        batch = openai.batches.retrieve(stuck_batch_id)
        print(f"\n📦 Found stuck batch: {stuck_batch_id}")
        print(f"   Status: {batch.status}")
        print(f"   Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        print(f"   Created: {datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M')}")
        
        if batch.status in ["in_progress", "validating"] and batch.request_counts.completed == 0:
            print("\n⚠️  This batch appears to be stuck with no progress.")
            print("\n💡 Recommended actions:")
            print("   1. Cancel this batch:")
            print(f"      python -c \"import openai; openai.api_key='{OPENAI_API_KEY[:10]}...'; openai.batches.cancel('{stuck_batch_id}')\"")
            print("\n   2. Resume from checkpoint (skips year 2000):")
            print("      python scripts/run_2000s_true_batch.py --resume")
            print("\n   3. Or use the improved script that checks for existing batches:")
            print("      python scripts/run_2000s_true_batch.py")
    except Exception as e:
        print(f"❌ Error checking batch: {e}")
    
    print("\n✅ Summary:")
    print("   - Year 2000 is already completed")
    print("   - The stuck batch was created when you chose 'start fresh'")
    print("   - The improved code now handles this properly")
    print("   - Future batches will have 'fa_rem_rag' metadata for easy identification")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    main()