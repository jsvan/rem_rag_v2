#!/usr/bin/env python3
"""
Cancel stuck batch and properly resume from checkpoint.
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
    # Initialize OpenAI client
    openai.api_key = OPENAI_API_KEY
    
    # Check checkpoint
    checkpoint_file = "2000s_true_batch_checkpoint.json"
    if not os.path.exists(checkpoint_file):
        print("❌ No checkpoint file found")
        return
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print("📋 Current checkpoint status:")
    print(f"   Completed years: {checkpoint.get('completed_years', [])}")
    
    # Show current batch statuses
    print("\n🔍 Checking recent batches...")
    stuck_batch_id = "batch_686ff924c11481909fafdeb759c5e2c5"
    
    try:
        batch = openai.batches.retrieve(stuck_batch_id)
        print(f"\n📦 Stuck batch: {stuck_batch_id}")
        print(f"   Status: {batch.status}")
        print(f"   Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        
        if batch.status in ["in_progress", "validating"]:
            response = input("\n❓ Cancel this batch and resume from checkpoint? (y/n): ")
            if response.lower() == 'y':
                print("🚫 Cancelling batch...")
                openai.batches.cancel(stuck_batch_id)
                print("✅ Batch cancelled")
                
                print("\n💡 To resume processing from year 2001, run:")
                print("   python scripts/run_2000s_true_batch.py --resume")
        else:
            print(f"\n✅ Batch is already {batch.status}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Show the completed batch info
    if '2000' in checkpoint.get('stats', {}):
        completed_batch_id = checkpoint['stats']['2000'].get('batch_id')
        if completed_batch_id:
            print(f"\n✅ Year 2000 was already completed with batch: {completed_batch_id}")
            print(f"   Articles: {checkpoint['stats']['2000'].get('articles_processed', 0)}")
            print(f"   Summaries stored: {checkpoint['stats']['2000'].get('summaries_stored', 0)}")

if __name__ == "__main__":
    main()