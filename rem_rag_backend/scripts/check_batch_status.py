#!/usr/bin/env python3
"""
Check the status of an OpenAI batch job.
"""

import sys
import openai
from datetime import datetime

# Add the project root to Python path
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def check_batch(batch_id: str):
    """Check the status of a batch."""
    openai.api_key = OPENAI_API_KEY
    
    try:
        batch = openai.batches.retrieve(batch_id)
        
        print(f"\n📊 Batch Status for: {batch_id}")
        print("=" * 70)
        print(f"Status: {batch.status}")
        print(f"Created: {datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hasattr(batch, 'expires_at') and batch.expires_at:
            print(f"Expires: {datetime.fromtimestamp(batch.expires_at).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Request counts
        counts = batch.request_counts
        if counts:
            print(f"\n📈 Progress:")
            print(f"  Total requests: {counts.total if hasattr(counts, 'total') else 0}")
            print(f"  Completed: {counts.completed if hasattr(counts, 'completed') else 0}")
            print(f"  Failed: {counts.failed if hasattr(counts, 'failed') else 0}")
            
            if hasattr(counts, 'total') and counts.total > 0:
                progress = (counts.completed / counts.total) * 100
                print(f"  Progress: {progress:.1f}%")
        
        if batch.status == "completed":
            print(f"\n✅ Batch completed!")
            print(f"Output file ID: {batch.output_file_id}")
            if hasattr(batch, 'error_file_id') and batch.error_file_id:
                print(f"Error file ID: {batch.error_file_id}")
                
        elif batch.status in ["failed", "expired", "cancelled"]:
            print(f"\n❌ Batch {batch.status}")
            if hasattr(batch, 'errors') and batch.errors:
                print(f"Errors: {batch.errors}")
        
        return batch
        
    except Exception as e:
        print(f"❌ Error checking batch: {e}")
        return None


def main():
    """Main function."""
    if len(sys.argv) > 1:
        batch_id = sys.argv[1]
    else:
        batch_id = input("Enter batch ID to check: ")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    check_batch(batch_id)


if __name__ == "__main__":
    main()