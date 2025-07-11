#!/usr/bin/env python3
"""
List recent OpenAI batch jobs.
"""

import openai
from datetime import datetime
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def list_batches(limit=10):
    """List recent batches."""
    openai.api_key = OPENAI_API_KEY
    
    try:
        batches = openai.batches.list(limit=limit)
        
        print(f"\n📋 Recent Batches (showing up to {limit})")
        print("=" * 100)
        print(f"{'ID':<40} {'Status':<12} {'Created':<20} {'Progress':<15}")
        print("-" * 100)
        
        for batch in batches.data:
            created = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M')
            
            # Get progress
            counts = batch.request_counts
            if counts and hasattr(counts, 'total') and counts.total > 0:
                progress = f"{counts.completed}/{counts.total} ({(counts.completed/counts.total)*100:.0f}%)"
            else:
                progress = "N/A"
            
            print(f"{batch.id:<40} {batch.status:<12} {created:<20} {progress:<15}")
        
        print("\n💡 Use 'python scripts/check_batch_status.py <batch_id>' to see details")
        
    except Exception as e:
        print(f"❌ Error listing batches: {e}")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    limit = 10
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except:
            pass
    
    list_batches(limit)