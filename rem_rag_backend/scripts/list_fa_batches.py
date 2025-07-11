#!/usr/bin/env python3
"""
List all Foreign Affairs REM RAG batches using metadata filtering.
"""

import os
import sys
from datetime import datetime
import openai

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def list_fa_batches(limit=20):
    """List all batches with fa_rem_rag metadata."""
    
    # Initialize OpenAI client
    openai.api_key = OPENAI_API_KEY
    
    print("📋 Foreign Affairs REM RAG Batches")
    print("=" * 120)
    print(f"{'ID':<40} {'Status':<12} {'Type':<15} {'Created':<20} {'Progress':<15} {'Year':<6}")
    print("-" * 120)
    
    try:
        # Get all recent batches
        batches = openai.batches.list(limit=limit)
        
        fa_batches = []
        for batch in batches.data:
            # Check if this is an FA batch by looking at metadata
            if batch.metadata and isinstance(batch.metadata, dict):
                if batch.metadata.get('project') == 'fa_rem_rag':
                    fa_batches.append(batch)
        
        if not fa_batches:
            print("\nNo Foreign Affairs batches found.")
            print("\n💡 Note: Metadata tagging was just added. Older batches won't have metadata.")
            return
        
        # Sort by creation time (newest first)
        fa_batches.sort(key=lambda x: x.created_at, reverse=True)
        
        for batch in fa_batches:
            created = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M')
            
            # Get batch type and year from metadata
            batch_type = batch.metadata.get('type', 'unknown')
            year = batch.metadata.get('year', 'N/A')
            
            # Get progress
            counts = batch.request_counts
            if counts and hasattr(counts, 'total') and counts.total > 0:
                progress = f"{counts.completed}/{counts.total} ({(counts.completed/counts.total)*100:.0f}%)"
            else:
                progress = "N/A"
            
            # Color code status
            status = batch.status
            if status == "completed":
                status_display = f"✅ {status}"
            elif status in ["in_progress", "validating", "finalizing"]:
                status_display = f"⏳ {status}"
            elif status == "failed":
                status_display = f"❌ {status}"
            else:
                status_display = f"   {status}"
            
            print(f"{batch.id:<40} {status_display:<20} {batch_type:<15} {created:<20} {progress:<15} {year:<6}")
        
        print(f"\n📊 Found {len(fa_batches)} Foreign Affairs batches")
        
        # Show metadata details for the most recent batch
        if fa_batches:
            latest = fa_batches[0]
            print(f"\n📄 Latest batch metadata:")
            for key, value in latest.metadata.items():
                print(f"   {key}: {value}")
        
        print("\n💡 Use 'python scripts/check_batch_status.py <batch_id>' for detailed status")
        print("💡 Use 'python scripts/recover_batch_results.py <batch_id>' to download results")
        
    except Exception as e:
        print(f"\n❌ Error listing batches: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="List Foreign Affairs REM RAG batches")
    parser.add_argument("--limit", type=int, default=20, 
                        help="Maximum number of batches to check (default: 20)")
    parser.add_argument("--all", action="store_true",
                        help="Check more batches (up to 100)")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    limit = 100 if args.all else args.limit
    list_fa_batches(limit)


if __name__ == "__main__":
    main()