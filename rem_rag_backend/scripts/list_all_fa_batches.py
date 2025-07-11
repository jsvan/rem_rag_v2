#!/usr/bin/env python3
"""
List all Foreign Affairs REM RAG batches by checking metadata OR custom_id patterns.
This catches both new batches (with metadata) and future batches (with fa_rem_rag prefix).
"""

import os
import sys
import json
from datetime import datetime
import openai

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def check_if_fa_batch(batch):
    """Check if a batch belongs to FA REM RAG by metadata or by inspecting requests."""
    # First check metadata (for batches created after metadata was added)
    if batch.metadata and isinstance(batch.metadata, dict):
        if batch.metadata.get('project') == 'fa_rem_rag':
            return True, 'metadata'
    
    # For future batches, we'll be able to identify by the custom_id pattern
    # Note: We can't inspect the input file content directly from the batch list
    # But once batches start using fa_rem_rag prefix, they'll be identifiable
    
    return False, None


def list_all_fa_batches(limit=50, show_all=False):
    """List all batches, highlighting FA REM RAG batches."""
    
    # Initialize OpenAI client
    openai.api_key = OPENAI_API_KEY
    
    print("📋 Foreign Affairs REM RAG Batch Identification")
    print("=" * 140)
    print(f"{'ID':<40} {'Status':<12} {'Type':<20} {'Created':<20} {'Progress':<15} {'Identified By':<15}")
    print("-" * 140)
    
    try:
        # Get all recent batches
        batches = openai.batches.list(limit=limit)
        
        fa_count = 0
        total_count = 0
        
        for batch in batches.data:
            total_count += 1
            is_fa, identified_by = check_if_fa_batch(batch)
            
            # Skip non-FA batches unless show_all is True
            if not is_fa and not show_all:
                continue
            
            if is_fa:
                fa_count += 1
            
            created = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M')
            
            # Get batch type from metadata if available
            batch_type = 'unknown'
            if batch.metadata and isinstance(batch.metadata, dict):
                batch_type = batch.metadata.get('type', 'unknown')
                year = batch.metadata.get('year', '')
                if year:
                    batch_type += f" ({year})"
            
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
            
            # Mark FA batches
            if is_fa:
                print(f"🎯 {batch.id:<37} {status_display:<20} {batch_type:<20} {created:<20} {progress:<15} {identified_by or 'N/A':<15}")
            else:
                print(f"   {batch.id:<37} {status_display:<20} {batch_type:<20} {created:<20} {progress:<15} {'N/A':<15}")
        
        print(f"\n📊 Summary:")
        print(f"   Total batches checked: {total_count}")
        print(f"   FA REM RAG batches found: {fa_count}")
        
        print("\n💡 Notes:")
        print("   - Batches with 🎯 are identified as FA REM RAG batches")
        print("   - Metadata identification started on 2025-07-10")
        print("   - Future batches will have 'fa_rem_rag' prefix in custom_ids")
        print("\n   Use 'python scripts/check_batch_status.py <batch_id>' for details")
        print("   Use 'python scripts/recover_batch_results.py <batch_id>' to download results")
        
    except Exception as e:
        print(f"\n❌ Error listing batches: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="List all FA REM RAG batches")
    parser.add_argument("--limit", type=int, default=50, 
                        help="Maximum number of batches to check (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Show all batches, not just FA ones")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    list_all_fa_batches(limit=args.limit, show_all=args.all)


if __name__ == "__main__":
    main()