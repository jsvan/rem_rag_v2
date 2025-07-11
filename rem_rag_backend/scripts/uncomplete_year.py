#!/usr/bin/env python3
"""
Remove a year from completed status to allow re-processing.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))


def uncomplete_year(year: int, checkpoint_file: str = "2000s_true_batch_checkpoint.json"):
    """Remove a year from completed status."""
    
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return
    
    # Load checkpoint
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    print(f"📋 Current checkpoint status:")
    print(f"   Completed years: {data.get('completed_years', [])}")
    
    if year not in data.get('completed_years', []):
        print(f"\n❌ Year {year} is not in completed years")
        return
    
    # Show year stats
    if 'stats' in data and str(year) in data['stats']:
        stats = data['stats'][str(year)]
        print(f"\n📊 Stats for {year}:")
        print(f"   Articles processed: {stats.get('articles_processed', 0)}")
        print(f"   Summaries stored: {stats.get('summaries_stored', 0)}")
        print(f"   REM nodes: {stats.get('rem_nodes', 0)}")
        print(f"   Batch ID: {stats.get('batch_id', 'N/A')}")
    
    response = input(f"\nRemove {year} from completed years? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Backup checkpoint
    backup_file = f"{checkpoint_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Backed up to: {backup_file}")
    
    # Remove from completed years
    data['completed_years'].remove(year)
    data['last_updated'] = datetime.now().isoformat()
    
    # Save updated checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Removed {year} from completed years")
    print("   The year will be re-processed on next run")
    print(f"   Note: Existing batch ID {stats.get('batch_id', 'N/A')} can be reused")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove a year from completed status")
    parser.add_argument("--year", type=int, default=2000,
                        help="Year to uncomplete (default: 2000)")
    parser.add_argument("--checkpoint", type=str, default="2000s_true_batch_checkpoint.json",
                        help="Checkpoint file path")
    
    args = parser.parse_args()
    
    uncomplete_year(args.year, args.checkpoint)