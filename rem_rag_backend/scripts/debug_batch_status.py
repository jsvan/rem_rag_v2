#!/usr/bin/env python3
"""
Debug batch status checking to understand why completed batches aren't being detected.
"""

import sys
import openai
from datetime import datetime
import json

# Add the project root to Python path
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


def debug_batch(batch_id: str):
    """Debug the batch status checking."""
    openai.api_key = OPENAI_API_KEY
    
    try:
        batch = openai.batches.retrieve(batch_id)
        
        print(f"\n🔍 Debugging Batch: {batch_id}")
        print("=" * 70)
        
        # Print raw batch object
        print("\n📦 Raw batch object type:", type(batch))
        print("\n📋 Batch attributes:")
        for attr in dir(batch):
            if not attr.startswith('_'):
                try:
                    value = getattr(batch, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <unable to access>")
        
        # Check status specifically
        print(f"\n🎯 Status check:")
        print(f"  batch.status = '{batch.status}'")
        print(f"  batch.status type = {type(batch.status)}")
        print(f"  batch.status == 'completed' = {batch.status == 'completed'}")
        print(f"  batch.status.lower() == 'completed' = {batch.status.lower() == 'completed'}")
        
        # Check if status might have extra whitespace
        print(f"  batch.status repr = {repr(batch.status)}")
        print(f"  batch.status length = {len(batch.status)}")
        
        # Check request counts
        print(f"\n📊 Request counts:")
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            print(f"  counts type: {type(counts)}")
            print(f"  counts attributes: {[attr for attr in dir(counts) if not attr.startswith('_')]}")
            if hasattr(counts, 'total'):
                print(f"  total: {counts.total}")
            if hasattr(counts, 'completed'):
                print(f"  completed: {counts.completed}")
            if hasattr(counts, 'failed'):
                print(f"  failed: {counts.failed}")
        
        # If completed, check output file
        if batch.status.lower() == "completed":
            print(f"\n✅ Batch shows as completed")
            if hasattr(batch, 'output_file_id'):
                print(f"  output_file_id: {batch.output_file_id}")
                
                # Try to download the file
                try:
                    print("\n📥 Attempting to download results...")
                    file_response = openai.files.content(batch.output_file_id)
                    print(f"  File response type: {type(file_response)}")
                    
                    # Check different ways to access content
                    if hasattr(file_response, 'text'):
                        print(f"  Has 'text' attribute")
                        sample = file_response.text[:200] if len(file_response.text) > 200 else file_response.text
                        print(f"  Sample text: {sample}...")
                    elif hasattr(file_response, 'content'):
                        print(f"  Has 'content' attribute")
                        content = file_response.content
                        if isinstance(content, bytes):
                            sample = content.decode('utf-8')[:200]
                        else:
                            sample = str(content)[:200]
                        print(f"  Sample content: {sample}...")
                    else:
                        print(f"  File response attributes: {[attr for attr in dir(file_response) if not attr.startswith('_')]}")
                    
                except Exception as e:
                    print(f"  ❌ Error downloading file: {e}")
                    print(f"  Error type: {type(e)}")
        
        return batch
        
    except Exception as e:
        print(f"❌ Error checking batch: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    if len(sys.argv) > 1:
        batch_id = sys.argv[1]
    else:
        # First list recent batches
        print("\n📋 Recent batches:")
        openai.api_key = OPENAI_API_KEY
        try:
            batches = openai.batches.list(limit=5)
            for i, batch in enumerate(batches.data):
                created = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M')
                print(f"{i+1}. {batch.id} - {batch.status} - {created}")
        except Exception as e:
            print(f"Error listing batches: {e}")
        
        batch_id = input("\nEnter batch ID to debug: ")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    debug_batch(batch_id)


if __name__ == "__main__":
    main()