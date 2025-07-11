#!/usr/bin/env python3
"""
Recover results from a completed batch that the script thinks is still in progress.
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

def recover_batch_results(batch_id: str):
    """Download and save results from a completed batch."""
    
    # Initialize OpenAI client
    openai.api_key = OPENAI_API_KEY
    
    print(f"🔍 Checking batch: {batch_id}")
    
    try:
        # Get batch info
        batch = openai.batches.retrieve(batch_id)
        
        print(f"\n📋 Batch Status: {batch.status}")
        print(f"   Total: {batch.request_counts.total}")
        print(f"   Completed: {batch.request_counts.completed}")
        print(f"   Failed: {batch.request_counts.failed}")
        
        if batch.status == "completed" and batch.output_file_id:
            print(f"\n✅ Batch is completed! Downloading results...")
            print(f"   Output file ID: {batch.output_file_id}")
            
            # Download the results
            file_response = openai.files.content(batch.output_file_id)
            
            # Handle different response formats
            if hasattr(file_response, 'text'):
                content = file_response.text
            elif hasattr(file_response, 'content'):
                content = file_response.content
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                content = file_response.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            
            # Save to file
            output_file = f"recovered_batch_{batch_id}.jsonl"
            with open(output_file, 'w') as f:
                f.write(content)
            
            # Count results
            results = []
            for line in content.strip().split('\n'):
                if line:
                    results.append(json.loads(line))
            
            print(f"\n💾 Saved {len(results)} results to {output_file}")
            
            # Show sample result
            if results:
                sample = results[0]
                print(f"\n📄 Sample result:")
                print(f"   ID: {sample.get('custom_id', 'N/A')}")
                if 'response' in sample and sample['response']:
                    if 'body' in sample['response']:
                        choices = sample['response']['body'].get('choices', [])
                        if choices and 'message' in choices[0]:
                            content = choices[0]['message'].get('content', '')
                            print(f"   Content preview: {content[:100]}...")
            
            return results
            
        elif batch.status == "failed":
            print(f"\n❌ Batch failed!")
            if batch.error_file_id:
                print("   Downloading error file...")
                error_response = openai.files.content(batch.error_file_id)
                print(f"   Errors: {error_response.text}")
        else:
            print(f"\n⏳ Batch is still: {batch.status}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check the completed batch from checkpoint
        checkpoint_file = "2000s_true_batch_checkpoint.json"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                if '2000' in checkpoint.get('stats', {}):
                    batch_id = checkpoint['stats']['2000'].get('batch_id')
                    if batch_id:
                        print(f"Using batch ID from checkpoint: {batch_id}")
                        recover_batch_results(batch_id)
                    else:
                        print("No batch ID found in checkpoint")
                else:
                    print("No 2000 stats in checkpoint")
        else:
            print(f"Usage: {sys.argv[0]} <batch_id>")
            print("Or ensure 2000s_true_batch_checkpoint.json exists")
    else:
        batch_id = sys.argv[1]
        recover_batch_results(batch_id)