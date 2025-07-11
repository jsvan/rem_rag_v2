#!/usr/bin/env python3
"""
Test batch completion detection directly.
"""

import sys
import openai
import asyncio
import time
from datetime import datetime

# Add the project root to Python path
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.config import OPENAI_API_KEY


async def test_batch_completion(batch_id: str):
    """Test batch completion detection."""
    openai.api_key = OPENAI_API_KEY
    
    print(f"\n🧪 Testing batch completion detection for: {batch_id}")
    print("=" * 70)
    
    try:
        # Initial check
        batch = openai.batches.retrieve(batch_id)
        print(f"\nInitial status: {batch.status}")
        print(f"Status type: {type(batch.status)}")
        print(f"Status repr: {repr(batch.status)}")
        print(f"Status == 'completed': {batch.status == 'completed'}")
        print(f"Status.lower() == 'completed': {batch.status.lower() == 'completed'}")
        
        if batch.status.lower() == 'completed':
            print("\n✅ Batch is completed! Testing file download...")
            
            try:
                # Test file download
                file_response = openai.files.content(batch.output_file_id)
                print(f"\nFile response type: {type(file_response)}")
                
                # Try different ways to access content
                content = None
                if hasattr(file_response, 'text'):
                    print("✓ Has 'text' attribute")
                    content = file_response.text
                elif hasattr(file_response, 'content'):
                    print("✓ Has 'content' attribute")
                    content = file_response.content
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                elif hasattr(file_response, 'read'):
                    print("✓ Has 'read' method")
                    content = file_response.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                else:
                    print("❌ No known method to access file content")
                    print(f"Available attributes: {[attr for attr in dir(file_response) if not attr.startswith('_')]}")
                
                if content:
                    lines = content.strip().split('\n')
                    print(f"\n📄 File contains {len(lines)} lines")
                    print(f"First line sample: {lines[0][:100]}...")
                    
                    # Test JSON parsing
                    import json
                    try:
                        first_result = json.loads(lines[0])
                        print(f"✓ JSON parsing successful")
                        print(f"First result keys: {list(first_result.keys())}")
                    except Exception as e:
                        print(f"❌ JSON parsing failed: {e}")
                        
            except Exception as e:
                print(f"\n❌ File download error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"\n⏳ Batch not completed yet. Current status: {batch.status}")
            
            # Show progress if available
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                if hasattr(counts, 'total') and counts.total > 0:
                    progress = (counts.completed / counts.total) * 100
                    print(f"Progress: {counts.completed}/{counts.total} ({progress:.1f}%)")
                    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function."""
    if len(sys.argv) > 1:
        batch_id = sys.argv[1]
    else:
        batch_id = input("Enter batch ID to test: ")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    await test_batch_completion(batch_id)


if __name__ == "__main__":
    asyncio.run(main())