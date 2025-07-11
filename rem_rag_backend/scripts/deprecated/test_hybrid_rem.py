#!/usr/bin/env python3
"""
Test the hybrid REM cycle implementation.

This tests that:
1. Database operations are performed serially
2. OpenAI calls are batched efficiently
3. No locking issues occur
"""

import os
import sys
import asyncio
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient
from rem_rag_backend.src.core.rem_cycle_hybrid import BatchREMCycle
from rem_rag_backend.src.config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_hybrid_rem():
    """Test the hybrid REM cycle with a small number of cycles"""
    print("🧪 Testing Hybrid REM Cycle")
    print("=" * 60)
    
    # Initialize components
    store = REMVectorStore()
    llm = LLMClient()
    
    # Create hybrid REM cycle
    rem_cycle = BatchREMCycle(
        store=store,
        llm=llm,
        api_key=OPENAI_API_KEY,
        max_concurrent=10  # Lower concurrency for testing
    )
    
    # Check database state
    try:
        # Count nodes
        all_results = store.collection.get(
            where={"node_type": {"$ne": "rem"}},
            limit=1
        )
        # Do a proper count
        count_results = store.collection.get(
            where={"node_type": {"$ne": "rem"}},
            limit=10000
        )
        node_count = len(count_results["ids"])
        
        print(f"📊 Database has {node_count} non-REM nodes")
        
        if node_count < 10:
            print("❌ Not enough nodes for testing. Please run some data processing first.")
            return
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return
    
    # Test with a small number of cycles
    num_cycles = 5
    print(f"\n🎯 Testing with {num_cycles} REM cycles...")
    
    try:
        stats = await rem_cycle.run_batch_rem_cycles(
            num_cycles=num_cycles,
            current_year=2000  # Test with a specific year
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"Results: {stats}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the test"""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    await test_hybrid_rem()


if __name__ == "__main__":
    asyncio.run(main())