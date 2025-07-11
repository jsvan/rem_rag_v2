#!/usr/bin/env python3
"""
Test the DatabaseWriterService in isolation.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.core.database_writer import DatabaseWriterService
import openai
from rem_rag_backend.src.config import OPENAI_API_KEY

# Set API key
openai.api_key = OPENAI_API_KEY

async def main():
    # Initialize components
    vector_store = REMVectorStore()
    writer_service = DatabaseWriterService(vector_store)
    
    print("Testing DatabaseWriterService...")
    
    # Start the service
    await writer_service.start()
    
    # Generate some test embeddings
    print("\nGenerating test embeddings...")
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=["Test text 1", "Test text 2", "Test text 3"]
    )
    embeddings = [e.embedding for e in response.data]
    
    # Add some test jobs
    print("\nAdding write jobs...")
    for i in range(3):
        await writer_service.add_write_job('learning', {
            'text': f"Test text {i+1}",
            'embedding': embeddings[i],
            'metadata': {
                'node_type': 'learning',
                'entity': f'TestEntity{i+1}',
                'year': 2024,
                'test': True
            }
        })
    
    # Give it a moment to process
    await asyncio.sleep(2)
    
    # Check stats
    stats = writer_service.get_stats()
    print(f"\nWriter stats: {stats}")
    
    # Stop the service
    await writer_service.stop()
    
    # Verify data was written
    print("\nChecking database...")
    results = vector_store.query(
        text="Test text",
        k=5,
        filter={"test": True}
    )
    
    print(f"Found {len(results['documents'])} test documents in database")
    
    # Clean up test data
    print("\nCleaning up test data...")
    # Note: ChromaDB doesn't have a direct delete by metadata filter,
    # so in production you'd track IDs or use a test collection

if __name__ == "__main__":
    asyncio.run(main())