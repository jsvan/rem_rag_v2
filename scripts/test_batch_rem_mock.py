#!/usr/bin/env python3
"""
Quick test of batch REM with mock data - cheap and fast for debugging.
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient


def create_mock_nodes(store: REMVectorStore, num_nodes: int = 30):
    """Create some mock nodes for testing"""
    print(f"📝 Creating {num_nodes} mock nodes...")
    
    mock_texts = [
        "The Soviet Union's nuclear strategy evolved significantly during the Cold War period.",
        "Khrushchev's approach to diplomacy combined threats with unexpected concessions.",
        "The Cuban Missile Crisis demonstrated the importance of back-channel communications.",
        "NATO's formation fundamentally altered the European security landscape.",
        "Economic interdependence became a key factor in preventing conflicts.",
        "The Berlin Wall symbolized the ideological divide between East and West.",
        "Détente represented a shift from confrontation to negotiation.",
        "Nuclear deterrence theory assumed rational actors on both sides.",
        "The domino theory influenced American foreign policy in Southeast Asia.",
        "Proxy wars became the preferred method of superpower competition.",
        "The Marshall Plan rebuilt Europe while containing Soviet influence.",
        "Brinkmanship became a dangerous tool of Cold War diplomacy.",
        "The Space Race served as peaceful competition between superpowers.",
        "Containment policy shaped American strategy for decades.",
        "The Non-Aligned Movement sought alternatives to bipolar politics.",
        "Arms control treaties attempted to manage nuclear proliferation.",
        "Cultural exchanges softened tensions during periods of détente.",
        "Economic sanctions became tools of coercive diplomacy.",
        "The United Nations struggled to mediate superpower conflicts.",
        "Regional organizations emerged to manage local security issues.",
        "Intelligence agencies played crucial roles in Cold War strategy.",
        "Ideological competition extended to the developing world.",
        "Military alliances created security dilemmas for non-aligned states.",
        "Technology transfer became a tool of influence and control.",
        "Human rights emerged as a factor in international relations.",
        "Energy politics shaped alliances and conflicts.",
        "Trade relationships crossed ideological boundaries.",
        "Nationalism challenged both superpower blocs.",
        "Environmental issues began to require international cooperation.",
        "The information age transformed diplomatic communications."
    ]
    
    # Add nodes with varied years
    node_ids = []
    for i, text in enumerate(mock_texts[:num_nodes]):
        year = 1950 + (i % 40)  # Years from 1950-1989
        metadata = {
            "node_type": "chunk",
            "year": year,
            "article_title": f"Mock Article {i+1}",
            "article_id": f"mock-{i+1}",
            "author": f"Test Author {i % 5}"
        }
        
        ids = store.add([text], [metadata])
        node_ids.extend(ids)
    
    print(f"✅ Created {len(node_ids)} mock nodes")
    return node_ids


def test_batch_creation():
    """Test just the batch file creation part"""
    print("\n🧪 Testing Batch File Creation")
    print("=" * 60)
    
    from src.core.rem_cycle_batch import REMCycleBatch, REMSample, REMBatchItem
    
    # Create mock samples
    samples = [
        REMSample(
            node_id="test-1",
            content="The Soviet Union's nuclear strategy evolved significantly.",
            metadata={"year": 1960, "article_title": "Nuclear Strategy"}
        ),
        REMSample(
            node_id="test-2",
            content="Détente represented a shift from confrontation to negotiation.",
            metadata={"year": 1972, "article_title": "Détente Era"}
        ),
        REMSample(
            node_id="test-3",
            content="The Berlin Wall symbolized the ideological divide.",
            metadata={"year": 1961, "article_title": "Berlin Crisis"}
        )
    ]
    
    # Create batch items
    batch_items = []
    for i in range(3):  # Just 3 dreams for testing
        batch_items.append(REMBatchItem(
            custom_id=f"test_dream_{i}",
            samples=samples,
            current_year=1975
        ))
    
    # Initialize batch REM (we won't actually use the vector store)
    llm = LLMClient()
    store = REMVectorStore()
    batch_rem = REMCycleBatch(llm, store)
    
    # Create batch file
    batch_file_path = batch_rem._create_batch_file(batch_items)
    
    print(f"✅ Created batch file: {batch_file_path}")
    
    # Read and display the batch file
    print("\n📄 Batch file contents (first 5 requests):")
    with open(batch_file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5]):
            req = json.loads(line)
            print(f"\nRequest {i+1}:")
            print(f"  Custom ID: {req['custom_id']}")
            print(f"  Method: {req['method']}")
            print(f"  URL: {req['url']}")
            print(f"  Model: {req['body']['model']}")
            print(f"  Max tokens: {req['body']['max_tokens']}")
            print(f"  Message preview: {req['body']['messages'][0]['content'][:100]}...")
    
    print(f"\n📊 Total requests in batch: {len(lines)}")
    print(f"   (Should be {len(batch_items) * 2} - one question + one synthesis per dream)")
    
    # Clean up
    batch_file_path.unlink()
    print("\n✅ Test completed successfully!")


def test_batch_api_small():
    """Test with a tiny batch to verify OpenAI integration"""
    print("\n🧪 Testing Small Batch with OpenAI API")
    print("=" * 60)
    
    from src.core.rem_cycle_batch import REMCycleBatch
    
    # Initialize
    llm = LLMClient()
    store = REMVectorStore()
    
    # Create a few mock nodes
    create_mock_nodes(store, 10)
    
    # Set up batch REM to process just 1 dream
    batch_rem = REMCycleBatch(llm, store)
    
    # Override the scaling to force just 1 dream
    print("\n🎯 Running batch REM with 1 dream (2 API calls)...")
    
    # Manually prepare just 1 batch item
    batch_items = batch_rem._prepare_batch_items(1, current_year=1975)
    
    if not batch_items:
        print("❌ No batch items prepared")
        return
    
    print(f"✅ Prepared {len(batch_items)} dream(s)")
    
    # Create and submit small batch
    batch_file_path = batch_rem._create_batch_file(batch_items)
    
    print("\n📤 Submitting mini batch to OpenAI...")
    print("   This will cost ~$0.001 (2 requests with GPT-4o-mini)")
    
    response = input("\nProceed with API call? (y/n): ")
    if response.lower() != 'y':
        batch_file_path.unlink()
        print("Cancelled.")
        return
    
    batch_id = batch_rem._submit_batch(batch_file_path)
    
    if not batch_id:
        print("❌ Failed to submit batch")
        return
    
    print(f"✅ Batch submitted: {batch_id}")
    print("⏳ Waiting for completion (this is a tiny batch, should be fast)...")
    
    results = batch_rem._wait_for_batch(batch_id)
    
    if results:
        print(f"\n✅ Batch completed! Got {len(results)} results")
        
        # Show the results
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Custom ID: {result['custom_id']}")
            print(f"  Status: {result['response']['status_code']}")
            if result['response']['status_code'] == 200:
                content = result['response']['body']['choices'][0]['message']['content']
                print(f"  Content preview: {content[:150]}...")
    else:
        print("❌ No results returned")


def main():
    """Run tests"""
    print("🚀 REM Batch Testing Suite")
    print("=" * 70)
    
    # Test 1: Just test batch file creation (no API calls)
    test_batch_creation()
    
    # Test 2: Small batch with real API
    print("\n" + "=" * 70)
    response = input("\nTest with real OpenAI API? This will cost ~$0.001 (y/n): ")
    if response.lower() == 'y':
        test_batch_api_small()
    else:
        print("Skipping API test.")
    
    print("\n✨ All tests completed!")


if __name__ == "__main__":
    main()