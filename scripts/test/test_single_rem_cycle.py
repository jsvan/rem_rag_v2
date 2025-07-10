#!/usr/bin/env python3
"""
Test a single REM cycle, showing the sampled chunks and resulting synthesis.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rem_cycle import REMCycle
from src.vector_store.chromadb_store import REMVectorStore
from src.llm.openai_client import LLMClient
from src.config import LLM_MODEL


def test_rem_cycle():
    """Run a single REM cycle and display the process."""
    print("🌙 Testing Single REM Cycle")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    rem_cycle = REMCycle(llm_client, vector_store)
    
    # Check database status
    print("\n📊 Database Status:")
    print("-" * 40)
    
    # Get counts by node type
    all_nodes = vector_store.sample(1000)  # Sample many to get counts
    node_types = {}
    for metadata in all_nodes['metadatas']:
        node_type = metadata.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"Total nodes available for sampling: {len(all_nodes['documents'])}")
    print("\nNode type distribution:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type:15}: {count:4}")
    
    # Sample nodes for REM
    print("\n\n🎲 Sampling Nodes for REM...")
    print("=" * 80)
    
    # Get samples
    samples = rem_cycle._sample_nodes(current_year=None)
    
    if not samples or len(samples) < 3:
        print("\n❌ Error: Not enough nodes to sample")
        return
    
    # Display the sampled nodes
    print("\n📚 Sampled Nodes:")
    print("-" * 80)
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*20} NODE {i+1} {'='*20}")
        print(f"Type: {sample.metadata.get('node_type', 'unknown')}")
        print(f"Year: {sample.metadata.get('year', 'unknown')}")
        if sample.metadata.get('title'):
            print(f"Article: {sample.metadata['title']}")
        if sample.metadata.get('entity'):
            print(f"Entity: {sample.metadata['entity']}")
        print("-" * 60)
        
        # Truncate long texts for display
        if len(sample.content) > 500:
            print(sample.content[:500] + "...")
        else:
            print(sample.content)
    
    # Generate the implicit question
    print("\n\n❓ Finding Implicit Question...")
    print("-" * 80)
    question = rem_cycle._find_implicit_question(samples)
    print(question)
    
    # Generate synthesis
    print("\n\n💡 Generating REM Synthesis...")
    print("-" * 80)
    synthesis = rem_cycle._generate_synthesis(samples, question)
    print(f"Question: {question}\n")
    print(f"Synthesis: {synthesis}")
    
    # Create and store the REM node
    rem_metadata = {
        "node_type": "rem",
        "synthesis_type": "rem_dream",
        "year": 2000,  # Using 2000 as test year
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_node_ids": ",".join([s.node_id for s in samples]),  # Convert list to string
        "implicit_question": question
    }
    
    rem_text = f"Question: {question}\n\nSynthesis: {synthesis}"
    
    # Use implant to store and potentially generate meta-synthesis
    print("\n\n🔗 Implanting REM Insight...")
    print("-" * 80)
    
    from src.core.implant import implant_knowledge_sync
    
    implant_result = implant_knowledge_sync(
        new_content=rem_text,
        vector_store=vector_store,
        llm_client=llm_client,
        metadata=rem_metadata,
        k=5
    )
    
    print(f"REM node stored: {implant_result['original_id']}")
    print(f"Related nodes found: {implant_result['existing_count']}")
    
    if implant_result['is_valuable'] and implant_result['synthesis']:
        print(f"\n✨ Meta-synthesis generated (comparing to existing REM insights):")
        print(implant_result['synthesis'])
    
    print("\n\n✅ REM Cycle Complete!")
    
    # Run another cycle to see variety
    print("\n\n" + "="*80)
    print("🎲 Running Another REM Cycle for Comparison...")
    print("="*80)
    
    samples2 = rem_cycle._sample_nodes(current_year=None)
    if samples2 and len(samples2) >= 3:
        question2 = rem_cycle._find_implicit_question(samples2)
        synthesis2 = rem_cycle._generate_synthesis(samples2, question2)
        
        print("\n❓ Generated Question:")
        print(question2)
        
        print("\n💡 REM Synthesis:")
        print(synthesis2)
        
        # Show node types sampled
        print("\n📊 Sampled node types:", 
              [s.metadata.get('node_type', 'unknown') for s in samples2])


if __name__ == "__main__":
    test_rem_cycle()