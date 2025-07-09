#!/usr/bin/env python3
"""
Test script to visualize a single REM cycle step with implant process.

This shows:
1. The 3 sampled nodes
2. The implicit question discovered
3. The initial synthesis
4. The implant step (finding similar knowledge and creating final synthesis)
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient
from src.core.rem_cycle import REMCycle, REMSample
from src.core.implant import implant_knowledge_sync


class REMStepVisualizer:
    """Visualize a single REM cycle step in detail"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.store = REMVectorStore()
        self.rem_cycle = REMCycle(self.llm, self.store)
        
    def run_single_dream(self, current_year: int = 1922):
        """Run and visualize a single REM dream"""
        print("🌙 REM Cycle Step-by-Step Visualization")
        print("=" * 80)
        
        # Step 1: Sample 3 nodes
        print("\n=== STEP 1: SAMPLING 3 NODES ===")
        samples = self.rem_cycle._sample_nodes(current_year)
        
        if not samples or len(samples) < 3:
            print("❌ Insufficient nodes in database for REM cycle")
            return
            
        for i, sample in enumerate(samples):
            print(f"\n📄 Node {i+1}:")
            print(f"   Type: {sample.metadata.get('node_type', 'unknown')}")
            print(f"   Year: {sample.metadata.get('year', 'unknown')}")
            if 'article_title' in sample.metadata:
                print(f"   Article: {sample.metadata['article_title']}")
            if 'entity' in sample.metadata:
                print(f"   Entity: {sample.metadata['entity']}")
            print(f"   Content length: {len(sample.content)} characters, {len(sample.content.split())} words")
            if 'chunker' in sample.metadata:
                print(f"   Chunker: {sample.metadata['chunker']}")
            if 'word_count' in sample.metadata:
                print(f"   Original word count: {sample.metadata['word_count']}")
            print(f"   Content:")
            print("-" * 40)
            print(sample.content)
            print("-" * 40)
        
        # Step 2: Find implicit question
        print("\n\n=== STEP 2: DISCOVERING IMPLICIT QUESTION ===")
        question = self.rem_cycle._find_implicit_question(samples)
        print(f"🔍 Question: {question}")
        
        # Step 3: Generate initial synthesis
        print("\n\n=== STEP 3: GENERATING INITIAL SYNTHESIS ===")
        synthesis = self.rem_cycle._generate_synthesis(samples, question)
        print(f"💡 Synthesis:")
        print(f"{synthesis}")
        
        # Step 4: Store REM node
        print("\n\n=== STEP 4: STORING REM NODE ===")
        metadata = {
            "node_type": "rem",
            "implicit_question": question,
            "source_node_ids": json.dumps([s.node_id for s in samples]),
            "source_years": json.dumps([s.metadata.get("year", "Unknown") for s in samples]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        years = [s.metadata.get("year") for s in samples if s.metadata.get("year")]
        if years:
            metadata["year_min"] = min(years)
            metadata["year_max"] = max(years)
        
        text_for_embedding = f"Question: {question}\n\nSynthesis: {synthesis}"
        
        node_ids = self.store.add(
            texts=[text_for_embedding],
            metadata=[metadata]
        )
        node_id = node_ids[0]
        print(f"✅ Stored REM node with ID: {node_id}")
        
        # Step 5: Implant step - find similar knowledge
        print("\n\n=== STEP 5: IMPLANT STEP (Finding Similar Knowledge) ===")
        
        # Query for similar content (using 3 neighbors as suggested)
        similar_results = self.store.query(
            text=text_for_embedding,
            k=3,
            filter=None  # Look across all knowledge
        )
        
        print(f"\n🔗 Found {len(similar_results['documents'])} similar nodes that will be synthesized together:")
        similar_count = 0
        for i, (doc, metadata) in enumerate(zip(similar_results['documents'], similar_results['metadatas'])):
            if doc != text_for_embedding:  # Skip self
                similar_count += 1
                print(f"\n   Similar Node {similar_count}:")
                print(f"   - Type: {metadata.get('node_type', 'unknown')}")
                print(f"   - Year: {metadata.get('year', 'unknown')}")
                if metadata.get('implicit_question'):
                    print(f"   - Question: {metadata['implicit_question']}")
                print(f"   - Length: {len(doc)} characters")
                print(f"   - Full content:")
                print("   " + "-" * 40)
                print(f"   {doc}")
                print("   " + "-" * 40)
        
        # Step 6: Generate implant synthesis
        print("\n\n=== STEP 6: GENERATING IMPLANT SYNTHESIS ===")
        print("🤔 How does this REM insight relate to what we already know?")
        print(f"📊 Creating ONE synthesis that compares the new REM insight with ALL {similar_count} similar nodes combined")
        
        implant_result = implant_knowledge_sync(
            new_content=text_for_embedding,
            vector_store=self.store,
            llm_client=self.llm,
            metadata={
                "node_type": "synthesis",
                "source_type": "rem_synthesis",
                "rem_node_id": node_id,
                "implicit_question": question,
                "source_years": json.dumps([s.metadata.get("year", "Unknown") for s in samples]),
                "generation_depth": 1,
                "synthesis_type": "rem_level",
                "timestamp": datetime.utcnow().isoformat()
            },
            context_filter=None,
            k=3  # Reduced from 5 to 3 as suggested
        )
        
        if implant_result['is_valuable']:
            print(f"\n✨ Valuable synthesis discovered!")
            print(f"📏 Synthesis length: {len(implant_result['synthesis'])} characters")
            print(f"\n💭 Full Synthesis:")
            print("-" * 60)
            print(implant_result['synthesis'])
            print("-" * 60)
            print(f"\n📝 Stored with ID: {implant_result['synthesis_id']}")
        else:
            print(f"\n📌 No new insights - synthesis was: {implant_result['synthesis']}")
        
        # Summary
        print("\n\n=== SUMMARY ===")
        print(f"✅ REM Node stored: {node_id}")
        print(f"✅ Synthesis valuable: {implant_result['is_valuable']}")
        print(f"📊 Total nodes in database: {self.store.get_stats()['total_documents']}")
        
        # Analysis of lengths
        print("\n\n📏 Length Analysis:")
        print(f"   Sampled chunk lengths: {[len(s.content) for s in samples]}")
        print(f"   REM synthesis length: {len(synthesis)} characters")
        if implant_result['is_valuable']:
            print(f"   Implant synthesis length: {len(implant_result['synthesis'])} characters")
        
    def show_database_stats(self):
        """Show current database statistics"""
        stats = self.store.get_stats()
        print("\n📊 Database Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Years covered: {stats['years']}")
        
        # Count by node type
        node_types = {}
        try:
            all_docs = self.store.collection.get(limit=10000, include=["metadatas"])
            for metadata in all_docs["metadatas"]:
                node_type = metadata.get('node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"\n   Nodes by type:")
            for node_type, count in sorted(node_types.items()):
                print(f"   - {node_type}: {count}")
                
        except Exception as e:
            print(f"   Error counting node types: {e}")


def main():
    """Run the REM step visualization"""
    visualizer = REMStepVisualizer()
    
    print("\n⚠️  This script visualizes a single REM cycle step")
    print("It shows the complete process including the implant step\n")
    
    # Show current database state
    visualizer.show_database_stats()
    
    try:
        # Run single dream
        visualizer.run_single_dream(current_year=1922)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()