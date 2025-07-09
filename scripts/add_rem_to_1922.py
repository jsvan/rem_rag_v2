#!/usr/bin/env python3
"""
Add REM cycles to an existing 1922 experiment.

This script can be run after run_1922_fixed.py to add REM cycle processing
to already stored articles.
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient
from src.core.rem_cycle import REMCycle


def run_rem_on_existing_data():
    """Run REM cycles on already processed 1922 data"""
    print("🌙 Running REM Cycles on Existing 1922 Data")
    print("=" * 50)
    
    # Initialize components
    llm = LLMClient()
    store = REMVectorStore()
    rem_cycle = REMCycle(llm, store)
    
    # Check if we have 1922 data
    stats = store.get_stats()
    print(f"\n📊 Current collection stats:")
    print(f"  Total documents: {stats['total_documents']}")
    
    # Check for 1922 content
    year_1922_results = store.get_by_year(1922)
    num_1922_docs = len(year_1922_results['ids'])
    print(f"  1922 documents: {num_1922_docs}")
    
    if num_1922_docs == 0:
        print("\n❌ No 1922 documents found. Please run run_1922_fixed.py first.")
        return
    
    # Run REM cycles
    print(f"\n🚀 Starting REM cycle generation...")
    print(f"  This will create connections between 1922 articles and any other stored content")
    
    # Run a modest number of dreams
    num_dreams = 25
    
    try:
        rem_node_ids = rem_cycle.run_cycle(
            num_dreams=num_dreams,
            current_year=1922
        )
        
        print(f"\n✅ Successfully created {len(rem_node_ids)} REM insights!")
        
        # Display some examples
        if rem_node_ids:
            print("\n📌 Sample REM Insights Created:")
            
            # Query for the REM nodes we just created
            for i, node_id in enumerate(rem_node_ids[:3]):  # Show first 3
                try:
                    # Get the specific REM node
                    result = store.collection.get(
                        ids=[node_id],
                        include=["documents", "metadatas"]
                    )
                    
                    if result["documents"]:
                        doc = result["documents"][0]
                        metadata = result["metadatas"][0]
                        
                        print(f"\n  Example {i+1}:")
                        
                        # Extract question
                        if "Question:" in doc:
                            question = doc.split("Synthesis:")[0].replace("Question:", "").strip()
                            print(f"  🔍 Question: {question}")
                            # Parse JSON string for years
                            source_years = metadata.get('source_years', '[]')
                            if isinstance(source_years, str):
                                try:
                                    import json
                                    source_years = json.loads(source_years)
                                except:
                                    source_years = []
                            print(f"  📅 Years: {source_years}")
                            
                            # Show synthesis preview
                            if "Synthesis:" in doc:
                                synthesis = doc.split("Synthesis:")[-1].strip()
                                print(f"  💡 Insight: {synthesis[:200]}...")
                                
                except Exception as e:
                    print(f"  ⚠️ Error displaying example {i+1}: {e}")
        
        # Analyze themes discovered
        print("\n\n🎯 Searching REM insights for key 1922 themes:")
        themes = [
            "League of Nations",
            "democracy", 
            "Soviet Russia",
            "international cooperation",
            "post-war order"
        ]
        
        for theme in themes:
            results = rem_cycle.query_rem_insights(theme, top_k=1)
            if results:
                print(f"\n  📌 {theme}:")
                insight = results[0]
                print(f"    Question: {insight['question']}")
                print(f"    Years connected: {insight['source_years']}")
                print(f"    Insight: {insight['synthesis'][:150]}...")
                
    except Exception as e:
        print(f"\n❌ REM cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final stats
    print("\n\n📈 Final Statistics:")
    new_stats = store.get_stats()
    print(f"  Total documents after REM: {new_stats['total_documents']}")
    
    # Count REM nodes
    rem_nodes = store.collection.get(
        where={"node_type": "rem"},
        limit=1000
    )
    print(f"  Total REM insights: {len(rem_nodes['ids'])}")


def main():
    """Run the REM cycle addition"""
    print("\n⚠️  This script adds REM cycles to existing 1922 data.")
    print("Make sure you've already run run_1922_fixed.py!\n")
    
    try:
        run_rem_on_existing_data()
        print("\n✅ REM cycles completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")


if __name__ == "__main__":
    main()