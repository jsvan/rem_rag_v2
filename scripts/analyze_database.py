#!/usr/bin/env python3
"""
Analyze the REM RAG database by node type and year.

This script provides comprehensive analysis of the stored knowledge:
- Node counts by type
- Year distribution
- Sample content from each type
- Entity/theme frequency analysis
- REM insight patterns
"""

import os
import sys
import json
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore


class DatabaseAnalyzer:
    """Analyze the REM RAG database contents"""
    
    def __init__(self):
        self.store = REMVectorStore()
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def run(self):
        """Execute comprehensive analysis"""
        print("🔍 REM RAG Database Analysis")
        print("=" * 70)
        
        # Get all nodes
        all_nodes = self.store.collection.get(limit=100000)
        total_count = len(all_nodes['ids'])
        
        if total_count == 0:
            print("❌ Database is empty!")
            return
            
        print(f"\n📊 Total nodes in database: {total_count:,}")
        
        # Analyze by node type
        self.analyze_node_types(all_nodes)
        
        # Analyze by year
        self.analyze_year_distribution(all_nodes)
        
        # Analyze entities/themes
        self.analyze_entities(all_nodes)
        
        # Analyze REM insights
        self.analyze_rem_insights(all_nodes)
        
        # Sample content
        self.show_sample_content(all_nodes)
        
        # Synthesis patterns
        self.analyze_synthesis_patterns(all_nodes)
    
    def analyze_node_types(self, all_nodes: Dict[str, Any]):
        """Analyze distribution by node type"""
        print("\n\n📋 Node Type Distribution:")
        print("-" * 40)
        
        type_counts = Counter()
        type_years = defaultdict(set)
        
        for i, metadata in enumerate(all_nodes['metadatas']):
            node_type = metadata.get('node_type', 'unknown')
            type_counts[node_type] += 1
            
            year = metadata.get('year')
            if year:
                type_years[node_type].add(year)
        
        # Sort by count
        for node_type, count in type_counts.most_common():
            years = sorted(type_years[node_type])
            year_range = f" ({min(years)}-{max(years)})" if years else ""
            print(f"  • {node_type:12s}: {count:6,} nodes{year_range}")
    
    def analyze_year_distribution(self, all_nodes: Dict[str, Any]):
        """Analyze distribution by year"""
        print("\n\n📅 Year Distribution:")
        print("-" * 40)
        
        year_counts = defaultdict(lambda: defaultdict(int))
        
        for metadata in all_nodes['metadatas']:
            year = metadata.get('year')
            node_type = metadata.get('node_type', 'unknown')
            if year:
                year_counts[year][node_type] += 1
        
        # Sort by year
        for year in sorted(year_counts.keys()):
            total = sum(year_counts[year].values())
            print(f"\n  {year}: {total:,} total nodes")
            
            # Show breakdown by type
            for node_type, count in sorted(year_counts[year].items()):
                print(f"    - {node_type}: {count:,}")
    
    def analyze_entities(self, all_nodes: Dict[str, Any]):
        """Analyze entity/theme distribution"""
        print("\n\n🏷️  Top Entities/Themes:")
        print("-" * 40)
        
        entity_counts = Counter()
        entity_years = defaultdict(set)
        
        for metadata in all_nodes['metadatas']:
            if metadata.get('node_type') == 'learning':
                entity = metadata.get('entity')
                if entity:
                    entity_counts[entity] += 1
                    year = metadata.get('year')
                    if year:
                        entity_years[entity].add(year)
        
        if not entity_counts:
            print("  No entity learnings found")
            return
        
        # Show top 20 entities
        print(f"\n  Total unique entities: {len(entity_counts)}")
        print("\n  Top 20 entities by frequency:")
        
        for entity, count in entity_counts.most_common(20):
            years = sorted(entity_years[entity])
            year_span = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
            print(f"    • {entity:30s}: {count:3} mentions ({year_span})")
    
    def analyze_rem_insights(self, all_nodes: Dict[str, Any]):
        """Analyze REM insights"""
        print("\n\n🌙 REM Insights Analysis:")
        print("-" * 40)
        
        rem_nodes = []
        rem_years = defaultdict(int)
        
        for i, metadata in enumerate(all_nodes['metadatas']):
            if metadata.get('node_type') == 'rem':
                rem_nodes.append({
                    'content': all_nodes['documents'][i],
                    'metadata': metadata
                })
                year = metadata.get('year')
                if year:
                    rem_years[year] += 1
        
        if not rem_nodes:
            print("  No REM insights found")
            return
        
        print(f"\n  Total REM insights: {len(rem_nodes)}")
        print("\n  REM insights by year:")
        for year in sorted(rem_years.keys()):
            print(f"    • {year}: {rem_years[year]} insights")
        
        # Sample a few REM questions
        print("\n  Sample REM questions:")
        sample_size = min(5, len(rem_nodes))
        for i, node in enumerate(rem_nodes[:sample_size]):
            question = node['metadata'].get('implicit_question', 'No question')
            year = node['metadata'].get('year', 'Unknown')
            print(f"\n  [{i+1}] Year {year}:")
            print(f"      Q: {question}")
    
    def show_sample_content(self, all_nodes: Dict[str, Any]):
        """Show sample content from each node type"""
        print("\n\n📝 Sample Content by Type:")
        print("-" * 40)
        
        # Group by node type
        by_type = defaultdict(list)
        for i, metadata in enumerate(all_nodes['metadatas']):
            node_type = metadata.get('node_type', 'unknown')
            by_type[node_type].append({
                'content': all_nodes['documents'][i],
                'metadata': metadata
            })
        
        # Show samples
        for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
            if node_type in by_type:
                print(f"\n  {node_type.upper()} Sample:")
                sample = by_type[node_type][0]
                
                # Show metadata
                year = sample['metadata'].get('year', 'Unknown')
                article = sample['metadata'].get('article_title', 'Unknown')[:50]
                
                print(f"    Year: {year}")
                print(f"    Article: {article}...")
                print(f"    Content: {sample['content'][:200]}...")
    
    def analyze_synthesis_patterns(self, all_nodes: Dict[str, Any]):
        """Analyze synthesis generation patterns"""
        print("\n\n🔗 Synthesis Patterns:")
        print("-" * 40)
        
        synthesis_count = 0
        synthesis_years = set()
        generation_depths = Counter()
        
        for metadata in all_nodes['metadatas']:
            if metadata.get('node_type') == 'synthesis':
                synthesis_count += 1
                year = metadata.get('year')
                if year:
                    synthesis_years.add(year)
                depth = metadata.get('generation_depth', 0)
                generation_depths[depth] += 1
        
        if synthesis_count == 0:
            print("  No synthesis nodes found")
            return
        
        print(f"\n  Total synthesis nodes: {synthesis_count:,}")
        print(f"  Years with synthesis: {min(synthesis_years)}-{max(synthesis_years)}")
        
        print("\n  Generation depth distribution:")
        for depth in sorted(generation_depths.keys()):
            print(f"    • Depth {depth}: {generation_depths[depth]:,} nodes")
        
        # Calculate synthesis rate
        total_chunks = sum(1 for m in all_nodes['metadatas'] if m.get('node_type') == 'chunk')
        total_summaries = sum(1 for m in all_nodes['metadatas'] if m.get('node_type') == 'summary')
        
        if total_chunks + total_summaries > 0:
            synthesis_rate = synthesis_count / (total_chunks + total_summaries) * 100
            print(f"\n  Synthesis rate: {synthesis_rate:.1f}% of chunks+summaries generated synthesis")


def main():
    """Run the database analyzer"""
    analyzer = DatabaseAnalyzer()
    
    try:
        analyzer.run()
        print("\n\n✅ Analysis complete!")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()