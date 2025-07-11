"""
1922 Experiment: Process the first year of Foreign Affairs articles with READING and REM cycles.

This experiment:
1. Loads all 1922 articles from local JSON files
2. Processes them chronologically through READING cycles
3. Runs quarterly REM cycles (25 dreams each)
4. Evaluates pattern discovery from the founding year of Foreign Affairs
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.data_processing.chunker import SmartChunker
from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src import config


class Year1922Experiment:
    """Run the 1922 experiment to test REM RAG's pattern discovery on Foreign Affairs' founding year"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore()
        self.chunker = SmartChunker()
        self.reading_cycle = ReadingCycle(self.llm, self.structured_llm, self.store, self.chunker)
        self.rem_cycle = REMCycle(self.llm, self.store)
        
    def run(self):
        """Execute the full 1922 experiment"""
        print("🚀 Starting 1922 Experiment - Foreign Affairs Founding Year")
        print("=" * 50)
        
        # Load 1922 articles
        print("\n📚 Loading 1922 articles from local JSON files...")
        articles_1922 = self._load_1922_articles()
        print(f"  Found {len(articles_1922)} articles from 1922")
        
        # Group articles by issue (volume)
        articles_by_issue = self._group_by_issue(articles_1922)
        print(f"  Articles distributed across {len(articles_by_issue)} issues")
        
        # Process each issue
        for issue_key in sorted(articles_by_issue.keys()):
            issue_articles = articles_by_issue[issue_key]
            volume, issue = issue_key
            
            print(f"\n📅 Processing Volume {volume}, Issue {issue} (1922)")
            print(f"  {len(issue_articles)} articles")
            print("-" * 40)
            
            # Print article titles for context
            print("  Articles in this issue:")
            for article in issue_articles:
                print(f"    - {article['title'][:60]}...")
            
            # READING cycle for this issue's articles
            print(f"\n  📖 Running READING cycle...")
            for i, article in enumerate(tqdm(issue_articles, desc=f"    Processing V{volume}I{issue} articles", unit="article")):
                try:
                    # Add article_id if missing
                    if 'article_id' not in article:
                        article['article_id'] = f"1922-v{volume:03d}-i{issue:02d}-{i+1:03d}"
                    
                    # ReadingCycle.process_article is async, so we need to run it
                    import asyncio
                    asyncio.run(self.reading_cycle.process_article(article))
                except Exception as e:
                    print(f"\n    ❌ Error processing article '{article.get('title', 'Unknown')}': {e}")
            
            # Run REM cycle after each issue (quarterly publication)
            print(f"  🌙 Running REM cycle (25 dreams)...")
            rem_nodes = self.rem_cycle.run_cycle(num_dreams=25, current_year=1922)
            print(f"    Generated {len(rem_nodes)} REM insights")
        
        # Analyze results
        print("\n📊 Analyzing Results")
        print("=" * 50)
        self._analyze_results()
        
        # Extract key themes from 1922
        print("\n🔍 Key Themes from Foreign Affairs' Founding Year")
        print("=" * 50)
        self._extract_founding_themes()
    
    def _load_1922_articles(self) -> List[Dict[str, Any]]:
        """Load all articles from 1922 from local JSON files"""
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
        pattern = os.path.join(data_dir, "1922_*.json")
        
        articles_1922 = []
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    
                    # Ensure required fields
                    if 'title' in article and 'content' in article:
                        # Extract metadata from filename if not in JSON
                        filename = os.path.basename(filepath)
                        parts = filename.split('_')
                        
                        if 'volume' not in article and len(parts) > 1:
                            article['volume'] = int(parts[1][1:])  # v001 -> 1
                        if 'issue' not in article and len(parts) > 2:
                            article['issue'] = int(parts[2][1:])   # i01 -> 1
                        
                        articles_1922.append(article)
                    else:
                        print(f"  ⚠️  Skipping {filename}: missing title or content")
            except Exception as e:
                print(f"  ❌ Error loading {filepath}: {e}")
        
        # Sort by volume and issue
        articles_1922.sort(key=lambda x: (x.get('volume', 0), x.get('issue', 0), x.get('title', '')))
        
        return articles_1922
    
    def _group_by_issue(self, articles: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        """Group articles by volume and issue"""
        by_issue = defaultdict(list)
        
        for article in articles:
            volume = article.get('volume', 1)
            issue = article.get('issue', 1)
            by_issue[(volume, issue)].append(article)
        
        return dict(by_issue)
    
    def _analyze_results(self):
        """Analyze the knowledge built from 1922"""
        # Count different node types
        try:
            node_types = ['chunk', 'learning', 'synthesis', 'learning_nothing', 'rem']
            all_nodes = []
            
            for node_type in node_types:
                # Query up to 100 nodes of each type
                nodes = self.store.similarity_search(
                    query="international relations 1922",
                    k=100,
                    where={"node_type": node_type}
                )
                all_nodes.extend(nodes)
        except:
            all_nodes = []
        
        node_counts = defaultdict(int)
        for node in all_nodes:
            node_type = node.metadata.get("node_type", "unknown") if hasattr(node, 'metadata') else "unknown"
            node_counts[node_type] += 1
        
        print("\n📈 Node Statistics:")
        for node_type, count in sorted(node_counts.items()):
            print(f"  - {node_type}: {count} nodes")
        
        # Sample some REM insights
        print("\n💡 Sample REM Insights from 1922:")
        rem_insights = self.rem_cycle.query_rem_insights("post-war international order", top_k=3)
        
        for i, insight in enumerate(rem_insights, 1):
            print(f"\n  Insight {i}:")
            print(f"  Question: {insight['question']}")
            print(f"  Years covered: {', '.join(map(str, insight['source_years']))}")
            print(f"  Synthesis preview: {insight['synthesis'][:200]}...")
    
    def _extract_founding_themes(self):
        """Extract key themes from Foreign Affairs' founding year"""
        # Key themes to explore from 1922
        themes = [
            "Versailles Treaty",
            "League of Nations", 
            "Soviet Russia",
            "reparations Germany",
            "popular diplomacy",
            "mandates system",
            "Little Entente",
            "Irish independence",
            "Pacific policy"
        ]
        
        print("\n🎯 Major themes from 1922 Foreign Affairs:")
        
        for theme in themes:
            insights = self.rem_cycle.query_rem_insights(theme, top_k=2)
            
            if insights:
                print(f"\n  Theme: {theme}")
                print(f"  Found {len(insights)} relevant REM insights")
                
                # Show the most relevant insight
                top_insight = insights[0]
                print(f"  Top insight: {top_insight['question']}")
                print(f"  Preview: {top_insight['synthesis'][:150]}...")
            else:
                # Try a simpler search
                nodes = self.store.similarity_search(
                    query=theme,
                    k=2,
                    where={"year": 1922}
                )
                if nodes:
                    print(f"\n  Theme: {theme}")
                    print(f"  Found {len(nodes)} relevant passages")
                    if hasattr(nodes[0], 'page_content'):
                        print(f"  Preview: {nodes[0].page_content[:150]}...")
        
        # Look for Elihu Root's founding vision
        print("\n\n📰 Elihu Root's Vision (Foreign Affairs founder):")
        root_nodes = self.store.similarity_search(
            query="Elihu Root popular diplomacy democracy foreign policy education",
            k=3,
            where={"year": 1922}
        )
        
        if root_nodes:
            print(f"  Found {len(root_nodes)} passages about Root's vision")
            for i, node in enumerate(root_nodes[:2], 1):
                if hasattr(node, 'page_content'):
                    print(f"\n  Passage {i}: {node.page_content[:200]}...")
                if hasattr(node, 'metadata') and 'article_title' in node.metadata:
                    print(f"  From: {node.metadata['article_title']}")


def main():
    """Run the experiment"""
    experiment = Year1922Experiment()
    
    try:
        experiment.run()
        print("\n✅ Experiment completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()