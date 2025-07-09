"""
1962 Experiment: Process a year of Foreign Affairs articles with READING and REM cycles.

This experiment:
1. Loads all 1962 articles from the HuggingFace dataset
2. Processes them chronologically through READING cycles
3. Runs monthly REM cycles (100 dreams each)
4. Evaluates: Can the system predict themes from year-end retrospectives?
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.data_processing.fa_loader import ForeignAffairsLoader
from src.data_processing.chunker import SmartChunker
from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src import config


class Year1962Experiment:
    """Run the 1962 experiment to test REM RAG's pattern discovery"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.store = REMVectorStore()
        self.loader = ForeignAffairsLoader()
        self.chunker = SmartChunker()
        self.reading_cycle = ReadingCycle(self.llm, self.structured_llm, self.store, self.chunker)
        self.rem_cycle = REMCycle(self.llm, self.store)
        
    def run(self):
        """Execute the full 1962 experiment"""
        print("🚀 Starting 1962 Experiment")
        print("=" * 50)
        
        # Load 1962 articles
        print("\n📚 Loading 1962 articles from Foreign Affairs...")
        articles_1962 = self._load_1962_articles()
        print(f"  Found {len(articles_1962)} articles from 1962")
        
        # Group articles by month
        articles_by_month = self._group_by_month(articles_1962)
        print(f"  Articles distributed across {len(articles_by_month)} months")
        
        # Process each month
        for month_num in sorted(articles_by_month.keys()):
            month_articles = articles_by_month[month_num]
            month_name = datetime(1962, month_num, 1).strftime("%B")
            
            print(f"\n📅 Processing {month_name} 1962 ({len(month_articles)} articles)")
            print("-" * 40)
            
            # READING cycle for this month's articles
            print(f"  📖 Running READING cycle...")
            for i, article in enumerate(tqdm(month_articles, desc=f"    Processing {month_name} articles", unit="article")):
                try:
                    # Add article_id if missing
                    if 'article_id' not in article:
                        article['article_id'] = f"1962-{month_num:02d}-{i+1:03d}"
                    
                    # ReadingCycle.process_article is async, so we need to run it
                    import asyncio
                    asyncio.run(self.reading_cycle.process_article(article))
                except Exception as e:
                    print(f"\n    ❌ Error processing article '{article.get('title', 'Unknown')}': {e}")
            
            # REM cycle at end of month
            print(f"  🌙 Running REM cycle (100 dreams)...")
            rem_nodes = self.rem_cycle.run_cycle(num_dreams=100, current_year=1962)
            print(f"    Generated {len(rem_nodes)} REM insights")
        
        # Analyze results
        print("\n📊 Analyzing Results")
        print("=" * 50)
        self._analyze_results()
        
        # Find retrospective predictions
        print("\n🔮 Checking Retrospective Predictions")
        print("=" * 50)
        self._check_retrospective_predictions()
    
    def _load_1962_articles(self) -> List[Dict[str, Any]]:
        """Load all articles from 1962"""
        all_data = self.loader.load_dataset()
        
        # Debug: Print first few items to see structure
        print("\n📋 Dataset structure:")
        print(f"  Type of all_data: {type(all_data)}")
        print(f"  Number of items: {len(all_data) if hasattr(all_data, '__len__') else 'Unknown'}")
        
        # Convert to list if it's a DataFrame
        if hasattr(all_data, 'to_dict'):
            print("  Converting DataFrame to list of dicts...")
            all_data = all_data.to_dict('records')
        
        # Print first article to see structure
        if all_data and len(all_data) > 0:
            print(f"\n  First article structure:")
            first_article = all_data[0]
            print(f"  Type: {type(first_article)}")
            if isinstance(first_article, dict):
                print(f"  Keys: {list(first_article.keys())[:10]}...")  # First 10 keys
                print(f"\n  Sample article:")
                for key, value in list(first_article.items())[:5]:
                    if isinstance(value, str):
                        print(f"    {key}: {value[:100]}..." if len(value) > 100 else f"    {key}: {value}")
                    else:
                        print(f"    {key}: {value}")
        
        articles_1962 = []
        for item in all_data:
            # Extract year from metadata
            if isinstance(item, dict):
                year = item.get("year")
                
                # If year is None, try to extract from URL
                if year is None and 'url' in item:
                    # URL format: www.foreignaffairs.com/articles/YYYY-MM-DD/...
                    import re
                    match = re.search(r'/(\d{4})-\d{2}-\d{2}/', item['url'])
                    if match:
                        year = int(match.group(1))
                
                if year == 1962:
                    # Add year to the item for later use
                    item['year'] = year
                    articles_1962.append(item)
            else:
                print(f"  ⚠️  Unexpected item type: {type(item)}")
        
        # Sort by date
        articles_1962.sort(key=lambda x: x.get("date", "1962-01-01"))
        
        return articles_1962
    
    def _group_by_month(self, articles: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group articles by month number"""
        by_month = defaultdict(list)
        
        for article in articles:
            # Parse date to get month
            date_str = article.get("date")
            month = 1  # Default to January
            
            # Try to extract date from URL if date field is missing
            if not date_str and 'url' in article:
                import re
                match = re.search(r'/\d{4}-(\d{2})-\d{2}/', article['url'])
                if match:
                    month = int(match.group(1))
            else:
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    month = date.month
                except:
                    month = 1
            
            by_month[month].append(article)
        
        return dict(by_month)
    
    def _analyze_results(self):
        """Analyze the knowledge built over the year"""
        # Count different node types
        # Note: REMVectorStore doesn't have get_all_nodes, so we'll query instead
        try:
            # Query for all node types we know about
            node_types = ['chunk', 'learning', 'synthesis', 'learning_nothing', 'rem']
            all_nodes = []
            
            for node_type in node_types:
                # Query up to 100 nodes of each type
                nodes = self.store.similarity_search(
                    query="*",  # Broad query
                    k=100,
                    where={"node_type": node_type}
                )
                all_nodes.extend(nodes)
        except:
            # If querying fails, just continue with empty results
            all_nodes = []
        
        node_counts = defaultdict(int)
        for node in all_nodes:
            # nodes from similarity_search have metadata as an attribute
            node_type = node.metadata.get("node_type", "unknown") if hasattr(node, 'metadata') else "unknown"
            node_counts[node_type] += 1
        
        print("\n📈 Node Statistics:")
        for node_type, count in sorted(node_counts.items()):
            print(f"  - {node_type}: {count} nodes")
        
        # Sample some REM insights
        print("\n💡 Sample REM Insights:")
        rem_insights = self.rem_cycle.query_rem_insights("Cold War patterns", top_k=3)
        
        for i, insight in enumerate(rem_insights, 1):
            print(f"\n  Insight {i}:")
            print(f"  Question: {insight['question']}")
            print(f"  Years covered: {', '.join(map(str, insight['source_years']))}")
            print(f"  Synthesis preview: {insight['synthesis'][:200]}...")
    
    def _check_retrospective_predictions(self):
        """Check if REM insights align with year-end retrospectives"""
        # Query for major themes that might appear in retrospectives
        themes = [
            "Cuban Missile Crisis",
            "Sino-Soviet relations", 
            "European integration",
            "decolonization Africa",
            "nuclear policy"
        ]
        
        print("\n🎯 Checking alignment with likely retrospective themes:")
        
        for theme in themes:
            insights = self.rem_cycle.query_rem_insights(theme, top_k=2)
            
            if insights:
                print(f"\n  Theme: {theme}")
                print(f"  Found {len(insights)} relevant REM insights")
                
                # Show the most relevant insight
                top_insight = insights[0]
                print(f"  Top insight question: {top_insight['question']}")
                print(f"  Covers years: {', '.join(map(str, top_insight['source_years']))}")
            else:
                print(f"\n  Theme: {theme} - No REM insights found")
        
        # Look for December articles that might be retrospectives
        print("\n\n📰 Searching for actual retrospective articles...")
        december_nodes = self.store.similarity_search(
            query="year in review retrospective 1962 summary",
            k=5,
            where={
                "node_type": "chunk",
                "year": 1962,
                "$and": [
                    {"article_title": {"$contains": "196"}},
                    {"$or": [
                        {"article_title": {"$contains": "Review"}},
                        {"article_title": {"$contains": "Retrospect"}},
                        {"article_title": {"$contains": "Year"}}
                    ]}
                ]
            }
        )
        
        if december_nodes:
            print(f"  Found {len(december_nodes)} potential retrospective articles")
            for node in december_nodes[:2]:
                print(f"    - {node.metadata.get('article_title', 'Unknown title')}")
        else:
            print("  No clear retrospective articles found in dataset")


def main():
    """Run the experiment"""
    experiment = Year1962Experiment()
    
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