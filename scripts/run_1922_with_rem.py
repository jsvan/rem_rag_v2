#!/usr/bin/env python3
"""
Enhanced 1922 Foreign Affairs experiment with integrated REM cycles.

This script processes 1922 articles in quarterly batches and runs REM cycles
to discover patterns across the content.
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.rem_cycle import REMCycle


class Enhanced1922Experiment:
    """Enhanced 1922 experiment with REM cycle integration"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        self.rem_cycle = REMCycle(self.llm, self.store)
        
        # Track articles by quarter for batch processing
        self.articles_by_quarter = defaultdict(list)
        
    def run(self):
        """Execute the enhanced experiment with REM cycles"""
        print("🚀 Starting Enhanced 1922 Experiment with REM Cycles")
        print("=" * 50)
        
        # Load and organize articles by quarter
        articles = self.load_1922_articles()
        print(f"\n📚 Loaded {len(articles)} articles from 1922")
        
        # Organize articles by quarter (based on issue number)
        for article in articles:
            quarter = self.get_quarter(article)
            self.articles_by_quarter[quarter].append(article)
        
        print("\n📊 Articles by quarter:")
        for quarter in sorted(self.articles_by_quarter.keys()):
            print(f"  Q{quarter}: {len(self.articles_by_quarter[quarter])} articles")
        
        # Process each quarter and run REM cycles
        for quarter in sorted(self.articles_by_quarter.keys()):
            print(f"\n\n{'='*60}")
            print(f"📖 Processing Q{quarter} 1922")
            print(f"{'='*60}")
            
            # Process articles in this quarter
            quarter_articles = self.articles_by_quarter[quarter]
            for i, article in enumerate(quarter_articles):
                print(f"\n[{i+1}/{len(quarter_articles)}] {article['title'][:60]}...")
                
                try:
                    self.process_article(article)
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            
            # Run REM cycle after each quarter
            print(f"\n\n🌙 Running REM Cycle for Q{quarter} 1922")
            print("-" * 40)
            self.run_quarterly_rem_cycle(quarter)
        
        # Final analysis
        print("\n\n📊 Final Analysis")
        print("=" * 50)
        self.analyze_results()
        self.analyze_rem_insights()
    
    def get_quarter(self, article: Dict[str, Any]) -> int:
        """Determine quarter based on issue number (1-4 assumed)"""
        issue = article.get('issue', 1)
        # Map issues to quarters (assuming quarterly publication)
        # Adjust this mapping based on actual publication schedule
        if issue <= 1:
            return 1
        elif issue <= 2:
            return 2
        elif issue <= 3:
            return 3
        else:
            return 4
    
    def run_quarterly_rem_cycle(self, quarter: int):
        """Run REM cycle after processing a quarter's articles"""
        # Run a smaller number of dreams for quarterly cycles
        num_dreams = 10  # Reduced for quarterly processing
        
        try:
            rem_node_ids = self.rem_cycle.run_cycle(
                num_dreams=num_dreams,
                current_year=1922
            )
            
            print(f"\n✨ Created {len(rem_node_ids)} REM insights for Q{quarter}")
            
            # Sample and display one REM insight as example
            if rem_node_ids:
                self.display_sample_rem_insight()
                
        except Exception as e:
            print(f"❌ REM cycle failed: {e}")
    
    def display_sample_rem_insight(self):
        """Display a sample REM insight that was just created"""
        try:
            # Query for the most recent REM node
            rem_results = self.store.collection.get(
                where={"node_type": "rem"},
                limit=1,
                include=["documents", "metadatas"]
            )
            
            if rem_results["documents"]:
                print("\n📌 Sample REM Insight:")
                doc = rem_results["documents"][0]
                metadata = rem_results["metadatas"][0]
                
                # Extract question and synthesis from the document
                if "Question:" in doc and "Synthesis:" in doc:
                    parts = doc.split("Synthesis:")
                    question = parts[0].replace("Question:", "").strip()
                    synthesis = parts[1].strip() if len(parts) > 1 else "N/A"
                    
                    print(f"  🔍 Implicit Question: {question}")
                    print(f"  📅 Source Years: {metadata.get('source_years', [])}")
                    print(f"  💡 Synthesis Preview: {synthesis[:200]}...")
                
        except Exception as e:
            print(f"  ⚠️ Could not display sample: {e}")
    
    def analyze_rem_insights(self):
        """Analyze the REM insights generated during the experiment"""
        print("\n\n🌙 REM Cycle Analysis")
        print("-" * 40)
        
        try:
            # Get all REM nodes
            rem_results = self.store.collection.get(
                where={"node_type": "rem"},
                limit=100,
                include=["documents", "metadatas"]
            )
            
            total_rem = len(rem_results["ids"])
            print(f"\n📊 Total REM insights generated: {total_rem}")
            
            if total_rem > 0:
                # Analyze themes in REM questions
                print("\n🔍 Sample Implicit Questions Discovered:")
                
                # Show up to 5 examples
                for i in range(min(5, total_rem)):
                    doc = rem_results["documents"][i]
                    metadata = rem_results["metadatas"][i]
                    
                    if "Question:" in doc:
                        question = doc.split("Synthesis:")[0].replace("Question:", "").strip()
                        # Parse JSON string for years
                        years = metadata.get('source_years', '[]')
                        if isinstance(years, str):
                            try:
                                import json
                                years = json.loads(years)
                            except:
                                years = []
                        print(f"\n  {i+1}. {question}")
                        print(f"     Connecting years: {years}")
                
                # Search REM insights for key 1922 themes
                print("\n\n🎯 REM Insights on Key 1922 Themes:")
                themes = ["League of Nations", "Soviet Russia", "democracy", "war"]
                
                for theme in themes:
                    results = self.store.query(
                        text=theme,
                        k=2,
                        filter={"node_type": "rem"}
                    )
                    
                    if results['documents']:
                        print(f"\n  📌 {theme}:")
                        for doc in results['documents'][:1]:  # Show first result
                            if "Synthesis:" in doc:
                                synthesis = doc.split("Synthesis:")[-1].strip()
                                print(f"    {synthesis[:150]}...")
                
        except Exception as e:
            print(f"❌ Error analyzing REM insights: {e}")
    
    def load_1922_articles(self) -> List[Dict[str, Any]]:
        """Load all 1922 articles from local JSON files"""
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
        pattern = os.path.join(data_dir, "1922_*.json")
        
        articles = []
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Transform to expected format
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title', 'Unknown'),
                        'author': article_data.get('author', 'Unknown'),
                        'year': article_data.get('year', 1922),
                        'volume': article_data.get('volume', 1),
                        'issue': article_data.get('issue', 1),
                        'url': article_data.get('url', ''),
                        'article_id': f"1922-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:
                        articles.append(article)
            except Exception as e:
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def process_article(self, article: Dict[str, Any]):
        """Process a single article with enhanced metadata"""
        # 1. Chunk the article
        try:
            chunks = self.chunker.chunk_article(article)
            print(f"  📄 Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  Chunking failed: {e}")
            # Fallback to word-based chunking
            text = article.get('text', '')
            if not text:
                return
            
            # Simple chunking by splitting into ~300 word chunks
            words = text.split()
            chunk_size = 300
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                # Add article context
                if article.get('title'):
                    chunk_text = f"[Article: {article['title']}]\n\n{chunk_text}"
                    
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'article_id': article['article_id'],
                        'title': article['title'],
                        'year': article['year'],
                        'chunk_index': len(chunks),
                        'word_count': len(chunk_words),
                        'chunker': 'fallback'
                    }
                })
            print(f"  📄 Created {len(chunks)} chunks (fallback word-based method)")
        
        # 2. Store chunks with enhanced metadata
        if chunks:
            texts = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                metadata['node_type'] = 'chunk'
                metadata['year'] = chunk.get('year', article['year'])
                metadata['article_title'] = chunk.get('title', article['title'])
                metadata['chunk_index'] = chunk.get('chunk_index', 0)
                metadata['article_id'] = chunk.get('article_id', article['article_id'])
                metadata['quarter'] = self.get_quarter(article)  # Add quarter info
                if 'section' in chunk:
                    metadata['section'] = chunk['section']
                metadatas.append(metadata)
            
            try:
                self.store.add(texts, metadatas)
                print(f"  ✅ Stored {len(chunks)} chunks")
            except Exception as e:
                print(f"  ❌ Storage failed: {e}")
        
        # 3. Generate and store article summary
        try:
            summary_prompt = f"""Extract the key insight from this 1922 Foreign Affairs piece. State the main idea directly as fact or insight, without mentioning "the article" or what it "discusses". Be concrete and specific.
            
            Title: {article['title']}
            Text excerpt: {article['text'][:1000]}..."""
            
            summary = self.llm.generate_sync(
                prompt=summary_prompt,
                system_prompt="You extract key insights from historical texts. State ideas directly without meta-commentary.",
                max_tokens=100
            )
            
            print(f"  💡 Summary: {summary[:100]}...")
            
            # Store summary
            self.store.add(
                [summary],
                [{
                    'node_type': 'summary',
                    'year': article['year'],
                    'article_title': article['title'],
                    'article_id': article['article_id'],
                    'quarter': self.get_quarter(article)
                }]
            )
        except Exception as e:
            print(f"  ⚠️  Summary generation failed: {e}")
    
    def analyze_results(self):
        """Analyze stored content including REM insights"""
        # Get collection statistics
        stats = self.store.get_stats()
        print(f"\n📈 Collection Statistics:")
        print(f"  Total nodes: {stats['total_documents']}")
        
        # Count by node type
        node_types = defaultdict(int)
        try:
            all_docs = self.store.collection.get(limit=10000, include=["metadatas"])
            for metadata in all_docs["metadatas"]:
                node_type = metadata.get('node_type', 'unknown')
                node_types[node_type] += 1
            
            print(f"\n📊 Nodes by type:")
            for node_type, count in sorted(node_types.items()):
                print(f"  - {node_type}: {count}")
                
        except Exception as e:
            print(f"  ❌ Error counting node types: {e}")
        
        # Query for key themes
        themes = [
            "League of Nations",
            "Versailles Treaty", 
            "Soviet Russia",
            "reparations",
            "popular diplomacy",
            "democracy and foreign policy"
        ]
        
        print("\n\n🔍 Searching for key 1922 themes:")
        for theme in themes:
            try:
                # Search across all node types
                results = self.store.query(theme, k=3)
                
                if results['documents']:
                    print(f"\n📌 {theme}:")
                    seen_types = set()
                    for i, doc in enumerate(results['documents']):
                        metadata = results['metadatas'][i]
                        node_type = metadata.get('node_type', 'unknown')
                        
                        # Show one example per node type
                        if node_type not in seen_types:
                            seen_types.add(node_type)
                            print(f"  [{node_type}] {doc[:150]}...")
                            
            except Exception as e:
                print(f"\n❌ Error querying '{theme}': {e}")


def main():
    """Run the enhanced experiment with REM cycles"""
    experiment = Enhanced1922Experiment()
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("\nThis enhanced experiment will:")
    print("1. Load 1922 Foreign Affairs articles")
    print("2. Process them in quarterly batches")
    print("3. Run REM cycles after each quarter")
    print("4. Discover patterns across disparate articles")
    print("5. Analyze both direct content and REM insights\n")
    
    try:
        experiment.run()
        print("\n✅ Enhanced experiment completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()