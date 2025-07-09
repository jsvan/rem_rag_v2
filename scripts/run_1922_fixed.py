#!/usr/bin/env python3
"""
Fixed script to run the 1922 Foreign Affairs experiment.

This version correctly uses the actual interfaces of the REM RAG system.
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient
from src.data_processing.sentence_chunker import SentenceAwareChunker


class Simple1922Experiment:
    """A simplified 1922 experiment that works with the actual interfaces"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        
    def run(self):
        """Execute the experiment"""
        print("🚀 Starting Fixed 1922 Experiment")
        print("=" * 50)
        
        # Load articles
        articles = self.load_1922_articles()
        print(f"\n📚 Loaded {len(articles)} articles from 1922")
        
        # Process articles
        print("\n📖 Processing articles...")
        for i, article in enumerate(articles):
            print(f"\n[{i+1}/{len(articles)}] {article['title'][:60]}...")
            
            try:
                # Process the article
                self.process_article(article)
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Analyze results
        print("\n\n📊 Analyzing Results")
        print("=" * 50)
        self.analyze_results()
    
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
                        'text': article_data.get('content', ''),  # chunker expects 'text'
                        'title': article_data.get('title', 'Unknown'),
                        'author': article_data.get('author', 'Unknown'),
                        'year': article_data.get('year', 1922),
                        'volume': article_data.get('volume', 1),
                        'issue': article_data.get('issue', 1),
                        'url': article_data.get('url', ''),
                        'article_id': f"1922-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:  # Only add if we have content
                        articles.append(article)
            except Exception as e:
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def process_article(self, article: Dict[str, Any]):
        """Process a single article"""
        # 1. Chunk the article
        try:
            chunks = self.chunker.chunk_article(article)  # Correct method name
            print(f"  📄 Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  Chunking failed: {e}")
            # Try basic word-based chunking
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
        
        # 2. Store chunks in vector database
        if chunks:
            texts = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            # Build metadata for each chunk
            for chunk in chunks:
                # Combine chunk-level metadata with article metadata
                metadata = chunk.get('metadata', {})
                metadata['node_type'] = 'chunk'
                metadata['year'] = chunk.get('year', article['year'])
                metadata['article_title'] = chunk.get('title', article['title'])
                metadata['chunk_index'] = chunk.get('chunk_index', 0)
                metadata['article_id'] = chunk.get('article_id', article['article_id'])
                if 'section' in chunk:
                    metadata['section'] = chunk['section']
                metadatas.append(metadata)
            
            try:
                self.store.add(texts, metadatas)
                print(f"  ✅ Stored {len(chunks)} chunks")
            except Exception as e:
                print(f"  ❌ Storage failed: {e}")
        
        # 3. Generate a summary (simplified version)
        try:
            summary_prompt = f"""Extract the key insight from this 1922 Foreign Affairs piece. State the main idea directly as fact or insight, without mentioning "the article" or what it "discusses". Be concrete and specific.
            
            Title: {article['title']}
            Text excerpt: {article['text'][:1000]}..."""
            
            summary = self.llm.generate_sync(
                prompt=summary_prompt,
                system_prompt="You extract key insights from historical texts. State ideas directly without meta-commentary. Example: Instead of 'The article argues that democracy needs education', write 'Democratic control of foreign policy requires public understanding of international relations.'",
                max_tokens=100
            )
            
            print(f"  💡 Summary: {summary[:100]}...")
            
            # Store summary as well
            self.store.add(
                [summary],
                [{
                    'node_type': 'summary',
                    'year': article['year'],
                    'article_title': article['title'],
                    'article_id': article['article_id']
                }]
            )
        except Exception as e:
            print(f"  ⚠️  Summary generation failed: {e}")
    
    def analyze_results(self):
        """Analyze what we've stored"""
        # Query for some key themes from 1922
        themes = [
            "League of Nations",
            "Versailles Treaty",
            "Soviet Russia",
            "reparations",
            "popular diplomacy"
        ]
        
        print("\n🔍 Searching for key 1922 themes:")
        for theme in themes:
            try:
                results = self.store.query(theme, k=3, filter={'year': 1922})
                
                if results['documents']:
                    print(f"\n📌 {theme}:")
                    for i, doc in enumerate(results['documents']):
                        metadata = results['metadatas'][i]
                        print(f"  - From: {metadata.get('article_title', 'Unknown')}")
                        print(f"    Preview: {doc[:100]}...")
            except Exception as e:
                print(f"\n❌ Error querying '{theme}': {e}")
        
        # Sample some random content
        print("\n\n🎲 Random sample of stored content:")
        try:
            sample = self.store.sample(n=5, filter={'year': 1922})
            if sample['documents']:
                for i, doc in enumerate(sample['documents']):
                    metadata = sample['metadatas'][i]
                    print(f"\n  [{i+1}] Type: {metadata.get('node_type', 'unknown')}")
                    print(f"      Article: {metadata.get('article_title', 'Unknown')}")
                    print(f"      Preview: {doc[:150]}...")
        except Exception as e:
            print(f"  ❌ Error sampling: {e}")


def main():
    """Run the fixed experiment"""
    experiment = Simple1922Experiment()
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("This experiment will:")
    print("1. Load 1922 Foreign Affairs articles")
    print("2. Chunk and store them in ChromaDB")
    print("3. Generate summaries using GPT-4")
    print("4. Search for key themes from 1922\n")
    
    try:
        experiment.run()
        print("\n✅ Experiment completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()