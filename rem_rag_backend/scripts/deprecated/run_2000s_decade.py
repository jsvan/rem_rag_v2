#!/usr/bin/env python3
"""
Process the entire 2000s decade of Foreign Affairs articles with yearly REM cycles.

This script:
1. Processes articles from 2000-2009 chronologically
2. Uses the modular implant function for all content
3. Runs REM cycles at the end of each year
4. Tracks progress and generates statistics
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.llm import LLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.implant import implant_knowledge_sync
from rem_rag_backend.src.core.rem_cycle import REMCycle
from rem_rag_backend.src.core.rem_cycle_batch import REMCycleBatch
from rem_rag_backend.src.config import REM_SCALING_FACTOR


class Decade2000sProcessor:
    """Process the entire 2000s decade with yearly REM cycles"""
    
    def __init__(self, use_batch_rem=True):
        self.llm = LLMClient()
        self.store = REMVectorStore()
        self.chunker = SentenceAwareChunker()
        # Use batch REM cycle for 50% cost savings
        if use_batch_rem:
            self.rem_cycle = REMCycleBatch(self.llm, self.store)
            print("🎯 Using Batch REM processing (50% cost savings)")
        else:
            self.rem_cycle = REMCycle(self.llm, self.store)
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def run(self):
        """Execute the decade processing"""
        print("🚀 Starting 2000s Decade Processing")
        print("=" * 70)
        print(f"Processing articles from 2000-2009 with yearly REM cycles")
        print(f"REM scaling factor: {REM_SCALING_FACTOR} (n/{int(1/REM_SCALING_FACTOR)} dreams)")
        print("=" * 70)
        
        start_time = time.time()
        
        # Process each year
        for year in range(2000, 2010):
            print(f"\n\n{'='*60}")
            print(f"📅 YEAR {year}")
            print(f"{'='*60}")
            
            # Load and process articles for this year
            articles = self.load_year_articles(year)
            print(f"\n📚 Loaded {len(articles)} articles from {year}")
            
            # Process articles
            print(f"\n📖 Processing {year} articles...")
            for i, article in enumerate(articles):
                print(f"\n[{i+1}/{len(articles)}] {article['title'][:60]}...")
                
                try:
                    self.process_article(article, year)
                    self.stats[year]['articles_processed'] += 1
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    self.stats[year]['articles_failed'] += 1
            
            # Run REM cycle at end of year
            print(f"\n\n🌙 Running REM cycle for {year}...")
            try:
                rem_ids = self.rem_cycle.run_cycle(current_year=year)
                self.stats[year]['rem_nodes'] = len(rem_ids)
                print(f"✨ Created {len(rem_ids)} REM insights for {year}")
            except Exception as e:
                print(f"❌ REM cycle failed for {year}: {e}")
                self.stats[year]['rem_failed'] = True
            
            # DEBUG: REMOVE FOR FULL DECADE
            # input(f"\n⏸️  Year {year} complete. Press Enter to continue to {year + 1}...")
        
        # Final analysis
        elapsed = time.time() - start_time
        print(f"\n\n{'='*70}")
        print(f"📊 DECADE PROCESSING COMPLETE")
        print(f"{'='*70}")
        self.print_final_stats(elapsed)
    
    def load_year_articles(self, year: int) -> List[Dict[str, Any]]:
        """Load all articles from a specific year"""
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
        pattern = os.path.join(data_dir, f"{year}_*.json")
        
        articles = []
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Transform to expected format
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title') or 'Unknown',
                        'author': article_data.get('author') or 'Unknown',
                        'year': article_data.get('year') or year,
                        'volume': article_data.get('volume') or 0,
                        'issue': article_data.get('issue') or 0,
                        'url': article_data.get('url') or '',
                        'article_id': f"{year}-{len(articles)+1:03d}"
                    }
                    
                    if article['text']:  # Only add if we have content
                        articles.append(article)
            except Exception as e:
                print(f"  ⚠️  Error loading {os.path.basename(filepath)}: {e}")
        
        return articles
    
    def process_article(self, article: Dict[str, Any], year: int):
        """Process a single article using the implant approach"""
        # 1. Chunk the article
        try:
            chunks = self.chunker.chunk_article(article)
            print(f"  📄 Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  Chunking failed: {e}, using fallback")
            chunks = self.fallback_chunking(article)
        
        # 2. Process each chunk through implant
        if chunks:
            for j, chunk in enumerate(chunks):
                # Build metadata for chunk
                metadata = {
                    'node_type': 'chunk',
                    'year': year,
                    'article_title': article['title'],
                    'article_id': article['article_id'],
                    'chunk_index': j,
                    'author': article.get('author') or 'Unknown',
                    'volume': article.get('volume') or 0,
                    'issue': article.get('issue') or 0
                }
                
                # Add any chunk-specific metadata
                if 'metadata' in chunk:
                    metadata.update(chunk['metadata'])
                
                # Clean metadata - remove any None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                # Use implant to store chunk and generate synthesis
                try:
                    # Filter to only look at knowledge from previous years
                    context_filter = {"year": {"$lt": year}}
                    
                    result = implant_knowledge_sync(
                        new_content=chunk['text'],
                        vector_store=self.store,
                        llm_client=self.llm,
                        metadata=metadata,
                        context_filter=context_filter,
                        k=5
                    )
                    
                    self.stats[year]['chunks_stored'] += 1
                    if result['is_valuable']:
                        self.stats[year]['syntheses_created'] += 1
                        
                except Exception as e:
                    print(f"    ❌ Failed to implant chunk {j}: {e}")
                    self.stats[year]['chunks_failed'] += 1
        
        # 3. Generate and implant article summary
        try:
            summary_prompt = f"""Extract the key insight from this {year} Foreign Affairs piece. State the main idea directly as fact or insight, without mentioning "the article" or what it "discusses". Be concrete and specific.
            
Title: {article['title']}
Text excerpt: {article['text'][:1500]}..."""
            
            summary = self.llm.generate_sync(
                prompt=summary_prompt,
                system_prompt="You extract key insights from historical texts. State ideas directly without meta-commentary.",
                max_tokens=500
            )
            
            print(f"  💡 Summary: {summary[:80]}...")
            
            # Implant summary
            summary_metadata = {
                'node_type': 'summary',
                'year': year,
                'article_title': article['title'],
                'article_id': article['article_id'],
                'author': article.get('author') or 'Unknown'
            }
            
            # Clean metadata - remove any None values
            summary_metadata = {k: v for k, v in summary_metadata.items() if v is not None}
            
            # Look for connections with all previous knowledge
            result = implant_knowledge_sync(
                new_content=summary,
                vector_store=self.store,
                llm_client=self.llm,
                metadata=summary_metadata,
                context_filter=None,  # Summary can connect to any previous knowledge
                k=5
            )
            
            self.stats[year]['summaries_created'] += 1
            if result['is_valuable']:
                self.stats[year]['summary_syntheses'] += 1
                
        except Exception as e:
            print(f"  ⚠️  Summary generation failed: {e}")
            self.stats[year]['summaries_failed'] += 1
        
        # 4. Extract and process themes/entities
        try:
            self.process_article_themes(article, year)
        except Exception as e:
            print(f"  ⚠️  Theme extraction failed: {e}")
    
    def process_article_themes(self, article: Dict[str, Any], year: int):
        """Extract and process themes/entities from the article"""
        theme_prompt = f"""List any prominent entities, concepts, or themes mentioned in this text where we learn something substantial. For each, write ONE sentence about what we learn.

Title: {article['title']}
Text excerpt: {article['text'][:1500]}...

Format your response as:
- Entity/Theme: What we learn about it

Only include entities/themes where the text provides meaningful information, not just mentions."""
        
        response = self.llm.generate_sync(
            prompt=theme_prompt,
            max_tokens=500
        )
        
        # Parse themes
        themes = []
        for line in response.strip().split('\n'):
            if line.strip().startswith('-'):
                parts = line.strip('- ').split(':', 1)
                if len(parts) == 2:
                    theme, learning = parts
                    themes.append((theme.strip(), learning.strip()))
        
        print(f"  🏷️  Extracted {len(themes)} themes")
        
        # Store each theme directly (no implant synthesis)
        for theme, learning in themes:
            metadata = {
                'node_type': 'learning',
                'year': year,
                'entity': theme,
                'article_title': article['title'],
                'article_id': article['article_id'],
                'author': article.get('author') or 'Unknown'
            }
            
            # Clean metadata - remove any None values  
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Store theme learning directly
            try:
                self.store.add(
                    [f"About {theme}: {learning}"],
                    [metadata]
                )
                self.stats[year]['themes_processed'] += 1
            except Exception as e:
                print(f"    ❌ Failed to store theme {theme}: {e}")
                self.stats[year].setdefault('themes_failed', 0)
                self.stats[year]['themes_failed'] += 1
    
    def fallback_chunking(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple word-based chunking as fallback"""
        text = article.get('text', '')
        if not text:
            return []
        
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
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words),
                    'chunker': 'fallback'
                }
            })
        
        return chunks
    
    def print_final_stats(self, elapsed_time: float):
        """Print comprehensive statistics"""
        total_articles = sum(self.stats[y]['articles_processed'] for y in range(2000, 2010))
        total_chunks = sum(self.stats[y]['chunks_stored'] for y in range(2000, 2010))
        total_syntheses = sum(self.stats[y]['syntheses_created'] for y in range(2000, 2010))
        total_rem = sum(self.stats[y]['rem_nodes'] for y in range(2000, 2010))
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total articles processed: {total_articles}")
        print(f"  • Total chunks stored: {total_chunks}")
        print(f"  • Total syntheses created: {total_syntheses}")
        print(f"  • Total REM insights: {total_rem}")
        print(f"  • Processing time: {elapsed_time/60:.1f} minutes")
        
        print(f"\n📅 Year-by-Year Breakdown:")
        for year in range(2000, 2010):
            stats = self.stats[year]
            print(f"\n  {year}:")
            print(f"    • Articles: {stats['articles_processed']} processed, {stats.get('articles_failed', 0)} failed")
            print(f"    • Chunks: {stats['chunks_stored']} stored, {stats['syntheses_created']} syntheses")
            print(f"    • Summaries: {stats['summaries_created']} created, {stats['summary_syntheses']} syntheses")
            print(f"    • Themes: {stats['themes_processed']} stored directly")
            print(f"    • REM: {stats['rem_nodes']} insights generated")
        
        # Query database for final counts by node type
        print(f"\n🗄️  Database Node Counts:")
        try:
            for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
                results = self.store.collection.get(
                    where={"node_type": node_type},
                    limit=1
                )
                # Do a proper count query
                count_results = self.store.collection.get(
                    where={"node_type": node_type},
                    limit=10000
                )
                count = len(count_results["ids"])
                print(f"    • {node_type}: {count} nodes")
        except Exception as e:
            print(f"    ❌ Error querying database: {e}")


def main():
    """Run the 2000s decade processor"""
    processor = Decade2000sProcessor()
    
    print("\n⚠️  Note: Make sure OPENAI_API_KEY is set in your environment!")
    print("\nThis will process ~1,074 articles from 2000-2009.")
    print("Estimated time: 2-3 hours (REM batches process async)")
    print("Estimated cost: ~$8-12 with GPT-4o-mini")
    print("  - Direct storage for themes/REM (no implant synthesis)")
    print("  - Batch API for REM cycles (50% discount)")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        processor.run()
        print("\n✅ Decade processing completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print("Progress saved - you can resume from where you left off")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()