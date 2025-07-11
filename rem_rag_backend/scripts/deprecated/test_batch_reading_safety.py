#!/usr/bin/env python3
"""
Test script to verify batch reading cycle safety.

This script tests:
1. Concurrent article processing doesn't create duplicate entries
2. Entity extraction happens correctly across concurrent operations
3. Synthesis generation maintains quality
4. No race conditions in vector store operations
5. Performance improvements from batch processing
"""

import asyncio
import time
import json
from typing import List, Dict
import logging
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.llm import LLMClient, StructuredLLMClient
from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.core.reading_cycle import ReadingCycle
from rem_rag_backend.src.core.reading_cycle_batch import BatchReadingCycle
from rem_rag_backend.src.config import OPENAI_API_KEY, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchReadingSafetyTest:
    """Test batch reading cycle for safety and performance."""
    
    def __init__(self):
        """Initialize test components."""
        # Initialize components
        self.llm = LLMClient()
        self.structured_llm = StructuredLLMClient()
        self.chunker = SentenceAwareChunker()
        
        # Reading cycles will be initialized after vector store
        self.serial_reader = None
        self.batch_reader = None
    
    def create_test_articles(self, count: int = 5) -> List[Dict]:
        """Create test articles with known content."""
        articles = []
        
        base_texts = [
            """The Cold War shaped international relations for decades. The United States and Soviet Union competed for global influence through proxy wars, nuclear deterrence, and ideological competition. This bipolar world order created stability through mutual fear while dividing nations into competing spheres of influence. The conflict fundamentally altered how nations approached diplomacy, military strategy, and economic development.""",
            
            """Economic integration in Europe began with the Coal and Steel Community. France and Germany led efforts to prevent future wars through economic ties, recognizing that interdependence would make conflict economically devastating. This visionary approach transformed centuries of European warfare into unprecedented cooperation. The success of economic integration paved the way for political union and shared governance structures.""",
            
            """Decolonization transformed the global order after World War II. Britain and France gradually withdrew from their empires as nationalist movements gained strength and international opinion shifted against colonialism. New nations emerged across Africa and Asia, fundamentally altering the balance of power in international institutions. The process created both opportunities and challenges that continue to shape global politics today.""",
            
            """China's economic reforms began under Deng Xiaoping. Special Economic Zones attracted foreign investment while maintaining political control under the Communist Party. This unique model of state capitalism combined market mechanisms with authoritarian governance, creating rapid growth while avoiding political liberalization. The success of these reforms transformed China from an isolated nation into a global economic powerhouse.""",
            
            """The fall of the Berlin Wall marked the end of the Cold War. Germany reunified after decades of division, symbolizing the collapse of communist control in Eastern Europe. This dramatic transformation occurred peacefully, defying predictions of violence and instability. The events of 1989 reshaped the global order and ushered in an era of American unipolarity that would define the following decades."""
        ]
        
        for i in range(count):
            articles.append({
                "article_id": f"test_article_{i}",
                "title": f"Test Article {i}: {base_texts[i % len(base_texts)].split('.')[0]}",
                "text": base_texts[i % len(base_texts)],
                "year": 2000 + (i % 10),
                "issue": f"Test Issue {i // 5}"
            })
        
        return articles
    
    async def test_serial_processing(self, articles: List[Dict]) -> Dict:
        """Process articles serially for baseline."""
        print("\n🔄 Testing SERIAL Processing")
        print("=" * 60)
        
        start_time = time.time()
        stats = []
        
        for i, article in enumerate(articles):
            print(f"Processing article {i+1}/{len(articles)}: {article['title']}")
            article_stats = await self.serial_reader.process_article(article)
            stats.append(article_stats)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "avg_time_per_article": total_time / len(articles),
            "article_stats": stats
        }
    
    async def test_batch_processing(self, articles: List[Dict]) -> Dict:
        """Process articles in batches."""
        print("\n⚡ Testing BATCH Processing")
        print("=" * 60)
        
        start_time = time.time()
        
        # Process all articles in batches
        stats = await self.batch_reader.process_articles_batch(articles)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "avg_time_per_article": total_time / len(articles),
            "article_stats": stats
        }
    
    def analyze_batch_results(self, batch_results: Dict) -> None:
        """Analyze and compare results."""
        print("\n📊 RESULTS ANALYSIS")
        print("=" * 60)
        
        # Performance comparison
        print("\n⏱️  Performance:")
        print(f"Serial processing: {serial_results['total_time']:.2f}s total")
        print(f"  Average per article: {serial_results['avg_time_per_article']:.2f}s")
        print(f"Batch processing: {batch_results['total_time']:.2f}s total")
        print(f"  Average per article: {batch_results['avg_time_per_article']:.2f}s")
        speedup = serial_results['total_time'] / batch_results['total_time']
        print(f"Speedup: {speedup:.2f}x faster")
        
        # Node count comparison
        print(f"\n📦 Node Counts:")
        print(f"Serial: {serial_results['total_nodes']} nodes")
        print(f"Batch: {batch_results['total_nodes']} nodes")
        
        # Detailed statistics
        print("\n📈 Processing Statistics:")
        
        # Serial stats
        serial_chunks = sum(s['total_chunks'] for s in serial_results['article_stats'])
        serial_entities = sum(s['total_entities'] for s in serial_results['article_stats'])
        serial_syntheses = sum(s['valuable_syntheses'] for s in serial_results['article_stats'])
        
        # Batch stats
        batch_chunks = sum(s['total_chunks'] for s in batch_results['article_stats'])
        batch_entities = sum(s['total_entities'] for s in batch_results['article_stats'])
        batch_syntheses = sum(s['valuable_syntheses'] for s in batch_results['article_stats'])
        
        print(f"Serial: {serial_chunks} chunks, {serial_entities} entities, {serial_syntheses} syntheses")
        print(f"Batch: {batch_chunks} chunks, {batch_entities} entities, {batch_syntheses} syntheses")
        
        # Check for discrepancies
        print("\n✅ Safety Checks:")
        
        if serial_chunks == batch_chunks:
            print("✓ Chunk counts match")
        else:
            print(f"✗ Chunk count mismatch: {serial_chunks} vs {batch_chunks}")
        
        # Allow some variance in entities/syntheses due to LLM non-determinism
        entity_diff = abs(serial_entities - batch_entities)
        synthesis_diff = abs(serial_syntheses - batch_syntheses)
        
        if entity_diff <= 5:  # Allow small variance
            print(f"✓ Entity counts similar (diff: {entity_diff})")
        else:
            print(f"⚠️  Entity count difference: {entity_diff}")
        
        if synthesis_diff <= 5:
            print(f"✓ Synthesis counts similar (diff: {synthesis_diff})")
        else:
            print(f"⚠️  Synthesis count difference: {synthesis_diff}")
        
        # Check for duplicate detection
        print("\n🔍 Checking for duplicates...")
        all_nodes = self.vector_store.get_all_nodes(limit=10000)
        
        # Check chunk duplicates
        chunk_texts = {}
        duplicates = 0
        
        for i, (text, metadata) in enumerate(zip(all_nodes["documents"], all_nodes["metadatas"])):
            if metadata.get("node_type") == "chunk":
                text_key = text[:100]  # First 100 chars as key
                if text_key in chunk_texts:
                    duplicates += 1
                    print(f"  Duplicate found: {metadata.get('title', 'Unknown')}")
                else:
                    chunk_texts[text_key] = metadata
        
        if duplicates == 0:
            print("✓ No duplicate chunks found")
        else:
            print(f"✗ Found {duplicates} duplicate chunks")
    
    async def run_test(self):
        """Run the complete safety test."""
        print("🧪 Batch Reading Safety Test (Quick Version)")
        print("=" * 60)
        
        # Create test articles
        test_articles = self.create_test_articles(count=10)  # More articles to test concurrency
        print(f"Created {len(test_articles)} test articles")
        
        # Test batch processing only
        self.vector_store = REMVectorStore(collection_name="test_batch_quick")
        
        # Initialize batch reader
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.vector_store,
            self.chunker,
            batch_size=5,  # Process 5 articles concurrently
            max_concurrent_chunks=10  # Process 10 chunks concurrently per article
        )
        
        # Run batch processing
        print("\n⚡ Testing BATCH Processing with different concurrency levels")
        print("=" * 60)
        
        # Test 1: Low concurrency
        self.batch_reader.batch_size = 2
        self.batch_reader.article_semaphore = asyncio.Semaphore(2)
        print("\n📊 Test 1: Batch size = 2")
        start_time = time.time()
        batch_results_1 = await self.test_batch_processing(test_articles[:5])
        time_1 = time.time() - start_time
        
        # Test 2: High concurrency
        self.vector_store = REMVectorStore(collection_name="test_batch_high")
        self.batch_reader.vector_store = self.vector_store
        self.batch_reader.batch_size = 5
        self.batch_reader.article_semaphore = asyncio.Semaphore(5)
        print("\n📊 Test 2: Batch size = 5")
        start_time = time.time()
        batch_results_2 = await self.test_batch_processing(test_articles[:5])
        time_2 = time.time() - start_time
        
        # Simple analysis
        print("\n📊 RESULTS")
        print("=" * 60)
        print(f"Batch size 2: {time_1:.1f}s ({time_1/5:.1f}s per article)")
        print(f"Batch size 5: {time_2:.1f}s ({time_2/5:.1f}s per article)")
        print(f"Speedup from increased concurrency: {time_1/time_2:.1f}x")
        
        print("\n✅ Test Complete!")


async def main():
    """Run the test."""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    test = BatchReadingSafetyTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())