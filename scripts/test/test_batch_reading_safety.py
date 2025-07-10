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
sys.path.insert(0, project_root)

from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.reading_cycle import ReadingCycle
from src.core.reading_cycle_batch import BatchReadingCycle
from src.config import OPENAI_API_KEY, DATA_DIR

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
            """The Cold War shaped international relations for decades. The United States and Soviet Union competed for global influence throughout the second half of the twentieth century.

NATO and the Warsaw Pact divided Europe into competing blocs. These military alliances represented fundamentally different ideological systems and approaches to governance.

Nuclear weapons created a balance of terror that prevented direct conflict. The doctrine of mutually assured destruction paradoxically maintained peace between the superpowers.""",
            
            """Economic integration in Europe began with the Coal and Steel Community. This revolutionary approach to preventing war through economic interdependence laid the foundation for modern Europe.

France and Germany led efforts to prevent future wars through economic ties. The visionary leaders of both nations recognized that economic cooperation could heal historical wounds.

The European Economic Community expanded to include more nations. Common markets and shared institutions fostered cooperation and prosperity across the continent.""",
            
            """Decolonization transformed the global order after World War II. The dismantling of European empires created dozens of new sovereign states across Africa, Asia, and the Caribbean.

Britain and France gradually withdrew from their empires. This process varied from peaceful transitions to violent struggles for independence.

New nations in Africa and Asia sought their own paths. The Non-Aligned Movement offered an alternative to Cold War blocs and superpower dominance.""",
            
            """China's economic reforms began under Deng Xiaoping. His pragmatic approach marked a dramatic shift from Maoist ideology to market-oriented policies.

Special Economic Zones attracted foreign investment. These experimental areas demonstrated the benefits of opening China to global markets.

State-owned enterprises gradually adopted market mechanisms. Rapid growth lifted millions out of poverty and transformed China into a global economic power.""",
            
            """The fall of the Berlin Wall marked the end of the Cold War. This symbolic moment in November 1989 represented the collapse of communist control in Eastern Europe.

Germany reunified after decades of division. The integration of East and West Germany proved more challenging than initially anticipated.

Eastern European nations transitioned to democracy. The Soviet Union dissolved into independent republics, fundamentally reshaping the global order."""
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
        
        # Get final database state
        all_nodes = self.vector_store.get_all_nodes(limit=10000)
        node_count = len(all_nodes["ids"])
        
        return {
            "total_time": total_time,
            "avg_time_per_article": total_time / len(articles),
            "total_nodes": node_count,
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
        
        # Get final database state
        all_nodes = self.vector_store.get_all_nodes(limit=10000)
        node_count = len(all_nodes["ids"])
        
        return {
            "total_time": total_time,
            "avg_time_per_article": total_time / len(articles),
            "total_nodes": node_count,
            "article_stats": stats
        }
    
    def analyze_results(self, serial_results: Dict, batch_results: Dict) -> None:
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
        print("🧪 Batch Reading Safety Test")
        print("=" * 60)
        
        # Create test articles
        test_articles = self.create_test_articles(count=5)
        print(f"Created {len(test_articles)} test articles")
        
        # Test 1: Serial processing (baseline)
        self.vector_store = REMVectorStore(collection_name="test_serial")
        
        # Initialize readers with vector store
        self.serial_reader = ReadingCycle(
            self.llm, 
            self.structured_llm, 
            self.vector_store, 
            self.chunker
        )
        
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.vector_store,
            self.chunker,
            batch_size=3,
            max_concurrent_chunks=5
        )
        
        serial_results = await self.test_serial_processing(test_articles)
        
        # Test 2: Batch processing - use new collection for fair comparison
        self.vector_store = REMVectorStore(collection_name="test_batch")
        
        # Re-initialize readers with new vector store
        self.serial_reader = ReadingCycle(
            self.llm, 
            self.structured_llm, 
            self.vector_store, 
            self.chunker
        )
        
        self.batch_reader = BatchReadingCycle(
            self.llm,
            self.structured_llm,
            self.vector_store,
            self.chunker,
            batch_size=3,
            max_concurrent_chunks=5
        )
        
        batch_results = await self.test_batch_processing(test_articles)
        
        # Analyze results
        self.analyze_results(serial_results, batch_results)
        
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