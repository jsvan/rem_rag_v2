#!/usr/bin/env python3
"""Process the full 1920s decade (1920-1929) with quarterly REM cycles"""

import asyncio
import logging
import json
from datetime import datetime
import os
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.utils.data_loader import load_years_data
from src.config import REM_CYCLES_PER_BATCH


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_1920s.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def process_decade_with_rem():
    """Process the entire 1920s decade with quarterly REM cycles."""
    
    start_time = datetime.now()
    
    # Initialize components
    logger.info("Initializing components...")
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="rem_rag_1920s")
    chunker = SentenceAwareChunker(max_words=300, min_chars=150)
    
    # Create cycles
    reading_cycle = ReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=vector_store,
        chunker=chunker
    )
    
    rem_cycle = REMCycle(
        llm_client=llm,
        vector_store=vector_store
    )
    
    # Load all 1920s data
    logger.info("Loading 1920s data...")
    years = list(range(1920, 1930))  # 1920-1929
    all_articles = []
    
    for year in years:
        logger.info(f"Loading {year} articles...")
        year_articles = load_years_data([year])
        all_articles.extend(year_articles)
        logger.info(f"  Found {len(year_articles)} articles for {year}")
    
    logger.info(f"\nTotal articles to process: {len(all_articles)}")
    
    # Process by quarter with REM cycles
    quarters = [
        ("Q1", [1920, 1921, 1922]),
        ("Q2", [1923, 1924]),
        ("Q3", [1925, 1926, 1927]),
        ("Q4", [1928, 1929])
    ]
    
    total_stats = {
        "articles_processed": 0,
        "chunks_created": 0,
        "entities_extracted": 0,
        "syntheses_stored": 0,
        "rem_insights": 0,
        "processing_time": 0
    }
    
    for quarter_name, quarter_years in quarters:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {quarter_name}: {quarter_years}")
        logger.info(f"{'='*60}")
        
        # Get articles for this quarter
        quarter_articles = [a for a in all_articles if a['year'] in quarter_years]
        logger.info(f"Articles in {quarter_name}: {len(quarter_articles)}")
        
        if not quarter_articles:
            logger.warning(f"No articles found for {quarter_name}")
            continue
        
        # Process articles chronologically
        quarter_start = datetime.now()
        reading_stats = await reading_cycle.process_articles_chronologically(
            quarter_articles,
            max_concurrent=3
        )
        
        # Update total stats
        total_stats["articles_processed"] += reading_stats["total_articles"]
        total_stats["chunks_created"] += reading_stats["total_chunks"]
        total_stats["entities_extracted"] += reading_stats["total_entities"]
        total_stats["syntheses_stored"] += reading_stats["total_syntheses"]
        
        logger.info(f"\n{quarter_name} READING complete:")
        logger.info(f"  Articles: {reading_stats['total_articles']}")
        logger.info(f"  Chunks: {reading_stats['total_chunks']}")
        logger.info(f"  Entities: {reading_stats['total_entities']}")
        logger.info(f"  Syntheses: {reading_stats['total_syntheses']}")
        
        # Run REM cycles for this quarter
        logger.info(f"\nRunning REM cycles for {quarter_name}...")
        
        # Calculate number of REM cycles (5 per year of content)
        num_cycles = len(quarter_years) * 5
        logger.info(f"Running {num_cycles} REM cycles...")
        
        rem_stats = await rem_cycle.run_rem_batch(
            current_year=quarter_years[-1],  # Use last year of quarter
            num_cycles=num_cycles
        )
        
        total_stats["rem_insights"] += rem_stats["valuable_insights"]
        
        logger.info(f"\n{quarter_name} REM complete:")
        logger.info(f"  Insights discovered: {rem_stats['valuable_insights']}/{rem_stats['total_attempted']}")
        logger.info(f"  Processing time: {rem_stats['processing_time']:.1f}s")
        
        quarter_time = (datetime.now() - quarter_start).total_seconds()
        logger.info(f"\n{quarter_name} total time: {quarter_time:.1f}s")
    
    # Final statistics
    total_time = (datetime.now() - start_time).total_seconds()
    total_stats["processing_time"] = total_time
    
    logger.info(f"\n{'='*60}")
    logger.info("DECADE PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles processed: {total_stats['articles_processed']}")
    logger.info(f"Total chunks created: {total_stats['chunks_created']}")
    logger.info(f"Total entities extracted: {total_stats['entities_extracted']}")
    logger.info(f"Total syntheses stored: {total_stats['syntheses_stored']}")
    logger.info(f"Total REM insights: {total_stats['rem_insights']}")
    logger.info(f"Total processing time: {total_time/60:.1f} minutes")
    
    # Sample some insights
    logger.info("\n=== Sample REM Insights from the 1920s ===")
    
    # Get REM dreams
    rem_results = vector_store.search(
        query="",
        filter={"node_type": "rem_dream"},
        k=10
    )
    
    if rem_results['documents']:
        for i, (text, meta) in enumerate(zip(rem_results['documents'][:5], rem_results['metadatas'][:5])):
            logger.info(f"\nInsight {i+1} (Year: {meta.get('year', 'Unknown')}):")
            logger.info(f"{text}")
    
    # Analyze patterns by year
    logger.info("\n=== Node Distribution by Year ===")
    all_nodes = vector_store.search("", k=10000)
    
    year_counts = {}
    for meta in all_nodes['metadatas']:
        year = meta.get('year', 'Unknown')
        node_type = meta.get('node_type', 'unknown')
        
        if year not in year_counts:
            year_counts[year] = {}
        if node_type not in year_counts[year]:
            year_counts[year][node_type] = 0
        year_counts[year][node_type] += 1
    
    for year in sorted(year_counts.keys()):
        if isinstance(year, int):
            logger.info(f"\n{year}:")
            for node_type, count in sorted(year_counts[year].items()):
                logger.info(f"  {node_type}: {count}")
    
    # Save statistics
    stats_file = "data/1920s_processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "processing_stats": total_stats,
            "year_distribution": year_counts,
            "processing_date": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nStatistics saved to {stats_file}")
    
    # Estimated cost
    estimated_cost = total_stats['articles_processed'] * 0.01  # ~$0.01 per article
    logger.info(f"\nEstimated cost: ${estimated_cost:.2f}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    asyncio.run(process_decade_with_rem())