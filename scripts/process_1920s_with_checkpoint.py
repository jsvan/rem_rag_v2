#!/usr/bin/env python3
"""Process 1920s with checkpoint/resume capability"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.reading_cycle import ReadingCycle
from src.core.rem_cycle import REMCycle
from src.utils.data_loader import load_years_data


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CHECKPOINT_FILE = "data/1920s_checkpoint.json"


def load_checkpoint():
    """Load checkpoint data if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "processed_years": [],
        "rem_cycles_completed": {},
        "stats": {
            "articles_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "syntheses_stored": 0,
            "rem_insights": 0
        }
    }


def save_checkpoint(checkpoint_data):
    """Save checkpoint data."""
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Checkpoint saved to {CHECKPOINT_FILE}")


async def process_year_with_checkpoint(year: int, reading_cycle: ReadingCycle, 
                                      rem_cycle: REMCycle, checkpoint: dict):
    """Process a single year with checkpoint support."""
    
    # Check if year already processed
    if year in checkpoint["processed_years"]:
        logger.info(f"Year {year} already processed, skipping...")
        return
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing Year: {year}")
    logger.info(f"{'='*50}")
    
    # Load articles for this year
    year_articles = load_years_data([year])
    logger.info(f"Found {len(year_articles)} articles for {year}")
    
    if not year_articles:
        logger.warning(f"No articles found for {year}")
        checkpoint["processed_years"].append(year)
        save_checkpoint(checkpoint)
        return
    
    # Process articles
    start_time = datetime.now()
    try:
        reading_stats = await reading_cycle.process_articles_chronologically(
            year_articles,
            max_concurrent=3
        )
        
        # Update checkpoint stats
        checkpoint["stats"]["articles_processed"] += reading_stats["total_articles"]
        checkpoint["stats"]["chunks_created"] += reading_stats["total_chunks"]
        checkpoint["stats"]["entities_extracted"] += reading_stats["total_entities"]
        checkpoint["stats"]["syntheses_stored"] += reading_stats["total_syntheses"]
        
        logger.info(f"\n{year} READING complete:")
        logger.info(f"  Articles: {reading_stats['total_articles']}")
        logger.info(f"  Chunks: {reading_stats['total_chunks']}")
        logger.info(f"  Entities: {reading_stats['total_entities']}")
        logger.info(f"  Syntheses: {reading_stats['total_syntheses']}")
        
        # Mark year as processed
        checkpoint["processed_years"].append(year)
        save_checkpoint(checkpoint)
        
    except Exception as e:
        logger.error(f"Error processing {year}: {e}")
        raise


async def run_quarterly_rem(quarter_name: str, year: int, num_cycles: int,
                          rem_cycle: REMCycle, checkpoint: dict):
    """Run REM cycles for a quarter with checkpoint support."""
    
    # Check if already completed
    if checkpoint["rem_cycles_completed"].get(quarter_name, 0) >= num_cycles:
        logger.info(f"REM cycles for {quarter_name} already completed")
        return
    
    # Calculate remaining cycles
    completed = checkpoint["rem_cycles_completed"].get(quarter_name, 0)
    remaining = num_cycles - completed
    
    if remaining > 0:
        logger.info(f"\nRunning {remaining} REM cycles for {quarter_name} (year: {year})...")
        
        try:
            rem_stats = await rem_cycle.run_rem_batch(
                current_year=year,
                num_cycles=remaining
            )
            
            checkpoint["stats"]["rem_insights"] += rem_stats["valuable_insights"]
            checkpoint["rem_cycles_completed"][quarter_name] = num_cycles
            save_checkpoint(checkpoint)
            
            logger.info(f"{quarter_name} REM complete:")
            logger.info(f"  Insights: {rem_stats['valuable_insights']}/{rem_stats['total_attempted']}")
            
        except Exception as e:
            logger.error(f"Error in REM cycles for {quarter_name}: {e}")
            # Save partial progress
            checkpoint["rem_cycles_completed"][quarter_name] = completed + rem_stats.get("total_attempted", 0)
            save_checkpoint(checkpoint)
            raise


async def process_1920s_resumable():
    """Process 1920s with resume capability."""
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    if checkpoint["processed_years"]:
        logger.info(f"Resuming from checkpoint. Already processed: {checkpoint['processed_years']}")
    
    # Initialize components
    logger.info("Initializing components...")
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="rem_rag_1920s")
    chunker = SentenceAwareChunker(max_words=300, min_chars=150)
    
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
    
    # Process each year
    years = list(range(1920, 1930))
    
    for year in years:
        await process_year_with_checkpoint(year, reading_cycle, rem_cycle, checkpoint)
    
    # Run quarterly REM cycles
    quarters = [
        ("1920-1922", 1922, 15),  # 3 years * 5 cycles
        ("1923-1924", 1924, 10),  # 2 years * 5 cycles
        ("1925-1927", 1927, 15),  # 3 years * 5 cycles
        ("1928-1929", 1929, 10),  # 2 years * 5 cycles
    ]
    
    for quarter_name, year, num_cycles in quarters:
        await run_quarterly_rem(quarter_name, year, num_cycles, rem_cycle, checkpoint)
    
    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("1920s DECADE PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles: {checkpoint['stats']['articles_processed']}")
    logger.info(f"Total chunks: {checkpoint['stats']['chunks_created']}")
    logger.info(f"Total entities: {checkpoint['stats']['entities_extracted']}")
    logger.info(f"Total syntheses: {checkpoint['stats']['syntheses_stored']}")
    logger.info(f"Total REM insights: {checkpoint['stats']['rem_insights']}")
    
    # Sample insights
    logger.info("\n=== Sample Decade Insights ===")
    rem_results = vector_store.search(
        query="patterns sovereignty intervention international order",
        filter={"node_type": "rem_dream"},
        k=5
    )
    
    for i, (text, meta) in enumerate(zip(rem_results['documents'], rem_results['metadatas'])):
        logger.info(f"\nInsight {i+1}:")
        logger.info(text)
    
    # Save final stats
    final_stats = {
        "checkpoint": checkpoint,
        "completion_date": datetime.now().isoformat(),
        "total_nodes": len(vector_store.search("", k=10000)['ids'])
    }
    
    with open("data/1920s_final_stats.json", 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    logger.info("\nProcessing complete! Stats saved to data/1920s_final_stats.json")


if __name__ == "__main__":
    asyncio.run(process_1920s_resumable())