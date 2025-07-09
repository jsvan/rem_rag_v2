"""READING cycle implementation - chronological processing with synthesis"""

import logging
from typing import List, Dict, Optional
import asyncio
from datetime import datetime

from ..llm import LLMClient, StructuredLLMClient
from ..vector_store import REMVectorStore
from ..data_processing.sentence_chunker import SentenceAwareChunker
from ..core.entity_processor import EntityProcessor
from ..core.implant import implant_knowledge
from ..config import (
    SYNTHESIS_PROMPT,
    IMPLANT_SYNTHESIS_PROMPT,
    NODE_TYPES,
    NEIGHBORS_COUNT,
    ARTICLE_SUMMARY_PROMPT
)

logger = logging.getLogger(__name__)


class ReadingCycle:
    """
    Implements the READING phase of REM RAG.
    
    For each chunk:
    1. Store the original chunk
    2. Extract and process entities with implant synthesis
    3. Generate chunk-level synthesis comparing with existing knowledge
    4. Build interconnected knowledge graph
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        structured_llm: StructuredLLMClient,
        vector_store: REMVectorStore,
        chunker: SentenceAwareChunker
    ):
        """Initialize the reading cycle."""
        self.llm = llm_client
        self.structured_llm = structured_llm
        self.vector_store = vector_store
        self.chunker = chunker
        self.entity_processor = EntityProcessor(structured_llm, vector_store)
    
    async def process_article(self, article: Dict) -> Dict[str, any]:
        """
        Process a single article through the READING cycle.
        
        Args:
            article: Dict with text, year, article_id, title
            
        Returns:
            Processing statistics
        """
        start_time = datetime.now()
        
        # Chunk the article
        chunks = self.chunker.chunk_article(article)
        logger.info(f"Processing article '{article['title']}' with {len(chunks)} chunks")
        
        # Statistics tracking
        stats = {
            "article_id": article["article_id"],
            "title": article["title"],
            "year": article["year"],
            "total_chunks": len(chunks),
            "total_entities": 0,
            "valuable_syntheses": 0,
            "chunk_syntheses": 0,
            "processing_time": 0
        }
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Process this chunk
            chunk_stats = await self._process_chunk(chunk)
            
            # Update statistics
            stats["total_entities"] += chunk_stats["entities_extracted"]
            stats["valuable_syntheses"] += chunk_stats["entity_syntheses"]
            if chunk_stats["chunk_synthesis_stored"]:
                stats["chunk_syntheses"] += 1
        
        # Generate and store article summary after processing all chunks
        summary_stats = await self._generate_article_summary(article)
        stats["summary_generated"] = summary_stats["summary_stored"]
        stats["summary_entities"] = summary_stats["entities_extracted"]
        stats["valuable_syntheses"] += summary_stats["entity_syntheses"]
        
        # Calculate processing time
        stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Completed article: {stats['total_chunks']} chunks, "
            f"{stats['total_entities']} entities, "
            f"{stats['valuable_syntheses']} syntheses in {stats['processing_time']:.1f}s"
        )
        
        return stats
    
    async def _process_chunk(self, chunk: Dict) -> Dict[str, any]:
        """
        Process a single chunk through READING cycle.
        
        Steps:
        1. Store original chunk
        2. Extract and process entities
        3. Generate chunk-level synthesis
        """
        chunk_text = chunk["text"]
        chunk_meta = chunk.get("metadata", {})
        year = chunk_meta.get("year", 1922)
        article_id = chunk_meta.get("article_id", "")
        
        # 1. Store original chunk through implant
        chunk_metadata = {
            "year": year,
            "article_id": article_id,
            "title": chunk_meta.get("article_title", ""),
            "chunk_index": chunk_meta.get("chunk_index", 0),
            "source_type": "chunk",
            "node_type": "chunk",
            "generation_depth": 0,
            "word_count": chunk_meta.get("word_count", 0),
            "chunker": chunk_meta.get("chunker", "sentence_aware"),
            **chunk_meta
        }
        
        # Use implant to store chunk (it will compare with existing knowledge)
        chunk_implant_result = await implant_knowledge(
            new_content=chunk_text,
            vector_store=self.vector_store,
            llm_client=self.llm,
            metadata=chunk_metadata,
            context_filter={"year": {"$lt": year}},  # Only look at earlier knowledge
            k=NEIGHBORS_COUNT
        )
        chunk_id = chunk_implant_result["original_id"]
        
        # 2. Extract and process entities
        entity_stats = await self.entity_processor.process_chunk_entities(
            chunk_text=chunk_text,
            year=year,
            article_id=article_id,
            chunk_metadata={"chunk_id": chunk_id}
        )
        
        # Note: Chunk synthesis already handled by implant_knowledge above
        chunk_synthesis_stored = chunk_implant_result["is_valuable"]
        
        return {
            "entities_extracted": entity_stats["total_entities"],
            "entity_syntheses": entity_stats["valuable_syntheses"],
            "chunk_synthesis_stored": chunk_synthesis_stored
        }
    
    async def _generate_article_summary(self, article: Dict) -> Dict[str, any]:
        """
        Generate and store article summary after all chunks are processed.
        
        Steps:
        1. Generate summary using full article text
        2. Store summary through implant
        3. Extract and process entities from summary
        """
        # Generate summary
        summary = await self.llm.generate(
            prompt=f"{ARTICLE_SUMMARY_PROMPT}\n\nTitle: {article['title']}\n\nText: {article['text']}",
            temperature=0.7,
            max_tokens=200
        )
        
        # Store summary through implant
        summary_metadata = {
            "year": article["year"],
            "article_id": article["article_id"],
            "title": article["title"],
            "source_type": "article_summary",
            "node_type": "summary",
            "generation_depth": 0
        }
        
        summary_implant_result = await implant_knowledge(
            new_content=summary,
            vector_store=self.vector_store,
            llm_client=self.llm,
            metadata=summary_metadata,
            context_filter={"year": {"$lt": article["year"]}},
            k=NEIGHBORS_COUNT
        )
        
        # Extract entities from summary
        entity_stats = await self.entity_processor.process_chunk_entities(
            chunk_text=summary,
            year=article["year"],
            article_id=article["article_id"],
            chunk_metadata={"summary_id": summary_implant_result["original_id"], "is_summary": True}
        )
        
        return {
            "summary_stored": True,
            "summary_id": summary_implant_result["original_id"],
            "summary_synthesis_stored": summary_implant_result["is_valuable"],
            "entities_extracted": entity_stats["total_entities"],
            "entity_syntheses": entity_stats["valuable_syntheses"]
        }
    
    async def process_articles_chronologically(
        self,
        articles: List[Dict],
        max_concurrent: int = 3
    ) -> Dict[str, any]:
        """
        Process multiple articles chronologically.
        
        IMPORTANT: Articles should be pre-sorted by year.
        
        Args:
            articles: List of article dicts sorted by year
            max_concurrent: Max concurrent article processing
            
        Returns:
            Aggregated statistics
        """
        total_start = datetime.now()
        
        # Process in batches by year to maintain chronological order
        articles_by_year = {}
        for article in articles:
            year = article["year"]
            if year not in articles_by_year:
                articles_by_year[year] = []
            articles_by_year[year].append(article)
        
        all_stats = []
        
        # Process each year sequentially
        for year in sorted(articles_by_year.keys()):
            year_articles = articles_by_year[year]
            logger.info(f"Processing {len(year_articles)} articles from {year}")
            
            # Process articles within the same year concurrently
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(article):
                async with semaphore:
                    return await self.process_article(article)
            
            year_stats = await asyncio.gather(*[
                process_with_semaphore(article) for article in year_articles
            ])
            
            all_stats.extend(year_stats)
        
        # Aggregate statistics
        total_stats = {
            "total_articles": len(articles),
            "total_chunks": sum(s["total_chunks"] for s in all_stats),
            "total_entities": sum(s["total_entities"] for s in all_stats),
            "total_syntheses": sum(s["valuable_syntheses"] for s in all_stats),
            "chunk_syntheses": sum(s["chunk_syntheses"] for s in all_stats),
            "total_time": (datetime.now() - total_start).total_seconds(),
            "years_processed": sorted(articles_by_year.keys())
        }
        
        logger.info(f"Completed processing: {total_stats}")
        
        return total_stats
    
    def get_entity_evolution(self, entity_name: str, limit: int = 50) -> List[Dict]:
        """
        Get the evolution of understanding about an entity over time.
        
        Args:
            entity_name: Name of entity to track
            limit: Max results to return
            
        Returns:
            List of knowledge nodes sorted by year
        """
        results = self.vector_store.get_by_entity(entity_name, limit=limit)
        
        # Sort by year
        if results["documents"]:
            combined = list(zip(
                results["documents"],
                results["metadatas"],
                results["ids"]
            ))
            
            # Sort by year, then by generation depth
            combined.sort(key=lambda x: (
                x[1].get("year", 9999),
                x[1].get("generation_depth", 0)
            ))
            
            return [
                {
                    "text": doc,
                    "year": meta.get("year"),
                    "node_type": meta.get("node_type"),
                    "generation_depth": meta.get("generation_depth"),
                    "id": doc_id
                }
                for doc, meta, doc_id in combined
            ]
        
        return []