"""Async batch processing version of the READING cycle for faster processing"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

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


class BatchReadingCycle:
    """
    Async batch processing version of the READING phase.
    
    Processes multiple articles concurrently for better performance.
    Each article still goes through the same steps:
    1. Chunk the article
    2. Store chunks with implant synthesis
    3. Extract and process entities
    4. Generate article summary
    
    But now multiple articles can be processed in parallel.
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        structured_llm: StructuredLLMClient,
        vector_store: REMVectorStore,
        chunker: SentenceAwareChunker,
        batch_size: int = 5,
        max_concurrent_chunks: int = 10
    ):
        """
        Initialize the batch reading cycle.
        
        Args:
            llm_client: LLM client for generation
            structured_llm: Structured LLM for entity extraction
            vector_store: Vector store for persistence
            chunker: Sentence-aware chunker
            batch_size: Number of articles to process concurrently
            max_concurrent_chunks: Max chunks to process concurrently within an article
        """
        self.llm = llm_client
        self.structured_llm = structured_llm
        self.vector_store = vector_store
        self.chunker = chunker
        self.entity_processor = EntityProcessor(structured_llm, vector_store)
        self.batch_size = batch_size
        self.max_concurrent_chunks = max_concurrent_chunks
        
        # Semaphores to control concurrency
        self.article_semaphore = asyncio.Semaphore(batch_size)
        self.chunk_semaphore = asyncio.Semaphore(max_concurrent_chunks)
    
    async def process_articles_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Process multiple articles in batches.
        
        Args:
            articles: List of article dicts with text, year, article_id, title
            
        Returns:
            List of processing statistics for each article
        """
        start_time = time.time()
        total_articles = len(articles)
        
        logger.info(f"Starting batch processing of {total_articles} articles")
        logger.info(f"Batch size: {self.batch_size}, Max concurrent chunks: {self.max_concurrent_chunks}")
        
        # Process articles in batches
        tasks = []
        for article in articles:
            task = self._process_article_with_semaphore(article)
            tasks.append(task)
        
        # Wait for all articles to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        stats = []
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing article {articles[i]['title']}: {result}")
                errors += 1
                stats.append({
                    "article_id": articles[i]["article_id"],
                    "title": articles[i]["title"],
                    "year": articles[i]["year"],
                    "error": str(result),
                    "total_chunks": 0,
                    "total_entities": 0,
                    "valuable_syntheses": 0
                })
            else:
                stats.append(result)
        
        # Log summary
        total_time = time.time() - start_time
        successful = total_articles - errors
        logger.info(
            f"Batch processing complete: {successful}/{total_articles} articles "
            f"processed successfully in {total_time:.1f}s "
            f"({total_time/total_articles:.1f}s per article average)"
        )
        
        return stats
    
    async def _process_article_with_semaphore(self, article: Dict) -> Dict:
        """Process a single article with semaphore control."""
        async with self.article_semaphore:
            return await self.process_article(article)
    
    async def process_article(self, article: Dict) -> Dict:
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
        
        # Process chunks concurrently with semaphore control
        chunk_tasks = []
        for i, chunk in enumerate(chunks):
            task = self._process_chunk_with_semaphore(chunk, i, len(chunks))
            chunk_tasks.append(task)
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Aggregate results
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {i+1}: {result}")
            else:
                stats["total_entities"] += result["entities_extracted"]
                stats["valuable_syntheses"] += result["entity_syntheses"]
                if result["chunk_synthesis_stored"]:
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
    
    async def _process_chunk_with_semaphore(self, chunk: Dict, index: int, total: int) -> Dict:
        """Process a chunk with semaphore control."""
        async with self.chunk_semaphore:
            logger.debug(f"Processing chunk {index+1}/{total}")
            return await self._process_chunk(chunk)
    
    async def _process_chunk(self, chunk: Dict) -> Dict:
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
        
        # Clean metadata - ChromaDB cannot handle None values
        chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
        
        # Use implant to store chunk (it will compare with existing knowledge)
        try:
            chunk_implant_result = await implant_knowledge(
                new_content=chunk_text,
                vector_store=self.vector_store,
                llm_client=self.llm,
                metadata=chunk_metadata,
                context_filter={"year": {"$lt": year}},  # Only look at earlier knowledge
                k=NEIGHBORS_COUNT
            )
            chunk_id = chunk_implant_result["original_id"]
        except Exception as e:
            # Log the error and continue processing
            logger.error(f"Failed to implant chunk: {e}")
            # Return minimal stats to continue processing
            return {
                "chunk_id": None,
                "entities_extracted": 0,
                "entity_syntheses": 0,
                "chunk_synthesis_stored": False
            }
        
        # 2. Extract and process entities
        entity_stats = await self.entity_processor.process_chunk_entities(
            chunk_text=chunk_text,
            year=year,
            article_id=article_id,
            chunk_metadata={
                "chunk_id": chunk_id,
                "article_title": chunk_meta.get("article_title", "")
            }
        )
        
        # Return statistics
        return {
            "chunk_id": chunk_id,
            "entities_extracted": entity_stats["total_entities"],
            "entity_syntheses": entity_stats["valuable_syntheses"],
            "chunk_synthesis_stored": chunk_implant_result.get("is_valuable", False)
        }
    
    async def _generate_article_summary(self, article: Dict) -> Dict:
        """
        Generate and store article-level summary.
        
        Args:
            article: The article dict
            
        Returns:
            Summary statistics
        """
        # Generate summary
        summary = await self.llm.generate(
            prompt=ARTICLE_SUMMARY_PROMPT.format(article_text=article["text"]),
            max_tokens=300
        )
        
        # Store summary through implant
        summary_metadata = {
            "year": article["year"],
            "article_id": article["article_id"],
            "title": article["title"],
            "node_type": "summary",
            "source_type": "summary",
            "generation_depth": 0
        }
        
        # Clean metadata - ChromaDB cannot handle None values
        summary_metadata = {k: v for k, v in summary_metadata.items() if v is not None}
        
        summary_result = await implant_knowledge(
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
            chunk_metadata={
                "chunk_id": summary_result["original_id"],
                "article_title": article["title"],
                "is_summary": True
            }
        )
        
        return {
            "summary_stored": True,
            "summary_id": summary_result["original_id"],
            "entities_extracted": entity_stats["total_entities"],
            "entity_syntheses": entity_stats["valuable_syntheses"]
        }