"""
Hybrid Reading Cycle - Serial database operations with batch OpenAI processing.

This version processes articles using:
1. Serial preparation of all chunks
2. Batch OpenAI processing for implant syntheses and summaries
3. Serial storage of results to database

This avoids ChromaDB locking issues while maintaining performance.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
from tqdm import tqdm

from ..llm import LLMClient, StructuredLLMClient, OpenAIBatchProcessor
from ..vector_store import REMVectorStore
from ..data_processing.sentence_chunker import SentenceAwareChunker
from ..core.entity_processor import EntityProcessor
from ..core.implant import implant_knowledge_sync
from ..config import (
    IMPLANT_SYNTHESIS_PROMPT,
    NODE_TYPES,
    NEIGHBORS_COUNT,
    ARTICLE_SUMMARY_PROMPT
)

logger = logging.getLogger(__name__)


class BatchReadingCycle:
    """
    Hybrid batch processing version of the READING phase.
    
    Key difference from original: All database operations are serial,
    only OpenAI calls are batched.
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        structured_llm: StructuredLLMClient,
        vector_store: REMVectorStore,
        chunker: SentenceAwareChunker,
        batch_size: int = 10,
        max_concurrent_chunks: int = 50
    ):
        """
        Initialize the batch reading cycle.
        
        Args:
            llm_client: LLM client for generation
            structured_llm: Structured LLM for entity extraction
            vector_store: Vector store for persistence
            chunker: Sentence-aware chunker
            batch_size: Number of articles to process in a batch
            max_concurrent_chunks: Max concurrent OpenAI calls
        """
        self.llm = llm_client
        self.structured_llm = structured_llm
        self.vector_store = vector_store
        self.chunker = chunker
        self.entity_processor = EntityProcessor(structured_llm, vector_store)
        self.batch_size = batch_size
        self.batch_processor = OpenAIBatchProcessor(llm_client, max_concurrent=max_concurrent_chunks)
    
    async def process_articles_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Process multiple articles using the hybrid approach.
        
        Args:
            articles: List of article dicts with text, year, article_id, title
            
        Returns:
            List of processing statistics for each article
        """
        start_time = time.time()
        total_articles = len(articles)
        
        logger.info(f"Starting hybrid batch processing of {total_articles} articles")
        print(f"\n📚 Processing {total_articles} articles in batches of {self.batch_size}")
        
        all_stats = []
        
        # Process articles in smaller batches to avoid memory issues
        for batch_start in range(0, total_articles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_articles)
            batch_articles = articles[batch_start:batch_end]
            
            print(f"\n📦 Processing batch {batch_start//self.batch_size + 1} "
                  f"(articles {batch_start+1}-{batch_end}/{total_articles})")
            
            try:
                batch_stats = await self._process_article_batch(batch_articles)
                all_stats.extend(batch_stats)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add error stats for failed articles
                for article in batch_articles:
                    all_stats.append({
                        "article_id": article["article_id"],
                        "title": article["title"],
                        "year": article["year"],
                        "error": str(e),
                        "total_chunks": 0,
                        "total_entities": 0,
                        "valuable_syntheses": 0
                    })
        
        # Log summary
        total_time = time.time() - start_time
        successful = sum(1 for s in all_stats if "error" not in s)
        logger.info(
            f"Hybrid batch processing complete: {successful}/{total_articles} articles "
            f"processed successfully in {total_time:.1f}s "
            f"({total_time/total_articles:.1f}s per article average)"
        )
        
        return all_stats
    
    async def _process_article_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a single batch of articles."""
        
        # Stage 1: Serial preparation - chunk all articles
        print("  📄 Stage 1: Chunking articles...")
        all_chunk_data = []
        article_chunks_map = {}  # Track which chunks belong to which article
        
        for article in articles:
            try:
                chunks = self.chunker.chunk_article(article)
                article_chunks_map[article["article_id"]] = len(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'article': article,
                        'chunk': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    all_chunk_data.append(chunk_data)
                    
            except Exception as e:
                logger.error(f"Failed to chunk article {article['title']}: {e}")
                article_chunks_map[article["article_id"]] = 0
        
        print(f"  ✅ Created {len(all_chunk_data)} chunks from {len(articles)} articles")
        
        # Stage 2: Serial database operations - store chunks and get existing knowledge
        print("\n  💾 Stage 2: Storing chunks and retrieving existing knowledge...")
        chunk_storage_data = []
        
        for chunk_data in all_chunk_data:
            try:
                # Store chunk and get existing knowledge
                storage_result = self._store_chunk_and_get_context(chunk_data)
                if storage_result:
                    chunk_storage_data.append(storage_result)
            except Exception as e:
                logger.error(f"Failed to store chunk: {e}")
        
        print(f"  ✅ Stored {len(chunk_storage_data)} chunks with context")
        
        # Stage 3: Batch OpenAI processing - generate syntheses
        print("\n  🤖 Stage 3: Generating syntheses with OpenAI...")
        synthesis_results = await self._batch_generate_syntheses(chunk_storage_data)
        
        # Stage 4: Serial storage of syntheses
        print("\n  💾 Stage 4: Storing valuable syntheses...")
        synthesis_stats = self._store_syntheses_serial(synthesis_results)
        
        # Stage 5: Batch generate article summaries
        print("\n  📝 Stage 5: Generating article summaries...")
        summary_prompts = []
        for article in articles:
            prompt_config = {
                'prompt': ARTICLE_SUMMARY_PROMPT.format(article_text=article["text"]),
                'max_tokens': 300,
                'metadata': {'article_id': article["article_id"], 'article': article}
            }
            summary_prompts.append(prompt_config)
        
        summary_results = await self.batch_processor.batch_generate(summary_prompts)
        
        # Stage 6: Serial storage of summaries
        print("\n  💾 Stage 6: Storing article summaries...")
        summary_stats = self._store_summaries_serial(summary_results)
        
        # Compile statistics
        stats = []
        for article in articles:
            article_id = article["article_id"]
            stats.append({
                "article_id": article_id,
                "title": article["title"],
                "year": article["year"],
                "total_chunks": article_chunks_map.get(article_id, 0),
                "total_entities": 0,  # Entity processing could be added later
                "valuable_syntheses": synthesis_stats.get(article_id, 0),
                "summary_generated": article_id in summary_stats,
                "processing_time": 0  # Could track individual times if needed
            })
        
        return stats
    
    def _store_chunk_and_get_context(self, chunk_data: Dict) -> Optional[Dict]:
        """Store a chunk and retrieve existing context for synthesis."""
        chunk = chunk_data['chunk']
        article = chunk_data['article']
        
        chunk_text = chunk["text"]
        chunk_meta = chunk.get("metadata", {})
        year = article["year"]
        
        # Prepare metadata
        chunk_metadata = {
            "year": year,
            "article_id": article["article_id"],
            "title": article["title"],
            "chunk_index": chunk_data['chunk_index'],
            "source_type": "chunk",
            "node_type": "chunk",
            "generation_depth": 0,
            "word_count": chunk_meta.get("word_count", 0),
            "chunker": chunk_meta.get("chunker", "sentence_aware"),
            **chunk_meta
        }
        
        # Clean metadata - ChromaDB cannot handle None values
        chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
        
        # Store chunk
        try:
            chunk_ids = self.vector_store.add([chunk_text], [chunk_metadata])
            chunk_id = chunk_ids[0] if chunk_ids else None
            
            if not chunk_id:
                return None
            
            # Query for existing related knowledge
            existing_results = self.vector_store.query(
                text=chunk_text,
                k=NEIGHBORS_COUNT,
                filter={"year": {"$lt": year}}  # Only earlier knowledge
            )
            
            return {
                'chunk_id': chunk_id,
                'chunk_text': chunk_text,
                'chunk_metadata': chunk_metadata,
                'existing_knowledge': existing_results,
                'article_id': article["article_id"]
            }
            
        except Exception as e:
            logger.error(f"Failed to store chunk: {e}")
            return None
    
    async def _batch_generate_syntheses(self, chunk_storage_data: List[Dict]) -> List[Dict]:
        """Generate syntheses for all chunks that have existing knowledge."""
        synthesis_prompts = []
        
        for data in chunk_storage_data:
            if data['existing_knowledge']['documents']:
                # Format existing knowledge with temporal context
                formatted_chunks = []
                existing = data['existing_knowledge']
                
                for i in range(min(NEIGHBORS_COUNT, len(existing["documents"]))):
                    doc = existing["documents"][i]
                    meta = existing["metadatas"][i] if i < len(existing["metadatas"]) else {}
                    year = meta.get("year", "Unknown Year")
                    formatted_chunk = f"{doc} (Year: {year})"
                    formatted_chunks.append(formatted_chunk)
                
                existing_text = "\n\n---\n\n".join(formatted_chunks)
                
                # Create synthesis prompt
                prompt_config = {
                    'prompt': IMPLANT_SYNTHESIS_PROMPT.format(
                        new_info=data['chunk_text'],
                        existing_knowledge=existing_text
                    ),
                    'temperature': 0.7,
                    'max_tokens': 500,
                    'metadata': {
                        'chunk_id': data['chunk_id'],
                        'chunk_metadata': data['chunk_metadata'],
                        'article_id': data['article_id']
                    }
                }
                synthesis_prompts.append(prompt_config)
        
        if not synthesis_prompts:
            return []
        
        # Batch generate syntheses
        print(f"     Generating {len(synthesis_prompts)} syntheses...")
        results = await self.batch_processor.batch_generate(
            synthesis_prompts
        )
        
        return results
    
    def _store_syntheses_serial(self, synthesis_results: List[Dict]) -> Dict[str, int]:
        """Store valuable syntheses serially."""
        stats = {}  # article_id -> count of valuable syntheses
        
        for result in synthesis_results:
            if result['text'] and result['text'].strip() != "NOTHING":
                try:
                    metadata = result['metadata']['chunk_metadata'].copy()
                    metadata.update({
                        "node_type": "synthesis",
                        "generation_depth": metadata.get("generation_depth", 0) + 1,
                        "parent_ids": result['metadata']['chunk_id'],
                        "synthesis_type": "implant"
                    })
                    
                    # Clean metadata
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    
                    # Store synthesis
                    self.vector_store.add([result['text']], [metadata])
                    
                    # Update stats
                    article_id = result['metadata']['article_id']
                    stats[article_id] = stats.get(article_id, 0) + 1
                    
                except Exception as e:
                    logger.error(f"Failed to store synthesis: {e}")
        
        return stats
    
    def _store_summaries_serial(self, summary_results: List[Dict]) -> Dict[str, bool]:
        """Store article summaries serially."""
        stored = {}  # article_id -> success
        
        for result in summary_results:
            if result['text']:
                try:
                    article = result['metadata']['article']
                    
                    summary_metadata = {
                        "year": article["year"],
                        "article_id": article["article_id"],
                        "title": article["title"],
                        "node_type": "summary",
                        "source_type": "summary",
                        "generation_depth": 0
                    }
                    
                    # Clean metadata
                    summary_metadata = {k: v for k, v in summary_metadata.items() if v is not None}
                    
                    # Store summary through regular vector store (not implant)
                    ids = self.vector_store.add([result['text']], [summary_metadata])
                    
                    if ids:
                        stored[article["article_id"]] = True
                        
                except Exception as e:
                    logger.error(f"Failed to store summary: {e}")
        
        return stored