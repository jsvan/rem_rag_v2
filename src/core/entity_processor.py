"""Entity extraction and processing with implant synthesis"""

import logging
from typing import List, Dict, Optional
import asyncio

from ..llm import StructuredLLMClient
from ..vector_store import REMVectorStore
from ..core.implant import implant_knowledge
from ..config import (
    ENTITY_EXTRACTION_PROMPT, 
    IMPLANT_SYNTHESIS_PROMPT,
    NODE_TYPES
)

logger = logging.getLogger(__name__)


class EntityProcessor:
    """
    Processes text chunks to extract entity learnings and synthesize with existing knowledge.
    
    The "implant" system ensures every new piece of knowledge is compared with
    what we already know, creating a rich interconnected knowledge graph.
    """
    
    def __init__(self, llm_client: StructuredLLMClient, vector_store: REMVectorStore):
        """
        Initialize the entity processor.
        
        Args:
            llm_client: LLM client for entity extraction and synthesis
            vector_store: Vector store for querying existing knowledge
        """
        self.llm = llm_client
        self.vector_store = vector_store
    
    async def process_chunk_entities(
        self, 
        chunk_text: str, 
        year: int, 
        article_id: str,
        chunk_metadata: Optional[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        Extract entities from chunk and process with implant synthesis.
        
        Args:
            chunk_text: Text to extract entities from
            year: Publication year
            article_id: Source article ID
            chunk_metadata: Additional metadata for the chunk
            
        Returns:
            Dict with statistics about processing
        """
        # Extract entities and learnings
        entities = await self.llm.extract_entities(chunk_text, ENTITY_EXTRACTION_PROMPT)
        
        logger.info(f"Extracted {len(entities)} entities from chunk")
        
        # Track statistics
        stats = {
            "total_entities": len(entities),
            "new_learnings": 0,
            "valuable_syntheses": 0,
            "redundant_learnings": 0,
            "entity_names": []
        }
        
        # Process each entity learning
        for entity_data in entities:
            entity_name = entity_data["entity"]
            learning = entity_data["learning"]
            
            stats["entity_names"].append(entity_name)
            
            # Process with implant synthesis
            synthesis_result = await self._process_entity_learning(
                entity_name=entity_name,
                learning=learning,
                year=year,
                article_id=article_id,
                chunk_metadata=chunk_metadata
            )
            
            # Update statistics
            if synthesis_result["stored_synthesis"]:
                stats["valuable_syntheses"] += 1
            elif synthesis_result["was_redundant"]:
                stats["redundant_learnings"] += 1
            else:
                stats["new_learnings"] += 1
        
        return stats
    
    async def _process_entity_learning(
        self,
        entity_name: str,
        learning: str,
        year: int,
        article_id: str,
        chunk_metadata: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Process a single entity learning with implant synthesis.
        
        Returns:
            Dict indicating what was stored
        """
        # Prepare metadata for the learning
        learning_metadata = {
            "entity": entity_name,
            "year": year,
            "article_id": article_id,
            "source_type": "entity_extraction",
            "node_type": "learning",
            "generation_depth": 0,
            **(chunk_metadata or {})  # Include any chunk metadata
        }
        
        # Use implant function to store learning and generate synthesis
        implant_result = await implant_knowledge(
            new_content=learning,
            vector_store=self.vector_store,
            llm_client=self.llm,
            metadata=learning_metadata,
            context_filter={"entity": entity_name},
            k=3
        )
        
        # Check if the synthesis indicated redundancy
        is_redundant = implant_result["synthesis"] and implant_result["synthesis"].strip() == "NOTHING"
        
        return {
            "stored_synthesis": implant_result["is_valuable"],
            "was_redundant": is_redundant,
            "learning_id": implant_result["original_id"]
        }
    
    async def batch_process_chunks(
        self,
        chunks: List[Dict],
        max_concurrent: int = 5
    ) -> Dict[str, int]:
        """
        Process multiple chunks concurrently.
        
        Args:
            chunks: List of chunk dicts with text, year, article_id
            max_concurrent: Max concurrent processing
            
        Returns:
            Aggregated statistics
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(chunk):
            async with semaphore:
                return await self.process_chunk_entities(
                    chunk_text=chunk["text"],
                    year=chunk["year"],
                    article_id=chunk["article_id"],
                    chunk_metadata=chunk.get("metadata")
                )
        
        # Process all chunks
        results = await asyncio.gather(*[
            process_with_semaphore(chunk) for chunk in chunks
        ])
        
        # Aggregate statistics
        total_stats = {
            "total_chunks": len(chunks),
            "total_entities": sum(r["total_entities"] for r in results),
            "new_learnings": sum(r["new_learnings"] for r in results),
            "valuable_syntheses": sum(r["valuable_syntheses"] for r in results),
            "redundant_learnings": sum(r["redundant_learnings"] for r in results),
            "unique_entities": len(set(
                entity for r in results for entity in r["entity_names"]
            ))
        }
        
        logger.info(
            f"Processed {total_stats['total_chunks']} chunks: "
            f"{total_stats['total_entities']} entities, "
            f"{total_stats['valuable_syntheses']} syntheses, "
            f"{total_stats['redundant_learnings']} redundant"
        )
        
        return total_stats