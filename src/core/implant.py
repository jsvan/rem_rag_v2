"""
Modular implant function for comparing new knowledge with existing knowledge.

This module provides a reusable function that implements the "implant" step
used throughout the REM RAG system. ALL content enters the system through
this function, which:
1. ALWAYS stores the original content first
2. Queries for related existing knowledge
3. Optionally generates and stores synthesis if valuable
"""

from typing import Dict, Optional, Any
import logging
import json

from ..llm import LLMClient
from ..vector_store import REMVectorStore
from ..config import IMPLANT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


async def implant_knowledge(
    new_content: str,
    vector_store: REMVectorStore,
    llm_client: LLMClient,
    metadata: Dict[str, Any],
    context_filter: Optional[Dict[str, Any]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Universal gateway for adding content to the knowledge base.
    ALWAYS stores the original content, then optionally generates synthesis.
    
    Args:
        new_content: The new information to implant
        vector_store: Vector store instance
        llm_client: LLM client instance
        metadata: Metadata for the original content
        context_filter: Filter for querying existing knowledge
        k: Number of neighbors to retrieve (default 3)
        
    Returns:
        dict: {
            "original_id": str,  # ID of stored original content
            "synthesis_id": str or None,  # ID of synthesis if valuable
            "synthesis": str or None,  # The synthesis text
            "is_valuable": bool,  # Whether synthesis was stored
            "existing_count": int  # Number of existing neighbors found
        }
    """
    # STEP 1: ALWAYS store the original content first
    original_ids = vector_store.add([new_content], [metadata])
    original_id = original_ids[0] if original_ids else None
    logger.info(f"Stored original content: {original_id}")
    
    # STEP 2: Query for related existing knowledge
    existing_results = vector_store.query(
        text=new_content,
        k=k,
        filter=context_filter
    )
    
    result = {
        "original_id": original_id,
        "synthesis_id": None,
        "synthesis": None,
        "is_valuable": False,
        "existing_count": len(existing_results["documents"])
    }
    
    # STEP 3: Generate synthesis if we have existing knowledge
    if existing_results["documents"]:
        # We have relevant existing knowledge
        existing_text = "\n\n---\n\n".join(existing_results["documents"][:k])
        
        # Generate synthesis
        synthesis = await llm_client.generate(
            prompt=IMPLANT_SYNTHESIS_PROMPT.format(
                new_info=new_content,
                existing_knowledge=existing_text
            ),
            temperature=0.7,
            max_tokens=500  # Increased to avoid truncation
        )
        
        result["synthesis"] = synthesis
        
        # Store synthesis if not redundant
        if synthesis.strip() != "NOTHING":
            result["is_valuable"] = True
            
            # Prepare synthesis metadata
            synthesis_metadata = metadata.copy()
            synthesis_metadata.update({
                "node_type": "synthesis",
                "generation_depth": synthesis_metadata.get("generation_depth", 0) + 1,
                "parent_ids": json.dumps([original_id]),  # Link to original content (JSON string for ChromaDB)
                "synthesis_type": synthesis_metadata.get("synthesis_type", "implant")
            })
            
            # Store the synthesis node
            synthesis_ids = vector_store.add([synthesis], [synthesis_metadata])
            result["synthesis_id"] = synthesis_ids[0] if synthesis_ids else None
            
            logger.info(f"Stored valuable synthesis: {result['synthesis_id']}")
        else:
            logger.debug("Synthesis was NOTHING - not storing")
    else:
        logger.debug("No existing knowledge found for synthesis")
    
    return result


def implant_knowledge_sync(
    new_content: str,
    vector_store: REMVectorStore,
    llm_client: LLMClient,
    metadata: Dict[str, Any],
    context_filter: Optional[Dict[str, Any]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Synchronous version of implant_knowledge for use in non-async contexts.
    ALWAYS stores the original content, then optionally generates synthesis.
    
    Args:
        new_content: The new information to implant
        vector_store: Vector store instance
        llm_client: LLM client instance
        metadata: Metadata for the original content
        context_filter: Filter for querying existing knowledge
        k: Number of neighbors to retrieve (default 3)
        
    Returns:
        dict: {
            "original_id": str,  # ID of stored original content
            "synthesis_id": str or None,  # ID of synthesis if valuable
            "synthesis": str or None,  # The synthesis text
            "is_valuable": bool,  # Whether synthesis was stored
            "existing_count": int  # Number of existing neighbors found
        }
    """
    # STEP 1: ALWAYS store the original content first
    original_ids = vector_store.add([new_content], [metadata])
    original_id = original_ids[0] if original_ids else None
    logger.info(f"Stored original content: {original_id}")
    
    # STEP 2: Query for related existing knowledge
    existing_results = vector_store.query(
        text=new_content,
        k=k,
        filter=context_filter
    )
    
    result = {
        "original_id": original_id,
        "synthesis_id": None,
        "synthesis": None,
        "is_valuable": False,
        "existing_count": len(existing_results["documents"])
    }
    
    # STEP 3: Generate synthesis if we have existing knowledge
    if existing_results["documents"]:
        # We have relevant existing knowledge
        existing_text = "\n\n---\n\n".join(existing_results["documents"][:k])
        
        # Generate synthesis
        synthesis = llm_client.generate_sync(
            prompt=IMPLANT_SYNTHESIS_PROMPT.format(
                new_info=new_content,
                existing_knowledge=existing_text
            ),
            temperature=0.7,
            max_tokens=500  # Increased to avoid truncation
        )
        
        result["synthesis"] = synthesis
        
        # Store synthesis if not redundant
        if synthesis.strip() != "NOTHING":
            result["is_valuable"] = True
            
            # Prepare synthesis metadata
            synthesis_metadata = metadata.copy()
            synthesis_metadata.update({
                "node_type": "synthesis",
                "generation_depth": synthesis_metadata.get("generation_depth", 0) + 1,
                "parent_ids": json.dumps([original_id]),  # Link to original content (JSON string for ChromaDB)
                "synthesis_type": synthesis_metadata.get("synthesis_type", "implant")
            })
            
            # Store the synthesis node
            synthesis_ids = vector_store.add([synthesis], [synthesis_metadata])
            result["synthesis_id"] = synthesis_ids[0] if synthesis_ids else None
            
            logger.info(f"Stored valuable synthesis: {result['synthesis_id']}")
        else:
            logger.debug("Synthesis was NOTHING - not storing")
    else:
        logger.debug("No existing knowledge found for synthesis")
    
    return result