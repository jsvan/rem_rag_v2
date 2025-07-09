"""
Modular implant function for comparing new knowledge with existing knowledge.

This module provides a reusable function that implements the "implant" step
used throughout the REM RAG system to generate syntheses comparing new
information with what's already known.
"""

from typing import Dict, Optional, Any
import logging

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
    Compare new knowledge with existing knowledge and generate synthesis.
    Always stores the synthesis as a separate node (unless it's "NOTHING").
    
    Args:
        new_content: The new information to implant
        vector_store: Vector store instance
        llm_client: LLM client instance
        metadata: Base metadata for the synthesis node
        context_filter: Filter for querying existing knowledge
        k: Number of neighbors to retrieve
        
    Returns:
        dict: {
            "synthesis": str,
            "is_valuable": bool,
            "synthesis_id": str or None,
            "existing_count": int
        }
    """
    # Query for related existing knowledge
    existing_results = vector_store.query(
        text=new_content,
        k=k,
        filter=context_filter
    )
    
    result = {
        "synthesis": None,
        "is_valuable": False,
        "synthesis_id": None,
        "existing_count": len(existing_results["documents"])
    }
    
    if existing_results["documents"]:
        # We have relevant existing knowledge
        existing_text = "\n\n---\n\n".join(existing_results["documents"][:3])
        
        # Generate synthesis
        synthesis = await llm_client.generate(
            prompt=IMPLANT_SYNTHESIS_PROMPT.format(
                new_info=new_content,
                existing_knowledge=existing_text
            ),
            temperature=0.7,
            max_tokens=200  # Increased to avoid truncation
        )
        
        result["synthesis"] = synthesis
        
        # Store if not redundant
        if synthesis.strip() != "NOTHING":
            result["is_valuable"] = True
            
            # Store the synthesis node
            synthesis_ids = vector_store.add([synthesis], [metadata])
            result["synthesis_id"] = synthesis_ids[0] if synthesis_ids else None
            
            logger.info(f"Stored valuable synthesis: {result['synthesis_id']}")
        else:
            logger.debug("Synthesis was NOTHING - not storing")
    else:
        logger.debug("No existing knowledge found for implant")
    
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
    
    Args:
        new_content: The new information to implant
        vector_store: Vector store instance
        llm_client: LLM client instance
        metadata: Base metadata for the synthesis node
        context_filter: Filter for querying existing knowledge
        k: Number of neighbors to retrieve
        
    Returns:
        dict: {
            "synthesis": str,
            "is_valuable": bool,
            "synthesis_id": str or None,
            "existing_count": int
        }
    """
    # Query for related existing knowledge
    existing_results = vector_store.query(
        text=new_content,
        k=k,
        filter=context_filter
    )
    
    result = {
        "synthesis": None,
        "is_valuable": False,
        "synthesis_id": None,
        "existing_count": len(existing_results["documents"])
    }
    
    if existing_results["documents"]:
        # We have relevant existing knowledge
        existing_text = "\n\n---\n\n".join(existing_results["documents"][:3])
        
        # Generate synthesis
        synthesis = llm_client.generate_sync(
            prompt=IMPLANT_SYNTHESIS_PROMPT.format(
                new_info=new_content,
                existing_knowledge=existing_text
            ),
            temperature=0.7,
            max_tokens=200  # Increased to avoid truncation
        )
        
        result["synthesis"] = synthesis
        
        # Store if not redundant
        if synthesis.strip() != "NOTHING":
            result["is_valuable"] = True
            
            # Store the synthesis node
            synthesis_ids = vector_store.add([synthesis], [metadata])
            result["synthesis_id"] = synthesis_ids[0] if synthesis_ids else None
            
            logger.info(f"Stored valuable synthesis: {result['synthesis_id']}")
        else:
            logger.debug("Synthesis was NOTHING - not storing")
    else:
        logger.debug("No existing knowledge found for implant")
    
    return result