"""
Hybrid REM Cycle - Serial database operations with batch OpenAI processing.

This version addresses ChromaDB locking issues by:
1. Serially collecting all node triplets from the database
2. Batch processing all REM dreams through OpenAI concurrently
3. Serially storing the results back to the database

This avoids concurrent database access while maintaining fast OpenAI processing.
"""

import random
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from ..llm import LLMClient, OpenAIBatchProcessor
from ..vector_store import REMVectorStore
from ..config import REM_QUESTION_PROMPT, SYNTHESIS_PROMPT, REM_SCALING_FACTOR

logger = logging.getLogger(__name__)


class REMTriplet:
    """Three nodes sampled for REM synthesis"""
    def __init__(self, nodes: List[Dict[str, Any]], current_year: Optional[int]):
        self.nodes = nodes
        self.current_year = current_year
        # Generate ID once at creation time
        self.id = f"rem_{nodes[0]['id'][:8]}_{datetime.now().timestamp()}"


class BatchREMCycle:
    """
    Hybrid REM cycle with serial DB operations and batch OpenAI processing.
    
    This implementation fixes ChromaDB locking issues by ensuring all database
    operations are performed serially, while still leveraging concurrent OpenAI
    API calls for performance.
    """
    
    def __init__(
        self, 
        store: REMVectorStore,
        llm: LLMClient,
        api_key: str,  # Required for compatibility
        max_concurrent: int = 50
    ):
        self.store = store
        self.llm = llm
        self.batch_processor = OpenAIBatchProcessor(llm, max_concurrent=max_concurrent)
        
    async def run_batch_rem_cycles(
        self, 
        num_cycles: int, 
        current_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run multiple REM cycles using the hybrid approach.
        
        Args:
            num_cycles: Number of REM dreams to process
            current_year: Year context for sampling
            
        Returns:
            Statistics about the REM processing
        """
        logger.info(f"Starting hybrid batch REM processing for {num_cycles} cycles")
        print(f"\n🌙 Starting Hybrid Batch REM cycle")
        print(f"🎯 Processing {num_cycles} REM dreams...")
        
        # Stage 1: Serial collection of node triplets
        print("📚 Stage 1: Collecting node triplets from database...")
        triplets = self._collect_triplets_serial(num_cycles, current_year)
        
        if not triplets:
            logger.warning("No valid triplets collected")
            return {
                'total_rem_nodes': 0,
                'valuable_syntheses': 0,
                'failed': 0
            }
        
        print(f"✅ Collected {len(triplets)} triplets")
        
        # Stage 2: Batch OpenAI processing
        print("\n🤖 Stage 2: Batch processing with OpenAI...")
        rem_insights = await self._batch_process_rem_dreams(triplets)
        
        # Stage 3: Serial storage of results
        print("\n💾 Stage 3: Storing REM insights to database...")
        stats = self._store_results_serial(rem_insights, triplets)
        
        print(f"\n✨ Hybrid REM cycle complete!")
        print(f"   • Total REM nodes: {stats['total_rem_nodes']}")
        print(f"   • Valuable syntheses: {stats['valuable_syntheses']}")
        print(f"   • Failed: {stats['failed']}")
        
        return stats
    
    def _collect_triplets_serial(
        self, 
        num_cycles: int, 
        current_year: Optional[int]
    ) -> List[REMTriplet]:
        """
        Serially collect node triplets from the database.
        
        This avoids concurrent database access issues.
        """
        triplets = []
        
        # First, get a large pool of nodes to sample from
        logger.info("Fetching node pool for sampling...")
        
        # Get all non-REM nodes
        try:
            all_results = self.store.collection.get(
                where={"node_type": {"$ne": "rem"}},
                limit=10000  # Adjust based on your database size
            )
            
            # Convert to list of node dicts
            node_pool = []
            for i in range(len(all_results["ids"])):
                node_pool.append({
                    "id": all_results["ids"][i],
                    "document": all_results["documents"][i] if i < len(all_results["documents"]) else "",
                    "metadata": all_results["metadatas"][i] if i < len(all_results["metadatas"]) else {}
                })
            
            logger.info(f"Loaded pool of {len(node_pool)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to load node pool: {e}")
            return []
        
        if len(node_pool) < 3:
            logger.warning("Not enough nodes for REM processing")
            return []
        
        # Separate nodes by year if current_year is specified
        current_year_nodes = []
        other_nodes = node_pool
        
        if current_year:
            current_year_nodes = [
                n for n in node_pool 
                if n.get("metadata", {}).get("year") == current_year
            ]
            other_nodes = [
                n for n in node_pool 
                if n.get("metadata", {}).get("year") != current_year
            ]
            logger.info(f"Found {len(current_year_nodes)} nodes from year {current_year}")
        
        # Collect triplets
        for i in range(num_cycles):
            try:
                nodes = []
                
                # Try to include one node from current year
                if current_year and current_year_nodes and len(nodes) < 3:
                    node = random.choice(current_year_nodes)
                    nodes.append(node)
                
                # Fill remaining slots from other nodes
                available = other_nodes if current_year else node_pool
                needed = 3 - len(nodes)
                
                if len(available) >= needed:
                    # Sample without replacement for this triplet
                    sampled = random.sample(available, needed)
                    nodes.extend(sampled)
                else:
                    # Not enough nodes
                    continue
                
                if len(nodes) == 3:
                    triplets.append(REMTriplet(nodes=nodes, current_year=current_year))
                    
            except Exception as e:
                logger.error(f"Error collecting triplet {i}: {e}")
                continue
        
        logger.info(f"Collected {len(triplets)} valid triplets")
        return triplets
    
    async def _batch_process_rem_dreams(
        self, 
        triplets: List[REMTriplet]
    ) -> List[Dict[str, Any]]:
        """
        Process all REM dreams through OpenAI in batches.
        
        Returns list of REM insights with questions and syntheses.
        """
        # Prepare all prompts
        question_prompts = []
        synthesis_prompts = []
        
        for triplet in triplets:
            # Format passages for question generation
            passages = self._format_passages(triplet.nodes)
            
            # Question prompt
            question_prompt = {
                'prompt': f"{REM_QUESTION_PROMPT}\n\n{passages}",
                'temperature': 0.7,
                'max_tokens': 500,
                'metadata': {'triplet_id': triplet.id, 'type': 'question'}
            }
            question_prompts.append(question_prompt)
            
            # Synthesis prompt (we'll insert the question later)
            synthesis_context = self._format_synthesis_context(triplet.nodes)
            # We'll update this with the question after we generate questions
            synthesis_prompt = {
                'prompt': synthesis_context,  # Will be updated with full prompt later
                'temperature': 0.7,
                'max_tokens': 500,
                'metadata': {'triplet_id': triplet.id, 'type': 'synthesis'}
            }
            synthesis_prompts.append(synthesis_prompt)
        
        # Process questions first
        logger.info(f"Generating {len(question_prompts)} implicit questions...")
        question_results = await self.batch_processor.batch_generate(
            question_prompts,
            progress_callback=lambda done, total: print(f"   Questions: {done}/{total}")
        )
        
        # Update synthesis prompts with questions
        questions_by_id = {}
        for result in question_results:
            if result['text']:
                triplet_id = result['metadata']['triplet_id']
                questions_by_id[triplet_id] = result['text']
        
        # Insert questions into synthesis prompts
        for i, prompt_config in enumerate(synthesis_prompts):
            triplet_id = prompt_config['metadata']['triplet_id']
            if triplet_id in questions_by_id:
                # Build the full REM synthesis prompt
                question = questions_by_id[triplet_id]
                context = f"Implicit Question: {question}\n\n{prompt_config['prompt']}"
                
                prompt_config['prompt'] = f"""Given the following implicit question and source materials, 
generate a synthesis that reveals hidden patterns and connections:

{context}

Write a concise synthesis in exactly 2 short paragraphs (150-200 words total) that directly addresses the core issue WITHOUT restating or referencing "the question" or "the implicit question".

Start your response by integrating the key concept from the question naturally into your opening sentence. 

Example opening (if the question was about state stability in crisis):
"Stability of states facing internal and external pressures reveals a consistent pattern across Iraq, Indonesia, and Afghanistan, where economic collapse and corruption create vulnerabilities that external actors exploit..."

NOT: "The implicit question explores how states maintain stability..."

Be direct and specific. Reveal non-obvious patterns across the time periods and offer insights that emerge from the juxtaposition.

Synthesis:"""
        
        # Process syntheses
        logger.info(f"Generating {len(synthesis_prompts)} syntheses...")
        synthesis_results = await self.batch_processor.batch_generate(
            synthesis_prompts,
            progress_callback=lambda done, total: print(f"   Syntheses: {done}/{total}")
        )
        
        # Combine results
        insights = []
        syntheses_by_id = {
            r['metadata']['triplet_id']: r['text'] 
            for r in synthesis_results if r['text']
        }
        
        # Log what we have
        logger.info(f"Questions generated: {len(questions_by_id)}")
        logger.info(f"Syntheses generated: {len(syntheses_by_id)}")
        
        for triplet in triplets:
            if triplet.id in questions_by_id and triplet.id in syntheses_by_id:
                insights.append({
                    'triplet_id': triplet.id,
                    'question': questions_by_id[triplet.id],
                    'synthesis': syntheses_by_id[triplet.id],
                    'nodes': triplet.nodes,
                    'current_year': triplet.current_year
                })
            else:
                logger.warning(f"Missing data for triplet {triplet.id}: "
                             f"has_question={triplet.id in questions_by_id}, "
                             f"has_synthesis={triplet.id in syntheses_by_id}")
        
        logger.info(f"Generated {len(insights)} complete REM insights")
        return insights
    
    def _store_results_serial(
        self, 
        insights: List[Dict[str, Any]], 
        triplets: List[REMTriplet]
    ) -> Dict[str, Any]:
        """
        Serially store all REM insights to the database.
        """
        stats = {
            'total_rem_nodes': 0,
            'valuable_syntheses': 0,
            'failed': 0
        }
        
        for insight in insights:
            try:
                # Prepare metadata
                nodes = insight['nodes']
                metadata = {
                    "node_type": "rem",
                    "implicit_question": insight['question'],
                    "source_node_ids": json.dumps([n['id'] for n in nodes]),
                    "source_years": json.dumps([
                        n.get('metadata', {}).get('year', 'Unknown') 
                        for n in nodes
                    ]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "generation_depth": 0,
                    "processing_method": "hybrid_batch"
                }
                
                # Add year metadata
                years = [
                    n.get('metadata', {}).get('year') 
                    for n in nodes 
                    if n.get('metadata', {}).get('year')
                ]
                
                if insight['current_year']:
                    metadata['year'] = insight['current_year']
                elif years:
                    metadata['year'] = max(years)
                
                if years:
                    metadata['year_min'] = min(years)
                    metadata['year_max'] = max(years)
                
                # Clean metadata - remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                # Create combined text for embedding
                text_for_embedding = f"{insight['question']}\n\n{insight['synthesis']}"
                
                # Store REM node
                ids = self.store.add([text_for_embedding], [metadata])
                
                if ids:
                    stats['total_rem_nodes'] += 1
                    # Check if synthesis is valuable (simple heuristic)
                    if len(insight['synthesis']) > 100:
                        stats['valuable_syntheses'] += 1
                
            except Exception as e:
                logger.error(f"Failed to store REM insight: {e}")
                stats['failed'] += 1
        
        return stats
    
    def _format_passages(self, nodes: List[Dict[str, Any]]) -> str:
        """Format node passages for question generation."""
        passages = []
        for i, node in enumerate(nodes):
            metadata = node.get('metadata', {})
            year = metadata.get('year', 'Unknown Year')
            title = metadata.get('article_title', 'Unknown')
            content = node.get('document', '')
            
            passage = (
                f"Passage {i+1} (from {title}, {year}):\n"
                f"{content} (Year: {year})"
            )
            passages.append(passage)
        
        return "\n\n".join(passages)
    
    def _format_synthesis_context(self, nodes: List[Dict[str, Any]]) -> str:
        """Format context for synthesis generation."""
        context = "Source Materials:\n\n"
        
        for i, node in enumerate(nodes):
            metadata = node.get('metadata', {})
            year = metadata.get('year', 'Unknown')
            content = node.get('document', '')
            
            context += f"Source {i+1} ({year}):\n"
            context += f"{content} (Year: {year})\n\n"
        
        return context