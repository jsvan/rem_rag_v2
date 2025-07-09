"""
REM Cycle: Random Episodic Memory synthesis for discovering implicit patterns.

The REM cycle runs periodically (e.g., monthly) to:
1. Sample 3 nodes from the knowledge base (1 recent + 2 random)
2. Find the implicit question that connects them
3. Generate a synthesis that reveals hidden patterns
4. Store as a REM node for future reference
"""

import random
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from ..llm import LLMClient
from ..vector_store import REMVectorStore
from ..core.implant import implant_knowledge_sync
from ..config import REM_QUESTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class REMSample:
    """Represents a sampled node for REM synthesis"""
    node_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None


class REMCycle:
    """Manages REM (Random Episodic Memory) synthesis cycles"""
    
    def __init__(self, llm_client: LLMClient, vector_store: REMVectorStore):
        self.llm = llm_client
        self.store = vector_store
    
    def run_cycle(self, num_dreams: int = 100, current_year: Optional[int] = None) -> List[str]:
        """
        Run a complete REM cycle with specified number of dreams.
        
        Args:
            num_dreams: Number of REM syntheses to generate
            current_year: Year context for sampling (uses current timestamp if None)
            
        Returns:
            List of node IDs for created REM nodes
        """
        print(f"\n🌙 Starting REM cycle with {num_dreams} dreams...")
        
        rem_node_ids = []
        for i in tqdm(range(num_dreams), desc="  Generating dreams", unit="dream"):
            try:
                # Sample nodes for this dream
                samples = self._sample_nodes(current_year)
                
                if not samples or len(samples) < 3:
                    print(f"  ⚠️  Insufficient nodes for dream {i+1}, skipping...")
                    continue
                
                # Generate implicit question
                question = self._find_implicit_question(samples)
                
                # Generate synthesis
                synthesis = self._generate_synthesis(samples, question)
                
                # Store REM node
                node_id = self._store_rem_node(synthesis, question, samples)
                rem_node_ids.append(node_id)
                
            except Exception as e:
                print(f"  ❌ Error in dream {i+1}: {str(e)}")
                continue
        
        print(f"✨ REM cycle complete! Created {len(rem_node_ids)} dream nodes.")
        return rem_node_ids
    
    def _sample_nodes(self, current_year: Optional[int] = None) -> List[REMSample]:
        """
        Sample 3 nodes: 1 from current year + 2 random from any time.
        
        Args:
            current_year: Year to sample recent node from
            
        Returns:
            List of 3 REMSample objects
        """
        samples = []
        
        # Get all available nodes (excluding REM nodes to avoid recursion)
        # Use collection.get() method from ChromaDB
        all_results = self.store.collection.get(
            where={"node_type": {"$ne": "rem"}},
            limit=10000
        )
        
        # Convert to list of dicts for compatibility
        all_nodes = []
        for i in range(len(all_results["ids"])):
            all_nodes.append({
                "id": all_results["ids"][i],
                "documents": [all_results["documents"][i]] if i < len(all_results["documents"]) else [""],
                "metadata": all_results["metadatas"][i] if i < len(all_results["metadatas"]) else {}
            })
        
        if not all_nodes or len(all_nodes) < 3:
            return []
        
        # Sample 1 from current year if specified
        if current_year:
            year_nodes = [
                n for n in all_nodes 
                if n.get("metadata", {}).get("year") == current_year
            ]
            if year_nodes:
                recent_node = random.choice(year_nodes)
                samples.append(self._node_to_sample(recent_node))
                # Remove from pool to avoid duplicates
                all_nodes = [n for n in all_nodes if n["id"] != recent_node["id"]]
        
        # Sample remaining nodes randomly
        remaining_needed = 3 - len(samples)
        if len(all_nodes) >= remaining_needed:
            random_nodes = random.sample(all_nodes, remaining_needed)
            samples.extend([self._node_to_sample(n) for n in random_nodes])
        
        return samples[:3]  # Ensure we have exactly 3
    
    def _node_to_sample(self, node: Dict[str, Any]) -> REMSample:
        """Convert a node dict to REMSample"""
        return REMSample(
            node_id=node["id"],
            content=node.get("documents", [""])[0],
            metadata=node.get("metadata", {}),
            timestamp=datetime.fromisoformat(node["metadata"]["timestamp"]) 
                     if "timestamp" in node.get("metadata", {}) else None
        )
    
    def _find_implicit_question(self, samples: List[REMSample]) -> str:
        """
        Find the implicit question that connects the sampled nodes.
        
        Args:
            samples: List of REMSample objects
            
        Returns:
            The implicit connecting question
        """
        # Prepare context from samples
        passages = "\n\n".join([
            f"Passage {i+1} (from {s.metadata.get('article_title', 'Unknown')}, "
            f"{s.metadata.get('year', 'Unknown year')}):\n{s.content}"
            for i, s in enumerate(samples)
        ])
        
        prompt = f"""{REM_QUESTION_PROMPT}
        
        {passages}"""
        
        response = self.llm.generate_sync(
            prompt=prompt,
            max_tokens=50
        )
        return response.strip()
    
    def _generate_synthesis(self, samples: List[REMSample], question: str) -> str:
        """
        Generate a synthesis that answers the implicit question.
        
        Args:
            samples: List of REMSample objects
            question: The implicit connecting question
            
        Returns:
            The synthesis text
        """
        # Build context
        context = f"Implicit Question: {question}\n\n"
        context += "Source Materials:\n\n"
        
        for i, sample in enumerate(samples):
            context += f"Source {i+1} ({sample.metadata.get('year', 'Unknown')}):\n"
            context += f"{sample.content}\n\n"
        
        prompt = f"""Given the following implicit question and source materials, 
        generate a synthesis that reveals hidden patterns and connections:

        {context}

        Write a concise synthesis in exactly 1-2 short paragraphs (no more than 150 words total) that:
        1. Answers the implicit question
        2. Reveals non-obvious patterns across the time periods
        3. Offers insights that emerge from the juxtaposition
        
        Be direct and specific. Focus on the most important insight. Keep paragraphs short and impactful.
        
        Synthesis:"""
        
        response = self.llm.generate_sync(
            prompt=prompt,
            max_tokens=300  # Increased to avoid truncation
        )
        return response.strip()
    
    def _store_rem_node(self, synthesis: str, question: str, samples: List[REMSample]) -> str:
        """
        Store the REM synthesis as a new node in the vector store.
        
        Args:
            synthesis: The generated synthesis text
            question: The implicit question
            samples: The source samples used
            
        Returns:
            The ID of the created node
        """
        # Prepare metadata
        metadata = {
            "node_type": "rem",
            "implicit_question": question,
            # Convert lists to JSON strings for ChromaDB compatibility
            "source_node_ids": json.dumps([s.node_id for s in samples]),
            "source_years": json.dumps([s.metadata.get("year", "Unknown") for s in samples]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add year range for searchability
        years = [s.metadata.get("year") for s in samples if s.metadata.get("year")]
        if years:
            metadata["year_min"] = min(years)
            metadata["year_max"] = max(years)
        
        # Create combined text for embedding
        text_for_embedding = f"Question: {question}\n\nSynthesis: {synthesis}"
        
        # Store in vector database
        node_ids = self.store.add(
            texts=[text_for_embedding],
            metadata=[metadata]
        )
        node_id = node_ids[0]
        
        # Now implant the REM synthesis to see how it relates to existing knowledge
        implant_result = implant_knowledge_sync(
            new_content=text_for_embedding,
            vector_store=self.store,
            llm_client=self.llm,
            metadata={
                "node_type": "synthesis",
                "source_type": "rem_synthesis",
                "rem_node_id": node_id,
                "implicit_question": question,
                "source_years": json.dumps([s.metadata.get("year", "Unknown") for s in samples]),
                "generation_depth": 1,
                "synthesis_type": "rem_level",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            context_filter=None,  # Look across all knowledge
            k=3
        )
        
        logger.debug(f"REM implant result: valuable={implant_result['is_valuable']}, "
                    f"existing_count={implant_result['existing_count']}")
        
        return node_id
    
    def query_rem_insights(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query REM nodes for insights related to a topic.
        
        Args:
            query: The query string
            top_k: Number of results to return
            
        Returns:
            List of relevant REM insights
        """
        results_dict = self.store.query(
            text=query,
            k=top_k,
            filter={"node_type": "rem"}
        )
        
        # Convert results to expected format
        results = []
        for i in range(len(results_dict["documents"])):
            results.append(type('Result', (), {
                'page_content': results_dict["documents"][i],
                'metadata': results_dict["metadatas"][i]
            }))
        
        insights = []
        for result in results:
            # Parse JSON strings back to lists
            source_years = result.metadata.get("source_years", "[]")
            if isinstance(source_years, str):
                try:
                    source_years = json.loads(source_years)
                except:
                    source_years = []
                    
            insights.append({
                "question": result.metadata.get("implicit_question", ""),
                "synthesis": result.page_content.split("Synthesis: ")[-1] if "Synthesis: " in result.page_content else result.page_content,
                "source_years": source_years,
                "score": result.metadata.get("score", 0)
            })
        
        return insights