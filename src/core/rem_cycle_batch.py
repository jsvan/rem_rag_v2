"""
REM Cycle with OpenAI Batch API support for 50% cost savings.

This version processes all REM dreams in a single batch request:
1. Sample all node triplets upfront
2. Create batch JSONL file with all questions and syntheses
3. Submit to OpenAI Batch API
4. Process results when complete
"""

import random
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLMClient
from ..vector_store import REMVectorStore
from ..config import REM_QUESTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class REMSample:
    """Represents a sampled node for REM synthesis"""
    node_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None


@dataclass
class REMBatchItem:
    """Represents a single REM dream to be processed in batch"""
    custom_id: str
    samples: List[REMSample]
    current_year: Optional[int]


class REMCycleBatch:
    """Batch-enabled REM cycle using OpenAI Batch API"""
    
    def __init__(self, llm_client: LLMClient, vector_store: REMVectorStore):
        self.llm = llm_client
        self.store = vector_store
    
    def run_cycle(self, current_year: Optional[int] = None) -> List[str]:
        """
        Run a complete REM cycle using batch processing.
        
        Args:
            current_year: Year context for sampling
            
        Returns:
            List of node IDs for created REM nodes
        """
        # Count non-REM nodes
        all_results = self.store.collection.get(
            where={"node_type": {"$ne": "rem"}}, 
            limit=10000
        )
        non_rem_count = len(all_results["ids"])
        
        # Apply n/4 scaling
        from ..config import REM_SCALING_FACTOR
        num_dreams = max(1, int(non_rem_count * REM_SCALING_FACTOR))
        
        print(f"\n🌙 Starting Batch REM cycle")
        print(f"📊 Database has {non_rem_count} non-REM nodes")
        print(f"🎯 Preparing {num_dreams} REM dreams for batch processing...")
        
        # Step 1: Sample all node triplets upfront
        batch_items = self._prepare_batch_items(num_dreams, current_year)
        
        if not batch_items:
            print("⚠️  No valid samples found for REM cycle")
            return []
        
        print(f"✅ Prepared {len(batch_items)} dreams for batch processing")
        
        # Step 2: Create batch file
        batch_file_path = self._create_batch_file(batch_items)
        
        # Step 3: Submit batch job
        print("📤 Submitting batch to OpenAI...")
        batch_id = self._submit_batch(batch_file_path)
        
        if not batch_id:
            print("❌ Failed to submit batch")
            return []
        
        print(f"✅ Batch submitted with ID: {batch_id}")
        
        # Step 4: Wait for completion
        print("⏳ Waiting for batch to complete (checking every 60 seconds)...")
        results = self._wait_for_batch(batch_id)
        
        if not results:
            print("❌ Batch processing failed")
            return []
        
        # Step 5: Process results and store REM nodes
        print(f"📥 Processing {len(results)} completed dreams...")
        rem_node_ids = self._process_batch_results(results, batch_items)
        
        print(f"✨ Batch REM cycle complete! Created {len(rem_node_ids)} dream nodes.")
        return rem_node_ids
    
    def _prepare_batch_items(self, num_dreams: int, current_year: Optional[int]) -> List[REMBatchItem]:
        """Prepare all REM batch items by sampling node triplets."""
        batch_items = []
        
        for i in range(num_dreams):
            try:
                samples = self._sample_nodes(current_year)
                if samples and len(samples) >= 3:
                    batch_items.append(REMBatchItem(
                        custom_id=f"rem_dream_{i}_{datetime.now().timestamp()}",
                        samples=samples,
                        current_year=current_year
                    ))
            except Exception as e:
                logger.error(f"Error preparing dream {i}: {e}")
                continue
        
        return batch_items
    
    def _create_batch_file(self, batch_items: List[REMBatchItem]) -> Path:
        """Create JSONL batch file with all REM processing requests."""
        batch_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        for item in batch_items:
            # Create request for finding implicit question
            question_request = self._create_question_request(item)
            batch_file.write(json.dumps(question_request) + '\n')
            
            # Create request for generating synthesis
            synthesis_request = self._create_synthesis_request(item)
            batch_file.write(json.dumps(synthesis_request) + '\n')
        
        batch_file.close()
        return Path(batch_file.name)
    
    def _create_question_request(self, item: REMBatchItem) -> dict:
        """Create batch request for finding implicit question."""
        passages = "\n\n".join([
            f"Passage {i+1} (from {s.metadata.get('article_title', 'Unknown')}, "
            f"{s.metadata.get('year', 'Unknown year')}):\n{s.content}"
            for i, s in enumerate(item.samples)
        ])
        
        prompt = f"""{REM_QUESTION_PROMPT}
        
        {passages}"""
        
        return {
            "custom_id": f"{item.custom_id}_question",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
        }
    
    def _create_synthesis_request(self, item: REMBatchItem) -> dict:
        """Create batch request for generating synthesis."""
        # We'll use a placeholder for the question since we don't have it yet
        # In processing, we'll match up question and synthesis responses
        context = "Source Materials:\n\n"
        
        for i, sample in enumerate(item.samples):
            context += f"Source {i+1} ({sample.metadata.get('year', 'Unknown')}):\n"
            context += f"{sample.content}\n\n"
        
        prompt = f"""Given the following source materials and the implicit question (to be determined), 
        generate a synthesis that reveals hidden patterns and connections:

        {context}

        Write a concise synthesis in exactly 1-2 short paragraphs (no more than 150 words total) that:
        1. Answers the implicit question
        2. Reveals non-obvious patterns across the time periods
        3. Offers insights that emerge from the juxtaposition
        
        Be direct and specific. Focus on the most important insight. Keep paragraphs short and impactful.
        
        Synthesis:"""
        
        return {
            "custom_id": f"{item.custom_id}_synthesis",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
        }
    
    def _submit_batch(self, batch_file_path: Path) -> Optional[str]:
        """Submit batch file to OpenAI and return batch ID."""
        try:
            import openai
            
            # Upload file
            with open(batch_file_path, 'rb') as f:
                file_response = openai.files.create(
                    file=f,
                    purpose="batch"
                )
            
            # Create batch
            batch_response = openai.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Clean up temp file
            batch_file_path.unlink()
            
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            # Clean up temp file
            if batch_file_path.exists():
                batch_file_path.unlink()
            return None
    
    def _wait_for_batch(self, batch_id: str, check_interval: int = 60) -> Optional[List[dict]]:
        """Wait for batch to complete and return results."""
        import openai
        
        start_time = time.time()
        check_count = 0
        
        while True:
            try:
                batch = openai.batches.retrieve(batch_id)
                
                if batch.status == "completed":
                    # Download results
                    elapsed_minutes = (time.time() - start_time) / 60
                    print(f"✅ Batch completed in {elapsed_minutes:.1f} minutes! Downloading results...")
                    
                    # Show final stats
                    counts = batch.request_counts
                    total = counts.total if hasattr(counts, 'total') else 0
                    completed = counts.completed if hasattr(counts, 'completed') else 0
                    failed = counts.failed if hasattr(counts, 'failed') else 0
                    print(f"   📊 Total: {total}, Completed: {completed}, Failed: {failed}")
                    
                    file_response = openai.files.content(batch.output_file_id)
                    results = []
                    
                    for line in file_response.text.strip().split('\n'):
                        if line:
                            results.append(json.loads(line))
                    
                    return results
                
                elif batch.status in ["failed", "expired", "cancelled"]:
                    print(f"❌ Batch {batch.status}: {batch.errors}")
                    return None
                
                else:
                    # Still processing
                    check_count += 1
                    elapsed_minutes = (time.time() - start_time) / 60
                    
                    counts = batch.request_counts
                    # request_counts is an object, not a dict
                    total = counts.total if hasattr(counts, 'total') else 0
                    completed = counts.completed if hasattr(counts, 'completed') else 0
                    failed = counts.failed if hasattr(counts, 'failed') else 0
                    
                    if total > 0:
                        progress = (completed / total) * 100
                        # Estimate time remaining based on progress
                        if completed > 0:
                            rate = completed / elapsed_minutes  # requests per minute
                            remaining = total - completed
                            eta_minutes = remaining / rate if rate > 0 else 0
                            print(f"⏳ [{elapsed_minutes:.1f}m elapsed] Status: {batch.status} - {completed}/{total} completed ({progress:.1f}%), {failed} failed - ETA: {eta_minutes:.1f}m")
                        else:
                            print(f"⏳ [{elapsed_minutes:.1f}m elapsed] Status: {batch.status} - {completed}/{total} completed ({progress:.1f}%), {failed} failed")
                    else:
                        print(f"⏳ [{elapsed_minutes:.1f}m elapsed] Status: {batch.status} - preparing batch...")
                    
                    # Add a note every 10 checks (10 minutes)
                    if check_count % 10 == 0:
                        print("   💡 Batch API processes jobs within 24 hours, often much faster")
                    
                    time.sleep(check_interval)
                    
            except Exception as e:
                logger.error(f"Error checking batch status: {e}")
                return None
    
    def _process_batch_results(self, results: List[dict], batch_items: List[REMBatchItem]) -> List[str]:
        """Process batch results and store REM nodes."""
        # Group results by dream ID
        questions = {}
        syntheses = {}
        
        for result in results:
            custom_id = result["custom_id"]
            
            if result["response"]["status_code"] != 200:
                logger.error(f"Request failed: {custom_id} - {result['response']}")
                continue
            
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            
            if custom_id.endswith("_question"):
                dream_id = custom_id.replace("_question", "")
                questions[dream_id] = content.strip()
            elif custom_id.endswith("_synthesis"):
                dream_id = custom_id.replace("_synthesis", "")
                syntheses[dream_id] = content.strip()
        
        # Store REM nodes
        rem_node_ids = []
        
        for item in batch_items:
            dream_id = item.custom_id
            
            if dream_id in questions and dream_id in syntheses:
                question = questions[dream_id]
                synthesis = syntheses[dream_id]
                
                # Store REM node
                node_id = self._store_rem_node(
                    synthesis, 
                    question, 
                    item.samples, 
                    item.current_year
                )
                
                if node_id:
                    rem_node_ids.append(node_id)
            else:
                logger.warning(f"Missing results for dream {dream_id}")
        
        return rem_node_ids
    
    # Copy the existing helper methods from rem_cycle.py
    def _sample_nodes(self, current_year: Optional[int] = None) -> List[REMSample]:
        """Sample 3 nodes: 1 from current year + 2 random from any time."""
        samples = []
        
        # Get all available nodes (excluding REM nodes to avoid recursion)
        sample_filter = {"node_type": {"$ne": "rem"}}
        
        # Get a large random sample to ensure variety
        all_results = self.store.sample(
            n=5000,  # Half the previous limit, but now properly randomized
            filter=sample_filter
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
    
    def _store_rem_node(self, synthesis: str, question: str, samples: List[REMSample], current_year: Optional[int] = None) -> str:
        """Store the REM synthesis as a new node in the vector store."""
        # Prepare metadata
        metadata = {
            "node_type": "rem",
            "implicit_question": question,
            # Convert lists to JSON strings for ChromaDB compatibility
            "source_node_ids": json.dumps([s.node_id for s in samples]),
            "source_years": json.dumps([s.metadata.get("year", "Unknown") for s in samples]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_depth": 0,
            "processing_method": "batch"  # Mark as batch-processed
        }
        
        # Add year metadata
        years = [s.metadata.get("year") for s in samples if s.metadata.get("year")]
        if current_year:
            metadata["year"] = current_year
        elif years:
            metadata["year"] = max(years)
            
        # Add year range for searchability
        if years:
            metadata["year_min"] = min(years)
            metadata["year_max"] = max(years)
        
        # Create combined text for embedding
        text_for_embedding = f"Question: {question}\n\nSynthesis: {synthesis}"
        
        # Store REM node directly (no implant synthesis needed)
        ids = self.store.add([text_for_embedding], [metadata])
        
        logger.debug(f"REM node stored directly: id={ids[0] if ids else 'None'}")
        
        return ids[0] if ids else None