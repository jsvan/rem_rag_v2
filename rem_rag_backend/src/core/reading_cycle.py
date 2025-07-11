"""
True Batch Reading Cycle - Uses OpenAI Batch API for 50% cost savings.

This version processes articles using the OpenAI Batch API:
1. Prepare all chunks and collect synthesis prompts
2. Create JSONL batch file with all requests
3. Submit to OpenAI Batch API
4. Wait for completion
5. Process results and store in database

Unlike the hybrid version, this trades immediate results for 50% cost savings.
"""

import json
import logging
import tempfile
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import openai

from ..llm import LLMClient, StructuredLLMClient
from ..vector_store import REMVectorStore
from ..data_processing.sentence_chunker import SentenceAwareChunker
from ..core.entity_processor import EntityProcessor
from ..core.database_writer import DatabaseWriterService
from ..config import (
    IMPLANT_SYNTHESIS_PROMPT,
    NODE_TYPES,
    NEIGHBORS_COUNT,
    ARTICLE_SUMMARY_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    OPENAI_API_KEY
)

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Represents a single request for batch processing"""
    custom_id: str
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    metadata: Dict[str, Any] = None


class ReadingCycle:
    """
    True batch processing version of the READING phase using OpenAI Batch API.
    
    Processes all articles in a single batch job for 50% cost savings.
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        structured_llm: StructuredLLMClient,
        vector_store: REMVectorStore,
        chunker: SentenceAwareChunker,
        api_key: str = None
    ):
        """
        Initialize the true batch reading cycle.
        
        Args:
            llm_client: LLM client for generation
            structured_llm: Structured LLM for entity extraction
            vector_store: Vector store for persistence
            chunker: Sentence-aware chunker
            api_key: OpenAI API key
        """
        self.llm = llm_client
        self.structured_llm = structured_llm
        self.vector_store = vector_store
        self.chunker = chunker
        self.entity_processor = EntityProcessor(structured_llm, vector_store)
        
        # Set up OpenAI client
        openai.api_key = api_key or OPENAI_API_KEY
        
        # Initialize writer service (will be started when needed)
        self.writer_service = None
    
    async def process_articles_batch(self, articles: List[Dict], existing_batch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple articles using OpenAI Batch API.
        
        Args:
            articles: List of article dicts with text, year, article_id, title
            existing_batch_id: Optional batch ID to resume from
            
        Returns:
            Processing statistics including batch_id for tracking
        """
        start_time = time.time()
        total_articles = len(articles)
        
        print(f"\n🚀 [{datetime.now().strftime('%H:%M:%S')}] Starting TRUE BATCH processing of {total_articles} articles")
        print("=" * 70)
        
        # Check if we should resume from existing batch
        if existing_batch_id:
            print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Checking existing batch: {existing_batch_id}")
            try:
                batch = openai.batches.retrieve(existing_batch_id)
                
                if batch.status == "completed":
                    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Batch already completed! Downloading results...")
                    results = await self._wait_for_batch_completion(existing_batch_id)
                    if results:
                        # Need to recreate batch_data for processing results
                        print(f"📄 [{datetime.now().strftime('%H:%M:%S')}] Preparing data for result processing...")
                        batch_data = self._prepare_batch_data(articles)
                        final_stats = self._process_batch_results(results, batch_data)
                        
                        elapsed = time.time() - start_time
                        return {
                            'batch_id': existing_batch_id,
                            'total_articles': total_articles,
                            'total_requests': len(results),
                            'processing_time': elapsed,
                            'resumed': True,
                            **final_stats
                        }
                elif batch.status in ["in_progress", "validating", "finalizing"]:
                    print(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] Batch is {batch.status}. Waiting for completion...")
                    print(f"   Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
                    
                    results = await self._wait_for_batch_completion(existing_batch_id)
                    if results:
                        # Need to recreate batch_data for processing results
                        print(f"📄 [{datetime.now().strftime('%H:%M:%S')}] Preparing data for result processing...")
                        batch_data = self._prepare_batch_data(articles)
                        final_stats = self._process_batch_results(results, batch_data)
                        
                        elapsed = time.time() - start_time
                        return {
                            'batch_id': existing_batch_id,
                            'total_articles': total_articles,
                            'total_requests': len(results),
                            'processing_time': elapsed,
                            'resumed': True,
                            **final_stats
                        }
                else:
                    print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] Batch is {batch.status}. Creating new batch...")
                    
            except Exception as e:
                print(f"⚠️  [{datetime.now().strftime('%H:%M:%S')}] Error checking batch: {e}")
                print("   Creating new batch...")
        
        # Stage 1: Prepare all data
        print(f"\n📄 [{datetime.now().strftime('%H:%M:%S')}] Stage 1: Preparing batch data...")
        batch_data = self._prepare_batch_data(articles)
        
        if not batch_data['requests']:
            print("⚠️  No requests to process")
            return {'error': 'No valid requests generated'}
        
        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Prepared {len(batch_data['requests'])} total requests")
        if batch_data['stats']:
            print(f"   - {batch_data['stats'].get('synthesis_prompts', 0)} synthesis prompts")
            print(f"   - {batch_data['stats'].get('entity_extraction_prompts', 0)} entity extraction prompts")
            print(f"   - {batch_data['stats'].get('summary_prompts', 0)} summary prompts")
        
        # Stage 2: Create and submit batch
        print(f"\n📤 [{datetime.now().strftime('%H:%M:%S')}] Stage 2: Submitting to OpenAI Batch API...")
        batch_id = await self._submit_batch(batch_data['requests'])
        
        if not batch_id:
            return {'error': 'Failed to submit batch'}
        
        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Batch submitted with ID: {batch_id}")
        print("💡 This will process at 50% reduced cost")
        
        # Stage 3: Wait for completion
        print(f"\n⏳ [{datetime.now().strftime('%H:%M:%S')}] Stage 3: Waiting for batch completion...")
        print("   (This can take up to 24 hours, but often completes much faster)")
        print("   (Checking every 30 seconds...)")
        
        results = await self._wait_for_batch_completion(batch_id)
        
        if not results:
            return {'error': 'Batch processing failed', 'batch_id': batch_id}
        
        # Stage 4: Process results
        print(f"\n💾 [{datetime.now().strftime('%H:%M:%S')}] Stage 4: Processing results and storing in database...")
        final_stats = self._process_batch_results(results, batch_data)
        
        elapsed = time.time() - start_time
        print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] Batch processing complete in {elapsed/60:.1f} minutes")
        print(f"📊 Processed {total_articles} articles with {len(results)} API calls")
        if final_stats.get('entities_extracted'):
            print(f"   - Extracted {final_stats.get('entities_extracted', 0)} entities")
            print(f"   - Stored {final_stats.get('learnings_stored', 0)} learnings")
        
        return {
            'batch_id': batch_id,
            'total_articles': total_articles,
            'total_requests': len(batch_data['requests']),
            'processing_time': elapsed,
            **final_stats
        }
    
    def _prepare_batch_data(self, articles: List[Dict]) -> Dict[str, Any]:
        """Prepare all batch requests from articles with efficient batching."""
        requests = []
        metadata_map = {}
        stats = defaultdict(int)
        
        # Phase 1: Chunk all articles first
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Phase 1a: Chunking all articles...")
        all_chunks_data = []
        
        with tqdm(total=len(articles), desc="  Chunking articles") as pbar:
            for article in articles:
                try:
                    chunks = self.chunker.chunk_article(article)
                    for i, chunk in enumerate(chunks):
                        chunk_data = {
                            'chunk': chunk,
                            'article': article,
                            'chunk_index': i,
                            'chunk_text': chunk["text"],
                            'chunk_metadata': self._prepare_chunk_metadata(chunk, article, i)
                        }
                        all_chunks_data.append(chunk_data)
                except Exception as e:
                    logger.error(f"Failed to chunk article {article['title']}: {e}")
                pbar.update(1)
        
        print(f"  ✅ [{datetime.now().strftime('%H:%M:%S')}] Created {len(all_chunks_data)} chunks from {len(articles)} articles")
        
        if not all_chunks_data:
            return {'requests': requests, 'metadata_map': metadata_map, 'stats': dict(stats)}
        
        # Phase 2: Batch generate embeddings for all chunks
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Phase 1b: Generating embeddings for all chunks...")
        chunk_texts = [cd['chunk_text'] for cd in all_chunks_data]
        chunk_embeddings = self._batch_generate_embeddings(chunk_texts)
        
        # Phase 3: Store all chunks in database
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Phase 1c: Storing chunks in database...")
        self._store_chunks_batch(all_chunks_data, chunk_embeddings)
        
        # Phase 4: Batch generate query embeddings and find similar content
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Phase 1d: Finding related knowledge for synthesis...")
        synthesis_candidates = self._find_synthesis_candidates(all_chunks_data)
        
        # Phase 5: Create synthesis and summary requests
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Phase 1e: Creating batch requests...")
        with tqdm(total=len(synthesis_candidates) + len(all_chunks_data) + len(articles), desc="  Creating requests") as pbar:
            # Create synthesis requests
            for candidate in synthesis_candidates:
                synthesis_req = self._create_synthesis_request(
                    candidate['chunk_id'],
                    candidate['chunk_text'],
                    candidate['chunk_metadata'],
                    candidate['existing_results'],
                    candidate['article_id']
                )
                if synthesis_req:
                    requests.append(synthesis_req)
                    metadata_map[synthesis_req.custom_id] = synthesis_req.metadata
                    stats['synthesis_prompts'] += 1
                pbar.update(1)
            
            # Create entity extraction requests for all chunks
            for chunk_data in all_chunks_data:
                # Generate chunk_id if not present
                chunk_id = chunk_data.get('chunk_id')
                if not chunk_id:
                    article = chunk_data['article']
                    chunk_id = f"{article['article_id']}_chunk_{chunk_data['chunk_index']}"
                
                entity_req = self._create_entity_extraction_request(
                    chunk_id,
                    chunk_data['chunk_text'],
                    chunk_data['chunk_metadata'],
                    chunk_data['article']['article_id']
                )
                if entity_req:
                    requests.append(entity_req)
                    metadata_map[entity_req.custom_id] = entity_req.metadata
                    stats['entity_extraction_prompts'] = stats.get('entity_extraction_prompts', 0) + 1
                pbar.update(1)
            
            # Create summary requests
            for article in articles:
                summary_req = self._create_summary_request(article)
                if summary_req:
                    requests.append(summary_req)
                    metadata_map[summary_req.custom_id] = summary_req.metadata
                    stats['summary_prompts'] += 1
                pbar.update(1)
        
        return {
            'requests': requests,
            'metadata_map': metadata_map,
            'stats': dict(stats)
        }
    
    def _prepare_chunk_metadata(self, chunk: Dict, article: Dict, chunk_index: int) -> Dict:
        """Prepare metadata for a chunk."""
        chunk_meta = chunk.get("metadata", {})
        
        metadata = {
            "year": article["year"],
            "article_id": article["article_id"],
            "title": article["title"],
            "chunk_index": chunk_index,
            "source_type": "chunk",
            "node_type": "chunk",
            "generation_depth": 0,
            "word_count": chunk_meta.get("word_count", 0),
            "chunker": chunk_meta.get("chunker", "sentence_aware"),
            **chunk_meta
        }
        
        # Clean metadata - ChromaDB cannot handle None values
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        batch_size = 100  # OpenAI recommends batches of 100 or less
        
        with tqdm(total=len(texts), desc="     Generating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    response = openai.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch_texts
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    # Add None embeddings for failed batch
                    embeddings.extend([None] * len(batch_texts))
                pbar.update(len(batch_texts))
        
        return embeddings
    
    def _store_chunks_batch(self, chunks_data: List[Dict], embeddings: List[List[float]]):
        """Store all chunks in database with pre-computed embeddings."""
        # Filter out chunks with failed embeddings
        valid_chunks = []
        valid_embeddings = []
        valid_metadatas = []
        
        for i, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings)):
            if embedding is not None:
                valid_chunks.append(chunk_data['chunk_text'])
                valid_embeddings.append(embedding)
                valid_metadatas.append(chunk_data['chunk_metadata'])
                # Store chunk_id back in chunk_data for later use
                chunk_data['chunk_id'] = f"chunk_{i}_{datetime.now().timestamp()}"
                # IMPORTANT: Store the embedding in chunk_data for reuse
                chunk_data['embedding'] = embedding
        
        # Store all valid chunks at once
        if valid_chunks:
            try:
                chunk_ids = [f"chunk_{i}_{datetime.now().timestamp()}" for i in range(len(valid_chunks))]
                result = self.vector_store.collection.add(
                    documents=valid_chunks,
                    embeddings=valid_embeddings,
                    metadatas=valid_metadatas,
                    ids=chunk_ids
                )
                # Update chunk_data with actual IDs
                chunk_idx = 0
                for chunk_data in chunks_data:
                    if chunk_data.get('chunk_id'):  # Only update those that were valid
                        chunk_data['chunk_id'] = chunk_ids[chunk_idx]
                        chunk_idx += 1
                print(f"     ✅ [{datetime.now().strftime('%H:%M:%S')}] Stored {len(valid_chunks)} chunks in database")
            except Exception as e:
                logger.error(f"Failed to store chunks: {e}")
    
    def _find_synthesis_candidates(self, chunks_data: List[Dict]) -> List[Dict]:
        """Find chunks that have related knowledge for synthesis."""
        candidates = []
        
        # No need to generate embeddings - reuse from chunk_data!
        print(f"     [{datetime.now().strftime('%H:%M:%S')}] Searching for related knowledge using existing embeddings...")
        
        # Perform similarity searches
        with tqdm(total=len(chunks_data), desc="     Finding syntheses") as pbar:
            for chunk_data in chunks_data:
                if 'chunk_id' not in chunk_data or 'embedding' not in chunk_data:
                    pbar.update(1)
                    continue
                
                try:
                    # Query using the embedding we already computed
                    year = chunk_data['article']['year']
                    results = self.vector_store.collection.query(
                        query_embeddings=[chunk_data['embedding']],
                        n_results=NEIGHBORS_COUNT,
                        where={"year": {"$lt": year}}
                    )
                    
                    # Convert results to expected format
                    existing_results = {
                        'documents': results['documents'][0] if results['documents'] else [],
                        'metadatas': results['metadatas'][0] if results['metadatas'] else []
                    }
                    
                    # If we found related knowledge, add as candidate
                    if existing_results['documents']:
                        candidates.append({
                            'chunk_id': chunk_data['chunk_id'],
                            'chunk_text': chunk_data['chunk_text'],
                            'chunk_metadata': chunk_data['chunk_metadata'],
                            'existing_results': existing_results,
                            'article_id': chunk_data['article']['article_id']
                        })
                except Exception as e:
                    logger.error(f"Failed to query for chunk: {e}")
                
                pbar.update(1)
        
        print(f"     ✅ [{datetime.now().strftime('%H:%M:%S')}] Found {len(candidates)} chunks with related knowledge")
        return candidates
    
    
    def _create_synthesis_request(
        self, 
        chunk_id: str,
        chunk_text: str,
        chunk_metadata: Dict,
        existing_results: Dict,
        article_id: str
    ) -> Optional[BatchRequest]:
        """Create a synthesis request for a chunk."""
        # Format existing knowledge
        formatted_chunks = []
        for i in range(min(NEIGHBORS_COUNT, len(existing_results["documents"]))):
            doc = existing_results["documents"][i]
            meta = existing_results["metadatas"][i] if i < len(existing_results["metadatas"]) else {}
            year = meta.get("year", "Unknown Year")
            formatted_chunk = f"{doc} (Year: {year})"
            formatted_chunks.append(formatted_chunk)
        
        existing_text = "\n\n---\n\n".join(formatted_chunks)
        
        # Create synthesis prompt
        prompt = IMPLANT_SYNTHESIS_PROMPT.format(
            new_info=chunk_text,
            existing_knowledge=existing_text
        )
        
        return BatchRequest(
            custom_id=f"fa_rem_rag_synthesis_{chunk_id}_{datetime.now().timestamp()}",
            prompt=prompt,
            temperature=0.7,
            max_tokens=500,
            metadata={
                'type': 'synthesis',
                'chunk_id': chunk_id,
                'chunk_metadata': chunk_metadata,
                'article_id': article_id
            }
        )
    
    def _create_summary_request(self, article: Dict) -> Optional[BatchRequest]:
        """Create a summary request for an article."""
        prompt = ARTICLE_SUMMARY_PROMPT.format(article_text=article["text"])
        
        return BatchRequest(
            custom_id=f"fa_rem_rag_summary_{article['article_id']}_{datetime.now().timestamp()}",
            prompt=prompt,
            max_tokens=300,
            metadata={
                'type': 'summary',
                'article': article
            }
        )
    
    def _create_entity_extraction_request(
        self, 
        chunk_id: str,
        chunk_text: str,
        chunk_metadata: Dict,
        article_id: str
    ) -> Optional[BatchRequest]:
        """Create an entity extraction request for a chunk."""
        # Create prompt for entity extraction
        prompt = f"{ENTITY_EXTRACTION_PROMPT}\n\nText: {chunk_text}"
        
        # We need to format this for standard completion since batch API doesn't support function calling
        # We'll parse the response as JSON in the results processing
        system_prompt = """Extract entities and what we learn about them. 
Return your response as a JSON array with objects containing 'entity' and 'learning' fields.
Example format:
[
  {"entity": "NATO", "learning": "NATO expanded eastward in 1999 despite Russian objections..."},
  {"entity": "Boris Yeltsin", "learning": "Yeltsin faced domestic opposition to NATO expansion..."}
]
If no entities with substantial learnings are found, return an empty array: []"""
        
        return BatchRequest(
            custom_id=f"fa_rem_rag_entities_{chunk_id}_{datetime.now().timestamp()}",
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
            metadata={
                'type': 'entity_extraction',
                'chunk_id': chunk_id,
                'chunk_metadata': chunk_metadata,
                'article_id': article_id
            }
        )
    
    async def _submit_batch(self, requests: List[BatchRequest]) -> Optional[str]:
        """Create JSONL file and submit batch to OpenAI."""
        # Create JSONL file
        batch_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        try:
            for req in requests:
                # Create OpenAI batch format
                batch_item = {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": req.prompt}
                        ],
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature
                    }
                }
                
                if req.system_prompt:
                    batch_item["body"]["messages"].insert(0, {
                        "role": "system",
                        "content": req.system_prompt
                    })
                
                batch_file.write(json.dumps(batch_item) + '\n')
            
            batch_file.close()
            
            # Upload file
            with open(batch_file.name, 'rb') as f:
                file_response = openai.files.create(
                    file=f,
                    purpose="batch"
                )
            
            # Create batch with metadata
            batch_response = openai.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "project": "fa_rem_rag",
                    "type": "reading_cycle",
                    "timestamp": datetime.now().isoformat(),
                    "articles_count": str(len(requests))
                }
            )
            
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            return None
        finally:
            # Clean up temp file
            Path(batch_file.name).unlink(missing_ok=True)
    
    async def _wait_for_batch_completion(self, batch_id: str, max_retries: int = 5) -> Optional[List[dict]]:
        """Wait for batch to complete and return results with retry logic."""
        start_time = time.time()
        consecutive_errors = 0
        check_count = 0
        
        # Backoff strategy: every minute for 5 minutes, then every 5 minutes until an hour, then every 10 minutes
        def get_check_interval(elapsed_minutes: float) -> int:
            if elapsed_minutes < 5:
                return 60  # Every minute for first 5 minutes
            elif elapsed_minutes < 60:
                return 300  # Every 5 minutes until 1 hour
            else:
                return 600  # Every 10 minutes thereafter
        
        with tqdm(total=100, desc="Batch progress", unit="%") as pbar:
            last_progress = 0
            last_status_print = 0
            
            while True:
                try:
                    batch = openai.batches.retrieve(batch_id)
                    consecutive_errors = 0  # Reset on successful request
                    check_count += 1
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    elapsed_minutes = elapsed_time / 60
                    
                    # Get appropriate check interval
                    check_interval = get_check_interval(elapsed_minutes)
                    
                    # Only print status occasionally (every 5 checks or every 10 minutes)
                    if check_count % 5 == 1 or (elapsed_time - last_status_print) > 600:
                        print(f"\n   [{datetime.now().strftime('%H:%M:%S')}] Batch status: {batch.status} (checking every {check_interval}s)")
                        last_status_print = elapsed_time
                    
                    if batch.status.lower() == "completed":
                        pbar.update(100 - last_progress)
                        print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] Batch completed! Downloading results...")
                        
                        # Download results with retry
                        for attempt in range(max_retries):
                            try:
                                file_response = openai.files.content(batch.output_file_id)
                                results = []
                                
                                # Handle different response formats
                                if hasattr(file_response, 'text'):
                                    content = file_response.text
                                elif hasattr(file_response, 'content'):
                                    content = file_response.content
                                    if isinstance(content, bytes):
                                        content = content.decode('utf-8')
                                else:
                                    # Try to read it as bytes
                                    content = file_response.read()
                                    if isinstance(content, bytes):
                                        content = content.decode('utf-8')
                                
                                for line in content.strip().split('\n'):
                                    if line:
                                        results.append(json.loads(line))
                                
                                return results
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 5
                                    logger.warning(f"Error downloading results (attempt {attempt + 1}/{max_retries}): {e}")
                                    print(f"⚠️  Error downloading results (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                                    print(f"    Retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    logger.error(f"Failed to download results after {max_retries} attempts: {e}")
                                    print(f"❌ Failed to download results after {max_retries} attempts")
                                    return None
                    
                    elif batch.status in ["failed", "expired", "cancelled"]:
                        print(f"\n❌ Batch {batch.status}: {batch.errors}")
                        return None
                    
                    else:
                        # Update progress
                        counts = batch.request_counts
                        total = counts.total if hasattr(counts, 'total') else 0
                        completed = counts.completed if hasattr(counts, 'completed') else 0
                        
                        if total > 0:
                            progress = int((completed / total) * 100)
                            if progress > last_progress:
                                pbar.update(progress - last_progress)
                                last_progress = progress
                        
                        await asyncio.sleep(check_interval)
                        
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error checking batch status (attempt {consecutive_errors}): {e}")
                    
                    # Check if it's a retryable error
                    error_str = str(e)
                    is_retryable = any(code in error_str for code in ['500', '502', '503', '504', 'server_error', 'timeout'])
                    
                    if is_retryable and consecutive_errors < max_retries:
                        # Exponential backoff
                        wait_time = min(300, check_interval * (2 ** (consecutive_errors - 1)))
                        print(f"\n⚠️  Server error (attempt {consecutive_errors}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"\n❌ Failed after {consecutive_errors} attempts: {e}")
                        return None
    
    def _process_batch_results(self, results: List[dict], batch_data: Dict) -> Dict:
        """Process batch results and store in database."""
        stats = defaultdict(int)
        metadata_map = batch_data['metadata_map']
        
        with tqdm(total=len(results), desc="Processing results") as pbar:
            for result in results:
                custom_id = result["custom_id"]
                
                if result["response"]["status_code"] != 200:
                    logger.error(f"Request failed: {custom_id}")
                    stats['failed_requests'] += 1
                    pbar.update(1)
                    continue
                
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                metadata = metadata_map.get(custom_id, {})
                
                if metadata.get('type') == 'synthesis':
                    # Store synthesis if valuable
                    if content and content.strip() != "NOTHING":
                        self._store_synthesis(content, metadata)
                        stats['syntheses_stored'] += 1
                    else:
                        stats['syntheses_skipped'] += 1
                        
                elif metadata.get('type') == 'summary':
                    # Store summary
                    if content:
                        self._store_summary(content, metadata)
                        stats['summaries_stored'] += 1
                
                elif metadata.get('type') == 'entity_extraction':
                    # Process extracted entities
                    entity_stats = self._process_entity_extraction_result(content, metadata)
                    stats['entities_extracted'] += entity_stats['total_entities']
                    stats['learnings_stored'] += entity_stats['learnings_stored']
                
                pbar.update(1)
        
        return dict(stats)
    
    def _store_synthesis(self, synthesis_text: str, metadata: Dict):
        """Store a synthesis in the database."""
        try:
            chunk_metadata = metadata['chunk_metadata'].copy()
            chunk_metadata.update({
                "node_type": "synthesis",
                "generation_depth": chunk_metadata.get("generation_depth", 0) + 1,
                "parent_ids": metadata['chunk_id'],
                "synthesis_type": "implant"
            })
            
            # Clean metadata
            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
            
            # Generate embedding once
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[synthesis_text]
            )
            embedding = response.data[0].embedding
            
            # Store synthesis with pre-computed embedding
            self.vector_store.add_with_embeddings([synthesis_text], [embedding], [chunk_metadata])
            
        except Exception as e:
            logger.error(f"Failed to store synthesis: {e}")
    
    def _store_summary(self, summary_text: str, metadata: Dict):
        """Store an article summary in the database."""
        try:
            article = metadata['article']
            
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
            
            # Generate embedding once
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[summary_text]
            )
            embedding = response.data[0].embedding
            
            # Store summary with pre-computed embedding
            self.vector_store.add_with_embeddings([summary_text], [embedding], [summary_metadata])
            
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
    
    def _process_entity_extraction_result(self, content: str, metadata: Dict) -> Dict[str, int]:
        """Process entity extraction result and store learnings with synthesis."""
        stats = {
            'total_entities': 0,
            'learnings_stored': 0
        }
        
        try:
            # Parse JSON response
            import json
            entities = json.loads(content)
            
            if not isinstance(entities, list):
                logger.error(f"Expected list of entities, got {type(entities)}")
                return stats
            
            stats['total_entities'] = len(entities)
            
            # Process each entity learning
            for entity_data in entities:
                if not isinstance(entity_data, dict) or 'entity' not in entity_data or 'learning' not in entity_data:
                    logger.warning(f"Invalid entity data format: {entity_data}")
                    continue
                
                entity_name = entity_data['entity']
                learning_text = entity_data['learning']
                
                # Prepare metadata for the learning
                chunk_metadata = metadata['chunk_metadata'].copy()
                learning_metadata = {
                    "entity": entity_name,
                    "year": chunk_metadata.get('year'),
                    "article_id": metadata['article_id'],
                    "source_type": "entity_extraction",
                    "node_type": "learning",
                    "generation_depth": 0,
                    "chunk_id": metadata['chunk_id'],
                    "title": chunk_metadata.get('title')
                }
                
                # Clean metadata
                learning_metadata = {k: v for k, v in learning_metadata.items() if v is not None}
                
                # Generate embedding for the learning
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=[learning_text]
                )
                learning_embedding = response.data[0].embedding
                
                # Store the learning
                self.vector_store.add_with_embeddings([learning_text], [learning_embedding], [learning_metadata])
                stats['learnings_stored'] += 1
                        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction JSON: {e}")
            logger.error(f"Content was: {content[:200]}...")
        except Exception as e:
            logger.error(f"Failed to process entity extraction: {e}")
            
        return stats
    
    async def _process_batch_results_async(self, results: List[dict], batch_data: Dict) -> Dict:
        """Process batch results asynchronously and store in database using writer service."""
        stats = defaultdict(int)
        metadata_map = batch_data['metadata_map']
        
        with tqdm(total=len(results), desc="Processing results") as pbar:
            for result in results:
                custom_id = result["custom_id"]
                
                if result["response"]["status_code"] != 200:
                    logger.error(f"Request failed: {custom_id}")
                    stats['failed_requests'] += 1
                    pbar.update(1)
                    continue
                
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                metadata = metadata_map.get(custom_id, {})
                
                if metadata.get('type') == 'synthesis':
                    # Store synthesis if valuable
                    if content and content.strip() != "NOTHING":
                        await self._store_synthesis_async(content, metadata)
                        stats['syntheses_stored'] += 1
                    else:
                        stats['syntheses_skipped'] += 1
                        
                elif metadata.get('type') == 'summary':
                    # Store summary
                    if content:
                        await self._store_summary_async(content, metadata)
                        stats['summaries_stored'] += 1
                
                elif metadata.get('type') == 'entity_extraction':
                    # Process extracted entities directly
                    entity_stats = await self._process_entity_extraction_async(content, metadata)
                    stats['entities_extracted'] += entity_stats['total_entities']
                    stats['learnings_stored'] += entity_stats['learnings_stored']
                
                pbar.update(1)
        
        return dict(stats)
    
    async def process_articles_async(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Process multiple articles using async requests instead of batch API.
        
        Args:
            articles: List of article dicts with text, year, article_id, title
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        total_articles = len(articles)
        
        print(f"\n🚀 [{datetime.now().strftime('%H:%M:%S')}] Starting ASYNC processing of {total_articles} articles")
        print("=" * 70)
        
        # Start the writer service
        self.writer_service = DatabaseWriterService(self.vector_store)
        await self.writer_service.start()
        
        try:
            # Prepare all data (same as batch)
            print(f"\n📄 [{datetime.now().strftime('%H:%M:%S')}] Stage 1: Preparing data...")
            batch_data = self._prepare_batch_data(articles)
            
            if not batch_data['requests']:
                print("⚠️  No requests to process")
                return {'error': 'No valid requests generated'}
            
            print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Prepared {len(batch_data['requests'])} total requests")
            
            # Process requests asynchronously
            print(f"\n⚡ [{datetime.now().strftime('%H:%M:%S')}] Stage 2: Processing requests asynchronously...")
            results = await self._process_requests_async(batch_data['requests'])
            
            # Process results (same as batch)
            print(f"\n💾 [{datetime.now().strftime('%H:%M:%S')}] Stage 3: Processing results and storing in database...")
            final_stats = await self._process_batch_results_async(results, batch_data)
            
            # Stop the writer service
            print(f"\n📝 [{datetime.now().strftime('%H:%M:%S')}] Finalizing database writes...")
            await self.writer_service.stop()
            
            elapsed = time.time() - start_time
            print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] Async processing complete in {elapsed/60:.1f} minutes")
            print(f"📊 Processed {total_articles} articles with {len(results)} API calls")
            
            # Get writer stats
            writer_stats = self.writer_service.get_stats()
            print(f"📊 Database writes: {writer_stats['items_written']} items in {writer_stats['batches_processed']} batches")
            
            return {
                'total_articles': total_articles,
                'total_requests': len(batch_data['requests']),
                'processing_time': elapsed,
                'database_writes': writer_stats['items_written'],
                **final_stats
            }
        finally:
            # Ensure writer service is stopped even on error
            if self.writer_service and self.writer_service.running:
                await self.writer_service.stop()
    
    async def _store_synthesis_async(self, synthesis_text: str, metadata: Dict):
        """Store a synthesis in the database using writer service."""
        try:
            chunk_metadata = metadata['chunk_metadata'].copy()
            chunk_metadata.update({
                "node_type": "synthesis",
                "generation_depth": chunk_metadata.get("generation_depth", 0) + 1,
                "parent_ids": metadata['chunk_id'],
                "synthesis_type": "implant"
            })
            
            # Clean metadata
            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
            
            # Generate embedding once
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[synthesis_text]
            )
            embedding = response.data[0].embedding
            
            # Add to writer queue
            await self.writer_service.add_write_job('synthesis', {
                'text': synthesis_text,
                'embedding': embedding,
                'metadata': chunk_metadata
            })
            
        except Exception as e:
            logger.error(f"Failed to store synthesis: {e}")
    
    async def _store_summary_async(self, summary_text: str, metadata: Dict):
        """Store an article summary in the database using writer service."""
        try:
            article = metadata['article']
            
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
            
            # Generate embedding once
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[summary_text]
            )
            embedding = response.data[0].embedding
            
            # Add to writer queue
            await self.writer_service.add_write_job('summary', {
                'text': summary_text,
                'embedding': embedding,
                'metadata': summary_metadata
            })
            
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
    
    async def _process_entity_extraction_async(self, content: str, metadata: Dict) -> Dict[str, int]:
        """Process entity extraction result asynchronously with async synthesis."""
        stats = {
            'total_entities': 0,
            'learnings_stored': 0
        }
        
        try:
            # Parse JSON response
            import json
            entities = json.loads(content)
            
            if not isinstance(entities, list):
                logger.error(f"Expected list of entities, got {type(entities)}")
                return stats
            
            stats['total_entities'] = len(entities)
            
            # Process each entity learning
            for entity_data in entities:
                if not isinstance(entity_data, dict) or 'entity' not in entity_data or 'learning' not in entity_data:
                    logger.warning(f"Invalid entity data format: {entity_data}")
                    continue
                
                entity_name = entity_data['entity']
                learning_text = entity_data['learning']
                
                # Prepare metadata for the learning
                chunk_metadata = metadata['chunk_metadata'].copy()
                learning_metadata = {
                    "entity": entity_name,
                    "year": chunk_metadata.get('year'),
                    "article_id": metadata['article_id'],
                    "source_type": "entity_extraction",
                    "node_type": "learning",
                    "generation_depth": 0,
                    "chunk_id": metadata['chunk_id'],
                    "title": chunk_metadata.get('title')
                }
                
                # Clean metadata
                learning_metadata = {k: v for k, v in learning_metadata.items() if v is not None}
                
                # Generate embedding for the learning
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=[learning_text]
                )
                learning_embedding = response.data[0].embedding
                
                # Add to writer queue
                await self.writer_service.add_write_job('learning', {
                    'text': learning_text,
                    'embedding': learning_embedding,
                    'metadata': learning_metadata
                })
                stats['learnings_stored'] += 1
                        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction JSON: {e}")
            logger.error(f"Content was: {content[:200]}...")
        except Exception as e:
            logger.error(f"Failed to process entity extraction: {e}")
            
        return stats
    
    
    async def _process_requests_async(self, requests: List[BatchRequest]) -> List[dict]:
        """Process all requests asynchronously with concurrency control."""
        results = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50  # Adjust based on rate limits
        
        with tqdm(total=len(requests), desc="Processing requests") as pbar:
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                batch_tasks = []
                
                for req in batch:
                    task = self._process_single_request_async(req)
                    batch_tasks.append(task)
                
                # Process batch concurrently
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results
    
    async def _process_single_request_async(self, request: BatchRequest) -> dict:
        """Process a single request asynchronously."""
        try:
            # Use the LLM client to generate response
            if request.system_prompt:
                response = await self.llm.generate(
                    request.prompt,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            else:
                response = await self.llm.generate(
                    request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            
            # Format as batch result for compatibility
            return {
                "custom_id": request.custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                "content": response
                            }
                        }]
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error processing request {request.custom_id}: {e}")
            return {
                "custom_id": request.custom_id,
                "response": {
                    "status_code": 500,
                    "body": {"error": str(e)}
                }
            }