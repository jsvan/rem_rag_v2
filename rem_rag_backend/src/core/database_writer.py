"""
Database Writer Service - Manages all database writes through a single async queue.

This service ensures that only one thread writes to the database at a time,
preventing SQLite deadlocks while maintaining high performance through batching.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WriteJob:
    """Represents a single write job to be processed."""
    job_type: str  # 'chunk', 'learning', 'synthesis', 'summary', 'batch'
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class DatabaseWriterService:
    """
    Async database writer service that processes all writes through a single queue.
    
    This ensures no database deadlocks by serializing all write operations while
    maintaining performance through intelligent batching.
    """
    
    def __init__(self, vector_store, batch_size: int = 50, batch_timeout: float = 1.0):
        """
        Initialize the database writer service.
        
        Args:
            vector_store: The vector store instance to write to
            batch_size: Maximum number of items to batch together
            batch_timeout: Maximum time to wait for a batch to fill (seconds)
        """
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.write_queue = asyncio.Queue()
        self.running = False
        self.writer_task = None
        
        # Statistics
        self.stats = defaultdict(int)
        self.start_time = None
        
    async def start(self):
        """Start the writer service in the background."""
        if self.running:
            logger.warning("Writer service already running")
            return
            
        self.running = True
        self.start_time = time.time()
        self.writer_task = asyncio.create_task(self._writer_loop())
        logger.info("Database writer service started")
        
    async def stop(self):
        """Gracefully shutdown the writer service."""
        if not self.running:
            return
            
        logger.info("Shutting down database writer service...")
        self.running = False
        
        # Process any remaining items
        await self.write_queue.join()
        
        # Cancel the writer task
        if self.writer_task:
            self.writer_task.cancel()
            try:
                await self.writer_task
            except asyncio.CancelledError:
                pass
                
        # Log final statistics
        self._log_stats()
        logger.info("Database writer service stopped")
        
    async def add_write_job(self, job_type: str, data: Dict[str, Any]):
        """
        Add a write job to the queue.
        
        Args:
            job_type: Type of write operation ('chunk', 'learning', 'synthesis', 'summary', 'batch')
            data: Data to write, format depends on job_type:
                - chunk/learning/synthesis/summary: {texts, embeddings, metadata}
                - batch: {items: [{job_type, texts, embeddings, metadata}, ...]}
        """
        job = WriteJob(job_type=job_type, data=data)
        await self.write_queue.put(job)
        self.stats['jobs_queued'] += 1
        
    async def _writer_loop(self):
        """Main writer loop that processes jobs from the queue."""
        batch = []
        last_batch_time = time.time()
        
        try:
            while self.running or not self.write_queue.empty():
                try:
                    # Try to get an item with timeout
                    timeout = max(0.1, self.batch_timeout - (time.time() - last_batch_time))
                    job = await asyncio.wait_for(
                        self.write_queue.get(), 
                        timeout=timeout
                    )
                    batch.append(job)
                    self.write_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Timeout reached, process batch if any
                    pass
                    
                # Check if we should process the batch
                should_process = (
                    len(batch) >= self.batch_size or
                    (time.time() - last_batch_time) >= self.batch_timeout or
                    (not self.running and batch)  # Process remaining on shutdown
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                    
        except asyncio.CancelledError:
            # Process any remaining items before cancellation
            if batch:
                await self._process_batch(batch)
            raise
            
    async def _process_batch(self, batch: List[WriteJob]):
        """Process a batch of write jobs."""
        start_time = time.time()
        
        # Group jobs by type for efficient processing
        jobs_by_type = defaultdict(list)
        for job in batch:
            jobs_by_type[job.job_type].append(job)
            
        total_written = 0
        
        try:
            # Process each job type
            for job_type, jobs in jobs_by_type.items():
                if job_type == 'batch':
                    # Handle batch jobs (which contain multiple sub-jobs)
                    for job in jobs:
                        count = await self._process_batch_job(job.data)
                        total_written += count
                else:
                    # Combine similar jobs for batch writing
                    count = await self._process_combined_jobs(job_type, jobs)
                    total_written += count
                    
            # Update statistics
            self.stats['batches_processed'] += 1
            self.stats['items_written'] += total_written
            self.stats['total_write_time'] += time.time() - start_time
            
            # Log progress occasionally
            if self.stats['batches_processed'] % 10 == 0:
                self._log_stats()
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats['write_errors'] += 1
            # Re-raise to handle retries at a higher level if needed
            raise
            
    async def _process_combined_jobs(self, job_type: str, jobs: List[WriteJob]) -> int:
        """Combine and process multiple jobs of the same type."""
        # Collect all data
        all_texts = []
        all_embeddings = []
        all_metadata = []
        
        for job in jobs:
            data = job.data
            if 'texts' in data:
                all_texts.extend(data['texts'])
                all_embeddings.extend(data['embeddings'])
                all_metadata.extend(data['metadata'])
            else:
                # Single item
                all_texts.append(data['text'])
                all_embeddings.append(data['embedding'])
                all_metadata.append(data['metadata'])
                
        # Write to database in one operation
        if all_texts:
            self.vector_store.add_with_embeddings(all_texts, all_embeddings, all_metadata)
            return len(all_texts)
            
        return 0
        
    async def _process_batch_job(self, data: Dict[str, Any]) -> int:
        """Process a batch job containing multiple sub-jobs."""
        items = data.get('items', [])
        
        # Group by type
        by_type = defaultdict(lambda: {'texts': [], 'embeddings': [], 'metadata': []})
        
        for item in items:
            job_type = item['job_type']
            by_type[job_type]['texts'].append(item['text'])
            by_type[job_type]['embeddings'].append(item['embedding'])
            by_type[job_type]['metadata'].append(item['metadata'])
            
        # Write each type
        total_written = 0
        for job_type, data in by_type.items():
            if data['texts']:
                self.vector_store.add_with_embeddings(
                    data['texts'], 
                    data['embeddings'], 
                    data['metadata']
                )
                total_written += len(data['texts'])
                
        return total_written
        
    def _log_stats(self):
        """Log current statistics."""
        if not self.start_time:
            return
            
        elapsed = time.time() - self.start_time
        avg_write_time = (self.stats['total_write_time'] / max(1, self.stats['batches_processed'])) * 1000
        items_per_second = self.stats['items_written'] / max(1, elapsed)
        
        logger.info(
            f"Writer stats: {self.stats['items_written']} items written in "
            f"{self.stats['batches_processed']} batches | "
            f"Avg write time: {avg_write_time:.1f}ms | "
            f"Rate: {items_per_second:.1f} items/s | "
            f"Queue size: {self.write_queue.qsize()}"
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = dict(self.stats)
        if self.start_time:
            stats['uptime_seconds'] = time.time() - self.start_time
        stats['queue_size'] = self.write_queue.qsize()
        return stats