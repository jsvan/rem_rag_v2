"""
OpenAI Batch Processor for efficient API calls.

This module provides utilities for batching OpenAI API calls while keeping
database operations serial to avoid locking issues.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
import time
from tqdm.asyncio import tqdm_asyncio

from .openai_client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIBatchProcessor:
    """
    Handles batch processing of OpenAI API calls.
    
    Key features:
    - Concurrent API calls with rate limiting
    - Error handling and retries
    - Progress tracking
    - Result aggregation
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        max_concurrent: int = 50,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the batch processor.
        
        Args:
            llm_client: LLM client instance
            max_concurrent: Maximum concurrent API calls
            retry_attempts: Number of retry attempts for failed calls
            retry_delay: Delay between retries in seconds
        """
        self.llm = llm_client
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def batch_generate(
        self,
        prompts: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts concurrently.
        
        Args:
            prompts: List of prompt configurations, each containing:
                - prompt: The prompt text
                - system_prompt: Optional system prompt
                - temperature: Optional temperature
                - max_tokens: Optional max tokens
                - metadata: Optional metadata to pass through
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results in same order as prompts, each containing:
                - text: Generated text or None if failed
                - error: Error message if failed
                - metadata: Pass-through metadata
        """
        logger.info(f"Starting batch generation for {len(prompts)} prompts")
        start_time = time.time()
        
        # Create tasks
        tasks = []
        for i, prompt_config in enumerate(prompts):
            task = self._process_single_prompt(i, prompt_config)
            tasks.append(task)
        
        # Process with progress tracking
        results = []
        
        # Use tqdm for progress tracking
        async for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="     API calls"):
            result = await coro
            results.append(result)
        
        # Sort results back to original order
        results.sort(key=lambda x: x['index'])
        
        elapsed = time.time() - start_time
        logger.info(
            f"Batch generation complete: {len(prompts)} prompts in {elapsed:.1f}s "
            f"({elapsed/len(prompts):.2f}s per prompt average)"
        )
        
        return results
    
    async def _process_single_prompt(
        self,
        index: int,
        prompt_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single prompt with retries and error handling."""
        async with self.semaphore:
            for attempt in range(self.retry_attempts):
                try:
                    # Extract configuration
                    prompt = prompt_config['prompt']
                    system_prompt = prompt_config.get('system_prompt')
                    temperature = prompt_config.get('temperature', 0.7)
                    max_tokens = prompt_config.get('max_tokens', 500)
                    
                    # Generate response
                    response = await self.llm.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return {
                        'index': index,
                        'text': response,
                        'error': None,
                        'metadata': prompt_config.get('metadata', {})
                    }
                    
                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    
                    logger.error(f"Failed to process prompt {index} after {self.retry_attempts} attempts: {e}")
                    return {
                        'index': index,
                        'text': None,
                        'error': str(e),
                        'metadata': prompt_config.get('metadata', {})
                    }
    
    async def batch_generate_simple(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> List[str]:
        """
        Simplified batch generation for list of simple prompts.
        
        Args:
            prompts: List of prompt strings
            system_prompt: Optional system prompt for all
            temperature: Temperature for all
            max_tokens: Max tokens for all
            
        Returns:
            List of generated texts (None for failures)
        """
        # Convert to full prompt configs
        prompt_configs = [
            {
                'prompt': prompt,
                'system_prompt': system_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            for prompt in prompts
        ]
        
        # Process
        results = await self.batch_generate(prompt_configs)
        
        # Extract just the text
        return [r['text'] for r in results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'max_concurrent': self.max_concurrent,
            'active_requests': self.max_concurrent - self.semaphore._value,
            'retry_attempts': self.retry_attempts
        }