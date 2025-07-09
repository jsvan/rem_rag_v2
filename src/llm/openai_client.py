"""OpenAI LLM client with async batch generation and cost tracking"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
import time
from datetime import datetime
import json
import os

import openai
from openai import AsyncOpenAI, OpenAI
import tiktoken

from ..config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async OpenAI client with batch processing, retry logic, and cost tracking.
    """
    
    # Pricing per 1M tokens (as of Dec 2023)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "text-embedding-3-small": {"input": 0.02}
    }
    
    def __init__(self, model: str = LLM_MODEL, max_retries: int = 3):
        """Initialize OpenAI clients."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.model = model
        self.max_retries = max_retries
        
        # Initialize clients
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _track_usage(self, usage: dict):
        """Track token usage and costs."""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
        
        # Calculate cost
        if self.model in self.PRICING:
            input_cost = (input_tokens / 1_000_000) * self.PRICING[self.model]["input"]
            output_cost = (output_tokens / 1_000_000) * self.PRICING[self.model]["output"]
            self.total_cost += input_cost + output_cost
    
    async def _generate_with_retry(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response with retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Track usage
                if response.usage:
                    self._track_usage(response.usage.model_dump())
                
                return response.choices[0].message.content
                
            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                logger.warning(f"Rate limit hit, waiting {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in generation (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a single response."""
        return await self._generate_with_retry(prompt, system_prompt, temperature, max_tokens)
    
    async def batch_generate(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 10
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts (strings) or dicts with 'prompt' and optional 'system_prompt'
            system_prompt: Default system prompt for all
            temperature: Generation temperature
            max_tokens: Max tokens per response
            max_concurrent: Max concurrent requests
            
        Returns:
            List of generated responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt_item):
            async with semaphore:
                if isinstance(prompt_item, dict):
                    prompt = prompt_item["prompt"]
                    sys_prompt = prompt_item.get("system_prompt", system_prompt)
                else:
                    prompt = prompt_item
                    sys_prompt = system_prompt
                
                return await self._generate_with_retry(
                    prompt, sys_prompt, temperature, max_tokens
                )
        
        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Synchronous generation for simple cases."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Track usage
        if response.usage:
            self._track_usage(response.usage.model_dump())
        
        return response.choices[0].message.content
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        avg_input = self.total_input_tokens / max(self.request_count, 1)
        avg_output = self.total_output_tokens / max(self.request_count, 1)
        
        return {
            "model": self.model,
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "average_input_tokens": avg_input,
            "average_output_tokens": avg_output,
            "total_cost_usd": round(self.total_cost, 4),
            "average_cost_per_request": round(self.total_cost / max(self.request_count, 1), 4)
        }
    
    def save_stats(self, filepath: str):
        """Save usage stats to file."""
        stats = self.get_stats()
        stats["timestamp"] = datetime.now().isoformat()
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved stats to {filepath}")


class StructuredLLMClient(LLMClient):
    """
    Extended client for structured outputs using function calling.
    """
    
    async def extract_entities(self, text: str, prompt: str) -> List[Dict[str, str]]:
        """
        Extract entities and learnings in structured format.
        
        Returns list of dicts with 'entity' and 'learning' keys.
        """
        messages = [
            {"role": "system", "content": "Extract entities and what we learn about them."},
            {"role": "user", "content": f"{prompt}\n\nText: {text}"}
        ]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities and learnings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "learning": {"type": "string"}
                                },
                                "required": ["entity", "learning"]
                            }
                        }
                    },
                    "required": ["entities"]
                }
            }
        }]
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_entities"}}
        )
        
        # Track usage
        if response.usage:
            self._track_usage(response.usage.model_dump())
        
        # Parse function call
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        
        return result["entities"]
    
    async def generate_question(self, texts: List[str], prompt: str) -> str:
        """
        Generate a question that connects multiple texts.
        
        Used in REM cycles to find patterns.
        """
        combined = "\n\n---\n\n".join(texts)
        full_prompt = f"{prompt}\n\nTexts:\n{combined}"
        
        return await self.generate(full_prompt, temperature=0.8)