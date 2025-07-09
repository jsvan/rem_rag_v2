"""LLM interfaces for text generation"""

from .openai_client import LLMClient, StructuredLLMClient

__all__ = ["LLMClient", "StructuredLLMClient"]