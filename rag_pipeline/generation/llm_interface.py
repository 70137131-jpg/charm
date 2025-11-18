"""LLM interfaces for multiple providers"""
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMResponse:
    """Response from LLM"""
    text: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """Abstract interface for LLMs"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate with conversation history"""
        pass


class OpenAILLM(LLMInterface):
    """OpenAI LLM interface"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI LLM

        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.)
            api_key: OpenAI API key
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_history(messages, temperature, max_tokens, **kwargs)

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate with conversation history"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            }
        )

    def generate_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """Generate with streaming response"""
        messages = [{"role": "user", "content": prompt}]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLM(LLMInterface):
    """Anthropic Claude interface"""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None
    ):
        """
        Initialize Anthropic LLM

        Args:
            model: Model name (claude-3-opus, claude-3-sonnet, claude-3-haiku)
            api_key: Anthropic API key
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")

        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_history(messages, temperature, max_tokens, **kwargs)

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate with conversation history"""
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            text=response.content[0].text,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create(
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMInterface:
        """
        Create an LLM instance

        Args:
            provider: Provider name ('openai' or 'anthropic')
            model: Model name (optional, uses default if not provided)
            api_key: API key (optional, reads from env if not provided)
            **kwargs: Additional arguments

        Returns:
            LLMInterface instance
        """
        provider = provider.lower()

        if provider == "openai":
            default_model = "gpt-3.5-turbo"
            return OpenAILLM(model=model or default_model, api_key=api_key, **kwargs)
        elif provider == "anthropic":
            default_model = "claude-3-sonnet-20240229"
            return AnthropicLLM(model=model or default_model, api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


class SamplingStrategies:
    """Different sampling strategies for LLMs"""

    @staticmethod
    def greedy(llm: LLMInterface, prompt: str, **kwargs) -> str:
        """Greedy decoding (temperature=0)"""
        response = llm.generate(prompt, temperature=0.0, **kwargs)
        return response.text

    @staticmethod
    def sampling(
        llm: LLMInterface,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Sampling with temperature and nucleus sampling"""
        response = llm.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        return response.text

    @staticmethod
    def beam_search(
        llm: LLMInterface,
        prompt: str,
        n: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple candidates

        Note: This is a simplified version. Full beam search requires
        API support for multiple completions.
        """
        responses = []
        for i in range(n):
            # Vary temperature slightly for diversity
            temp = 0.5 + (i * 0.2)
            response = llm.generate(prompt, temperature=temp, **kwargs)
            responses.append(response.text)
        return responses

    @staticmethod
    def self_consistency(
        llm: LLMInterface,
        prompt: str,
        n: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Self-consistency: generate multiple responses and pick most common

        Useful for reasoning tasks
        """
        responses = []
        for _ in range(n):
            response = llm.generate(prompt, temperature=temperature, **kwargs)
            responses.append(response.text)

        # Simple majority voting (in practice, use more sophisticated methods)
        from collections import Counter
        counter = Counter(responses)
        most_common = counter.most_common(1)[0][0]

        return most_common
