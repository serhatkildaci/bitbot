"""
BitBot LLM Integration
======================

Large Language Model integration using Ollama for local model deployment
and serving with streaming support and OpenAI-compatible API.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
import time

try:
    import httpx
    from openai import AsyncOpenAI
except ImportError:
    httpx = None
    AsyncOpenAI = None

from loguru import logger
from ..config.settings import LLMConfig
from ..stt.whisper_engine import TranscriptionResult


@dataclass
class LLMMessage:
    """Message for LLM conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    finish_reason: str
    model: str
    usage: Optional[Dict[str, int]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class OllamaClient:
    """Ollama client with OpenAI-compatible API."""
    
    def __init__(self, config: LLMConfig):
        if AsyncOpenAI is None:
            raise ImportError("openai not available. Install with: pip install openai")
        if httpx is None:
            raise ImportError("httpx not available. Install with: pip install httpx")
        
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self.conversation_history: List[LLMMessage] = []
        self.system_prompt = self._get_default_system_prompt()
        self._max_history_length = 20  # Keep last 20 messages
        
        logger.info(f"OllamaClient initialized for model: {config.model_name}")
    
    async def initialize(self) -> bool:
        """Initialize the Ollama client."""
        try:
            # Create OpenAI client pointing to Ollama
            self.client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key="ollama",  # Ollama doesn't require real API key
                http_client=httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0),
                    limits=httpx.Limits(max_connections=10)
                )
            )
            
            # Test connection by checking available models
            success = await self._test_connection()
            if success:
                logger.info("OllamaClient initialized successfully")
                # Initialize conversation with system prompt
                self.conversation_history = [
                    LLMMessage(role="system", content=self.system_prompt)
                ]
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize OllamaClient: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # Test with a simple API call to Ollama
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config.base_url}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    logger.info(f"Available models: {available_models}")
                    
                    # Check if our configured model is available
                    model_found = any(self.config.model_name in model for model in available_models)
                    if not model_found:
                        logger.warning(f"Model '{self.config.model_name}' not found. Available: {available_models}")
                        # Still return True - model might be auto-downloadable
                    
                    return True
                else:
                    logger.error(f"Ollama API returned status {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for BitBot."""
        return """You are BitBot, a fully local AI assistant focused on privacy, clarity, and utility.

Your core principles:
- Run entirely on the user's device — no cloud, no external data sharing
- Prioritize accurate, concise, and actionable responses
- Communicate in a natural, friendly, and professional tone
- Be transparent: if you're uncertain about something, say so clearly
- Never guess, never make up facts
- Avoid unnecessary chatter; be helpful without being verbose
- Never collect, store, or transmit personal data

Always prioritize user trust, clarity, and usefulness. Reply naturally, stay grounded, and only elaborate when asked."""

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        # Update the first message if it exists and is a system message
        if self.conversation_history and self.conversation_history[0].role == "system":
            self.conversation_history[0].content = prompt
        else:
            # Insert system message at the beginning
            self.conversation_history.insert(0, LLMMessage(role="system", content=prompt))
        logger.info("System prompt updated")
    
    def add_user_message(self, content: str) -> LLMMessage:
        """Add user message to conversation."""
        message = LLMMessage(role="user", content=content)
        self.conversation_history.append(message)
        self._trim_conversation_history()
        return message
    
    def add_assistant_message(self, content: str) -> LLMMessage:
        """Add assistant message to conversation."""
        message = LLMMessage(role="assistant", content=content)
        self.conversation_history.append(message)
        self._trim_conversation_history()
        return message
    
    def _trim_conversation_history(self):
        """Keep conversation history within reasonable limits."""
        # Always keep system prompt (first message)
        if len(self.conversation_history) > self._max_history_length:
            system_msg = self.conversation_history[0] if self.conversation_history[0].role == "system" else None
            # Keep most recent messages
            recent_messages = self.conversation_history[-self._max_history_length+1:]
            
            if system_msg:
                self.conversation_history = [system_msg] + recent_messages
            else:
                self.conversation_history = recent_messages
    
    def _format_messages_for_api(self) -> List[Dict[str, str]]:
        """Format conversation history for OpenAI API."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]
    
    async def generate_response(self, user_input: str) -> Optional[LLMResponse]:
        """Generate response to user input."""
        try:
            # Add user message
            self.add_user_message(user_input)
            
            # Use Ollama's native API
            import httpx
            
            # Prepare prompt from conversation history
            prompt = self._build_prompt_from_history()
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                },
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("response", "").strip()
                    
                    if content:
                        # Add assistant response to history
                        self.add_assistant_message(content)
                        
                        return LLMResponse(
                            content=content,
                            finish_reason="stop",
                            model=self.config.model_name,
                            usage=None
                        )
                    else:
                        logger.warning("Empty response from model")
                        return None
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def _build_prompt_from_history(self) -> str:
        """Build a prompt string from conversation history."""
        prompt_parts = []
        
        for message in self.conversation_history:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")  # Prompt for next response
        
        return "\n\n".join(prompt_parts)
    
    async def generate_streaming_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """Generate streaming response to user input."""
        if not self.client:
            logger.error("Client not initialized")
            return
        
        try:
            # Add user message
            self.add_user_message(user_input)
            
            # Prepare messages for API
            messages = self._format_messages_for_api()
            
            # Generate streaming response
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Add complete response to history
            if full_response.strip():
                self.add_assistant_message(full_response)
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
    
    async def process_transcription(self, transcription: TranscriptionResult) -> Optional[LLMResponse]:
        """Process transcription result and generate response."""
        if not transcription.text.strip():
            return None
        
        logger.info(f"Processing transcription: '{transcription.text}'")
        return await self.generate_response(transcription.text)
    
    def clear_conversation(self):
        """Clear conversation history (keeping system prompt)."""
        system_msg = None
        if self.conversation_history and self.conversation_history[0].role == "system":
            system_msg = self.conversation_history[0]
        
        self.conversation_history = [system_msg] if system_msg else []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        return {
            "message_count": len(self.conversation_history),
            "last_message_time": self.conversation_history[-1].timestamp if self.conversation_history else 0,
            "model": self.config.model_name,
            "system_prompt_set": any(msg.role == "system" for msg in self.conversation_history)
        }
    
    async def cleanup(self):
        """Clean up client resources."""
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()
        self.client = None
        self.conversation_history = []
        logger.info("OllamaClient cleaned up")


class LLMEngine:
    """High-level LLM engine wrapper."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client: Optional[OllamaClient] = None
        self._response_callbacks: List[Callable[[LLMResponse], None]] = []
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the LLM engine."""
        try:
            self.client = OllamaClient(self.config)
            success = await self.client.initialize()
            if success:
                self._is_initialized = True
                logger.info("LLMEngine initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize LLMEngine: {e}")
            return False
    
    def add_response_callback(self, callback: Callable[[LLMResponse], None]):
        """Add callback for LLM responses."""
        self._response_callbacks.append(callback)
    
    async def process_user_input(self, user_input: str) -> Optional[LLMResponse]:
        """Process user input and generate response."""
        if not self.client:
            logger.error("LLM client not initialized")
            return None
        
        response = await self.client.generate_response(user_input)
        if response:
            # Call all response callbacks
            for callback in self._response_callbacks:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(f"Error in LLM response callback: {e}")
        
        return response
    
    async def process_transcription(self, transcription: TranscriptionResult) -> Optional[LLMResponse]:
        """Process speech transcription and generate response."""
        if not self.client:
            logger.error("LLM client not initialized")
            return None
        
        response = await self.client.process_transcription(transcription)
        if response:
            # Call all response callbacks
            for callback in self._response_callbacks:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(f"Error in LLM response callback: {e}")
        
        return response
    
    async def stream_response(self, user_input: str, callback: Callable[[str], None]):
        """Generate streaming response with callback for each token."""
        if not self.client:
            logger.error("LLM client not initialized")
            return
        
        async for token in self.client.generate_streaming_response(user_input):
            callback(token)
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt for the LLM."""
        if self.client:
            self.client.set_system_prompt(prompt)
    
    def clear_conversation(self):
        """Clear conversation history."""
        if self.client:
            self.client.clear_conversation()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        if self.client:
            return self.client.get_conversation_summary()
        return {}
    
    @property
    def is_ready(self) -> bool:
        """Check if LLM engine is ready."""
        return self._is_initialized and self.client is not None
    
    async def cleanup(self):
        """Clean up LLM engine."""
        if self.client:
            await self.client.cleanup()
            self.client = None
        self._is_initialized = False
        logger.info("LLMEngine cleaned up") 