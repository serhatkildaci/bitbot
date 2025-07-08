"""
BitBot Text-to-Speech Engine
============================

Text-to-speech synthesis using RealtimeTTS with support for multiple
backends (Piper TTS, Kyutai TTS) for low-latency speech generation.
"""

import asyncio
import threading
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time

try:
    from RealtimeTTS import TextToAudioStream, CoquiEngine, PiperEngine, OpenAIEngine
    import sounddevice as sd
    import numpy as np
except ImportError:
    TextToAudioStream = None
    CoquiEngine = None
    PiperEngine = None
    OpenAIEngine = None
    sd = None
    np = None

from loguru import logger
from ..config.settings import TTSConfig, AudioConfig
from ..llm.ollama_client import LLMResponse


@dataclass
class TTSRequest:
    """Text-to-speech synthesis request."""
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    priority: int = 1  # 1=high, 2=normal, 3=low
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class RealtimeTTSEngine:
    """RealtimeTTS engine with multiple backend support."""
    
    def __init__(self, config: TTSConfig, audio_config: AudioConfig):
        if TextToAudioStream is None:
            raise ImportError("RealtimeTTS not available. Install with: pip install RealtimeTTS")
        
        self.config = config
        self.audio_config = audio_config
        self.stream: Optional[TextToAudioStream] = None
        self.engine = None
        self._is_speaking = False
        self._speech_lock = threading.Lock()
        self._request_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        
        logger.info(f"RealtimeTTSEngine initialized with backend: {config.engine}")
    
    async def initialize(self) -> bool:
        """Initialize the TTS engine and stream."""
        try:
            # Initialize the appropriate engine based on config
            if self.config.engine.lower() == "piper":
                self.engine = await self._initialize_piper_engine()
            elif self.config.engine.lower() == "kyutai":
                self.engine = await self._initialize_kyutai_engine()
            else:
                logger.error(f"Unsupported TTS engine: {self.config.engine}")
                return False
            
            if not self.engine:
                return False
            
            # Create the audio stream
            self.stream = TextToAudioStream(
                self.engine,
                output_device_index=self.audio_config.output_device,
                blocksize=self.audio_config.chunk_size,
                buffer_threshold_seconds=0.1,  # Low latency threshold
                play_chunk_length=1024,
                overlap_wave_length=1024,
                ensure_chunk_size=True
            )
            
            logger.info("RealtimeTTS stream initialized successfully")
            
            # Start request processing task
            self._processing_task = asyncio.create_task(self._process_requests())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RealtimeTTS engine: {e}")
            return False
    
    async def _initialize_piper_engine(self):
        """Initialize Piper TTS engine."""
        try:
            if PiperEngine is None:
                logger.error("PiperEngine not available")
                return None
            
            # Run blocking initialization in executor
            loop = asyncio.get_event_loop()
            engine = await loop.run_in_executor(
                None,
                lambda: PiperEngine(
                    voice=self.config.voice or "en_US-lessac-medium",
                    speed=self.config.speed,
                    volume=1.0
                )
            )
            
            logger.info(f"Piper engine initialized with voice: {self.config.voice}")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper engine: {e}")
            return None
    
    async def _initialize_kyutai_engine(self):
        """Initialize Kyutai TTS engine (Coqui-based)."""
        try:
            if CoquiEngine is None:
                logger.error("CoquiEngine not available for Kyutai")
                return None
            
            # Run blocking initialization in executor
            loop = asyncio.get_event_loop()
            engine = await loop.run_in_executor(
                None,
                lambda: CoquiEngine(
                    voice=self.config.voice or "p335",
                    speed=self.config.speed,
                    local_models_path="./models/tts"  # Local model storage
                )
            )
            
            logger.info(f"Kyutai (Coqui) engine initialized with voice: {self.config.voice}")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to initialize Kyutai engine: {e}")
            return None
    
    async def speak_text(self, text: str, priority: int = 1) -> bool:
        """Queue text for speech synthesis."""
        if not self.stream:
            logger.error("TTS stream not initialized")
            return False
        
        request = TTSRequest(
            text=text,
            voice=self.config.voice,
            speed=self.config.speed,
            pitch=self.config.pitch,
            priority=priority
        )
        
        try:
            await self._request_queue.put(request)
            logger.debug(f"Queued TTS request: '{text[:50]}...'")
            return True
        except Exception as e:
            logger.error(f"Failed to queue TTS request: {e}")
            return False
    
    async def speak_text_immediate(self, text: str) -> bool:
        """Speak text immediately, interrupting current speech."""
        if not self.stream:
            logger.error("TTS stream not initialized")
            return False
        
        try:
            # Stop current speech
            await self.stop_speaking()
            
            # Clear queue
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Speak immediately
            await self.speak_text(text, priority=1)
            return True
            
        except Exception as e:
            logger.error(f"Failed to speak text immediately: {e}")
            return False
    
    async def _process_requests(self):
        """Process TTS requests from the queue."""
        while True:
            try:
                # Get next request
                request = await self._request_queue.get()
                
                # Process the request
                await self._synthesize_speech(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing TTS request: {e}")
                await asyncio.sleep(0.1)
    
    async def _synthesize_speech(self, request: TTSRequest):
        """Synthesize speech for a request."""
        with self._speech_lock:
            if self._is_speaking:
                # Wait for current speech to finish or timeout
                for _ in range(50):  # 5 second timeout
                    if not self._is_speaking:
                        break
                    await asyncio.sleep(0.1)
        
        try:
            self._is_speaking = True
            
            # Clean the text
            clean_text = self._clean_text(request.text)
            if not clean_text.strip():
                return
            
            logger.info(f"Synthesizing speech: '{clean_text[:100]}...'")
            
            # Run speech synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._play_speech_blocking,
                clean_text
            )
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
        finally:
            self._is_speaking = False
    
    def _play_speech_blocking(self, text: str):
        """Play speech using RealtimeTTS (blocking operation)."""
        try:
            # Feed text to the stream and play
            self.stream.feed(text)
            self.stream.play_async(fast_sentence_fragment=True)
            
            # Wait for playback to complete
            while self.stream.is_playing():
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in speech playback: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better TTS synthesis."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove or replace problematic characters
        text = text.replace("*", "")  # Remove asterisks
        text = text.replace("#", "number")  # Replace hash symbols
        text = text.replace("@", "at")  # Replace @ symbols
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += "."
        
        return text
    
    async def stop_speaking(self):
        """Stop current speech synthesis."""
        if self.stream and self._is_speaking:
            try:
                # Stop the stream
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.stream.stop
                )
                self._is_speaking = False
                logger.info("Speech synthesis stopped")
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    async def set_voice(self, voice: str):
        """Change the TTS voice."""
        self.config.voice = voice
        logger.info(f"TTS voice changed to: {voice}")
        # Note: May require engine reinitialization for some backends
    
    async def set_speed(self, speed: float):
        """Change the speech speed."""
        self.config.speed = max(0.5, min(2.0, speed))  # Clamp between 0.5x and 2.0x
        logger.info(f"TTS speed changed to: {self.config.speed}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices for the current engine."""
        if self.config.engine.lower() == "piper":
            # Return common Piper voices
            return [
                "en_US-lessac-medium",
                "en_US-lessac-high", 
                "en_US-amy-medium",
                "en_US-danny-low",
                "en_US-kathleen-low"
            ]
        elif self.config.engine.lower() == "kyutai":
            # Return common Coqui voices
            return [
                "p335", "p336", "p339", "p340", "p341",
                "default", "premium"
            ]
        else:
            return []
    
    async def cleanup(self):
        """Clean up TTS engine resources."""
        # Stop processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        
        # Stop current speech
        await self.stop_speaking()
        
        # Clean up stream
        if self.stream:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.stream.stop)
            except Exception as e:
                logger.warning(f"Error stopping TTS stream: {e}")
            self.stream = None
        
        self.engine = None
        logger.info("RealtimeTTS engine cleaned up")


class TTSEngine:
    """High-level TTS engine wrapper."""
    
    def __init__(self, config: TTSConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.engine: Optional[RealtimeTTSEngine] = None
        self._synthesis_callbacks: List[Callable[[str], None]] = []
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            self.engine = RealtimeTTSEngine(self.config, self.audio_config)
            success = await self.engine.initialize()
            if success:
                self._is_initialized = True
                logger.info("TTSEngine initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize TTSEngine: {e}")
            return False
    
    def add_synthesis_callback(self, callback: Callable[[str], None]):
        """Add callback for speech synthesis events."""
        self._synthesis_callbacks.append(callback)
    
    async def speak(self, text: str, interrupt: bool = False) -> bool:
        """Speak the given text."""
        if not self.engine:
            logger.error("TTS engine not initialized")
            return False
        
        # Call synthesis callbacks
        for callback in self._synthesis_callbacks:
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Error in synthesis callback: {e}")
        
        if interrupt:
            return await self.engine.speak_text_immediate(text)
        else:
            return await self.engine.speak_text(text)
    
    async def speak_response(self, llm_response: LLMResponse, interrupt: bool = True) -> bool:
        """Speak an LLM response."""
        if not llm_response.content.strip():
            return False
        
        logger.info(f"Speaking LLM response: '{llm_response.content[:100]}...'")
        return await self.speak(llm_response.content, interrupt=interrupt)
    
    async def stop(self):
        """Stop current speech."""
        if self.engine:
            await self.engine.stop_speaking()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.engine.is_speaking() if self.engine else False
    
    async def set_voice(self, voice: str):
        """Change TTS voice."""
        if self.engine:
            await self.engine.set_voice(voice)
    
    async def set_speed(self, speed: float):
        """Change speech speed."""
        if self.engine:
            await self.engine.set_speed(speed)
    
    def get_available_voices(self) -> List[str]:
        """Get available voices."""
        return self.engine.get_available_voices() if self.engine else []
    
    @property
    def is_ready(self) -> bool:
        """Check if TTS engine is ready."""
        return self._is_initialized and self.engine is not None
    
    async def cleanup(self):
        """Clean up TTS engine."""
        if self.engine:
            await self.engine.cleanup()
            self.engine = None
        self._is_initialized = False
        logger.info("TTSEngine cleaned up") 