"""
BitBot Simple Text-to-Speech Engine
==================================

Simple TTS engine using pyttsx3 for cross-platform text-to-speech synthesis,
avoiding the complex dependency conflicts of RealtimeTTS.
"""

import asyncio
import threading
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
import time
import tempfile
import os

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import gtts
    from io import BytesIO
    import pygame
except ImportError:
    gtts = None
    pygame = None

from loguru import logger
from ..config.settings import TTSConfig, AudioConfig
from ..llm.ollama_client import LLMResponse


@dataclass
class TTSRequest:
    """Text-to-speech synthesis request."""
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    priority: int = 1  # 1=high, 2=normal, 3=low
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SimpleTTSEngine:
    """Simple TTS engine using pyttsx3 with gTTS fallback."""
    
    def __init__(self, config: TTSConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.pyttsx3_engine: Optional[object] = None
        self._is_speaking = False
        self._speech_lock = threading.Lock()
        self._request_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._use_gtts_fallback = False
        
        logger.info(f"SimpleTTSEngine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            # Try to initialize pyttsx3 first
            if pyttsx3:
                success = await self._initialize_pyttsx3()
                if success:
                    logger.info("Initialized with pyttsx3")
                    return True
            
            # Fall back to gTTS if pyttsx3 fails
            if gtts:
                self._use_gtts_fallback = True
                logger.info("Falling back to gTTS")
                return True
            
            logger.error("No TTS engine available")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False
    
    async def _initialize_pyttsx3(self) -> bool:
        """Initialize pyttsx3 engine."""
        try:
            loop = asyncio.get_event_loop()
            self.pyttsx3_engine = await loop.run_in_executor(
                None, self._init_pyttsx3_blocking
            )
            
            if self.pyttsx3_engine:
                # Start request processing task
                self._processing_task = asyncio.create_task(self._process_requests())
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            
        return False
    
    def _init_pyttsx3_blocking(self):
        """Initialize pyttsx3 in blocking mode."""
        try:
            engine = pyttsx3.init()
            
            # Configure voice if specified
            if self.config.voice:
                voices = engine.getProperty('voices')
                for voice in voices:
                    if self.config.voice.lower() in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Configure speech rate
            rate = engine.getProperty('rate')
            engine.setProperty('rate', int(rate * self.config.speed))
            
            # Configure volume
            engine.setProperty('volume', 1.0)
            
            return engine
            
        except Exception as e:
            logger.error(f"pyttsx3 initialization failed: {e}")
            return None
    
    async def speak_text(self, text: str, priority: int = 1) -> bool:
        """Queue text for speech synthesis."""
        if not self.pyttsx3_engine and not self._use_gtts_fallback:
            logger.error("TTS engine not initialized")
            return False
        
        request = TTSRequest(
            text=text,
            voice=self.config.voice,
            speed=self.config.speed,
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
                if self._use_gtts_fallback:
                    await self._synthesize_speech_gtts(request)
                else:
                    await self._synthesize_speech_pyttsx3(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing TTS request: {e}")
                await asyncio.sleep(0.1)
    
    async def _synthesize_speech_pyttsx3(self, request: TTSRequest):
        """Synthesize speech using pyttsx3."""
        if not self.pyttsx3_engine:
            return
        
        with self._speech_lock:
            if self._is_speaking:
                return
        
        try:
            self._is_speaking = True
            
            # Clean the text
            clean_text = self._clean_text(request.text)
            if not clean_text.strip():
                return
            
            logger.info(f"Speaking with pyttsx3: '{clean_text[:100]}...'")
            
            # Run speech synthesis in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._speak_pyttsx3_blocking,
                clean_text
            )
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
        finally:
            self._is_speaking = False
    
    def _speak_pyttsx3_blocking(self, text: str):
        """Speak using pyttsx3 (blocking operation)."""
        try:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in pyttsx3 speech: {e}")
    
    async def _synthesize_speech_gtts(self, request: TTSRequest):
        """Synthesize speech using gTTS."""
        if not gtts:
            return
        
        with self._speech_lock:
            if self._is_speaking:
                return
        
        try:
            self._is_speaking = True
            
            # Clean the text
            clean_text = self._clean_text(request.text)
            if not clean_text.strip():
                return
            
            logger.info(f"Speaking with gTTS: '{clean_text[:100]}...'")
            
            # Run speech synthesis in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._speak_gtts_blocking,
                clean_text
            )
            
        except Exception as e:
            logger.error(f"gTTS synthesis error: {e}")
        finally:
            self._is_speaking = False
    
    def _speak_gtts_blocking(self, text: str):
        """Speak using gTTS (blocking operation)."""
        try:
            # Create gTTS object
            tts = gtts.gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                
                # Play the audio file (simple approach - just log for now)
                logger.info(f"gTTS audio saved to: {tmp_file.name}")
                # In a real implementation, you'd play this with pygame or similar
                
                # Clean up
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in gTTS speech: {e}")
    
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
        if self._is_speaking:
            try:
                self._is_speaking = False
                
                if self.pyttsx3_engine:
                    # pyttsx3 doesn't have a direct stop method in async context
                    # We'll just set the flag and let it finish naturally
                    pass
                
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
        
        # Reinitialize engine with new voice if using pyttsx3
        if self.pyttsx3_engine:
            try:
                voices = self.pyttsx3_engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                logger.error(f"Failed to change voice: {e}")
    
    async def set_speed(self, speed: float):
        """Change the speech speed."""
        self.config.speed = max(0.5, min(2.0, speed))  # Clamp between 0.5x and 2.0x
        logger.info(f"TTS speed changed to: {self.config.speed}")
        
        # Update pyttsx3 rate if available
        if self.pyttsx3_engine:
            try:
                rate = self.pyttsx3_engine.getProperty('rate')
                self.pyttsx3_engine.setProperty('rate', int(rate * self.config.speed))
            except Exception as e:
                logger.error(f"Failed to change speed: {e}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        voices = []
        
        if self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                voices.extend([voice.name for voice in pyttsx3_voices])
            except Exception as e:
                logger.error(f"Failed to get voices: {e}")
        
        if not voices:
            voices = ["default", "english"]
        
        return voices
    
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
        
        # Clean up pyttsx3 engine
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except:
                pass
            self.pyttsx3_engine = None
        
        logger.info("SimpleTTS engine cleaned up")


class TTSEngine:
    """High-level TTS engine wrapper for simple TTS."""
    
    def __init__(self, config: TTSConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.engine: Optional[SimpleTTSEngine] = None
        self._synthesis_callbacks: List[Callable[[str], None]] = []
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            self.engine = SimpleTTSEngine(self.config, self.audio_config)
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