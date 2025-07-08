"""
BitBot Wake Word Detection
==========================

Wake word detection using Picovoice Porcupine for "Hey BitBot!" 
trigger phrase with high accuracy and low resource usage.
"""

import asyncio
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass
import time
from pathlib import Path

try:
    import pvporcupine
    import numpy as np
except ImportError:
    pvporcupine = None
    np = None

from loguru import logger
from ..config.settings import WakeWordConfig, AudioConfig
from ..audio.manager import AudioChunk


@dataclass
class WakeWordDetection:
    """Wake word detection result."""
    keyword: str
    confidence: float
    timestamp: float
    audio_chunk: Optional[AudioChunk] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PorcupineDetector:
    """Porcupine wake word detector."""
    
    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        if pvporcupine is None:
            raise ImportError("pvporcupine not available. Install with: pip install pvporcupine")
        
        self.config = config
        self.audio_config = audio_config
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self._detection_callbacks: List[Callable[[WakeWordDetection], None]] = []
        self._is_listening = False
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        
        logger.info(f"PorcupineDetector initialized for keyword: '{config.keyword}'")
    
    async def initialize(self) -> bool:
        """Initialize the Porcupine wake word detector."""
        try:
            # Check for access key
            if not self.config.access_key:
                logger.error("Porcupine access key not provided. Get one from https://console.picovoice.ai/")
                return False
            
            # Get built-in keywords or use custom keyword
            keywords = self._get_keywords()
            if not keywords:
                logger.error("No valid keywords found")
                return False
            
            # Run Porcupine initialization in executor
            loop = asyncio.get_event_loop()
            self.porcupine = await loop.run_in_executor(
                None,
                self._initialize_porcupine,
                keywords
            )
            
            if self.porcupine:
                logger.info("Porcupine wake word detector initialized successfully")
                logger.info(f"Sample rate: {self.porcupine.sample_rate}Hz")
                logger.info(f"Frame length: {self.porcupine.frame_length}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            return False
    
    def _initialize_porcupine(self, keywords):
        """Initialize Porcupine (blocking operation)."""
        try:
            # For built-in keywords
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                return pvporcupine.create(
                    access_key=self.config.access_key,
                    keywords=keywords,
                    sensitivities=[self.config.sensitivity] * len(keywords)
                )
            # For custom keyword files
            elif isinstance(keywords, list) and all(isinstance(k, Path) for k in keywords):
                return pvporcupine.create(
                    access_key=self.config.access_key,
                    keyword_paths=[str(k) for k in keywords],
                    sensitivities=[self.config.sensitivity] * len(keywords)
                )
            else:
                logger.error("Invalid keyword format")
                return None
                
        except Exception as e:
            logger.error(f"Porcupine initialization error: {e}")
            return None
    
    def _get_keywords(self):
        """Get keywords for detection."""
        # For now, use built-in keywords that are close to "Hey BitBot"
        # In production, you'd want to train a custom keyword
        
        # Available built-in keywords (some common ones)
        available_keywords = [
            "hey google",   # Closest to "Hey BitBot"
            "alexa",
            "hey siri", 
            "jarvis",
            "computer"
        ]
        
        # Check if we have custom keyword files
        custom_keyword_dir = Path("./models/wake_words/")
        if custom_keyword_dir.exists():
            custom_keywords = list(custom_keyword_dir.glob("*.ppn"))
            if custom_keywords:
                logger.info(f"Found custom keywords: {[k.name for k in custom_keywords]}")
                return custom_keywords
        
        # For MVP, use "hey google" as it's similar to "Hey BitBot"
        # Users can train custom "Hey BitBot" keyword later
        logger.info("Using built-in keyword 'hey google' as proxy for 'Hey BitBot'")
        logger.info("To use 'Hey BitBot', train a custom keyword at https://console.picovoice.ai/")
        
        return ["hey google"]
    
    def add_detection_callback(self, callback: Callable[[WakeWordDetection], None]):
        """Add callback for wake word detections."""
        self._detection_callbacks.append(callback)
    
    def process_audio_chunk(self, chunk: AudioChunk):
        """Process audio chunk for wake word detection."""
        if not self.porcupine or not self._is_listening:
            return
        
        try:
            # Convert audio to the format expected by Porcupine
            audio_data = self._prepare_audio_data(chunk)
            if audio_data is None:
                return
            
            with self._buffer_lock:
                self._audio_buffer.extend(audio_data)
                
                # Process when we have enough frames
                frame_length = self.porcupine.frame_length
                while len(self._audio_buffer) >= frame_length:
                    frame = self._audio_buffer[:frame_length]
                    self._audio_buffer = self._audio_buffer[frame_length:]
                    
                    # Check for wake word
                    keyword_index = self.porcupine.process(frame)
                    if keyword_index >= 0:
                        self._handle_detection(keyword_index, chunk)
                        
        except Exception as e:
            logger.error(f"Error processing audio for wake word: {e}")
    
    def _prepare_audio_data(self, chunk: AudioChunk) -> Optional[List[int]]:
        """Prepare audio data for Porcupine processing."""
        try:
            audio_data = chunk.data
            
            # Flatten if multi-channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Resample if necessary (Porcupine expects 16kHz)
            if chunk.sample_rate != self.porcupine.sample_rate:
                # Simple resampling - in production use proper resampling
                ratio = self.porcupine.sample_rate / chunk.sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Convert to 16-bit PCM integers
            audio_data = (audio_data * 32767).astype(np.int16).tolist()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preparing audio data: {e}")
            return None
    
    def _handle_detection(self, keyword_index: int, chunk: AudioChunk):
        """Handle wake word detection."""
        try:
            # Map keyword index to keyword name
            keyword = self._get_keyword_name(keyword_index)
            
            detection = WakeWordDetection(
                keyword=keyword,
                confidence=self.config.sensitivity,  # Porcupine doesn't provide confidence
                timestamp=time.time(),
                audio_chunk=chunk
            )
            
            logger.info(f"Wake word detected: '{keyword}' at {detection.timestamp}")
            
            # Call all detection callbacks
            for callback in self._detection_callbacks:
                try:
                    callback(detection)
                except Exception as e:
                    logger.error(f"Error in wake word callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling wake word detection: {e}")
    
    def _get_keyword_name(self, index: int) -> str:
        """Get keyword name from index."""
        # This would map to the actual keywords used
        # For now, we're using "hey google" as proxy
        keyword_map = {
            0: "hey google"  # Will be interpreted as "Hey BitBot"
        }
        return keyword_map.get(index, "unknown")
    
    async def start_listening(self):
        """Start listening for wake words."""
        if not self.porcupine:
            logger.error("Porcupine not initialized")
            return
        
        self._is_listening = True
        logger.info("Wake word detection started")
    
    async def stop_listening(self):
        """Stop listening for wake words."""
        self._is_listening = False
        with self._buffer_lock:
            self._audio_buffer = []
        logger.info("Wake word detection stopped")
    
    def is_listening(self) -> bool:
        """Check if currently listening for wake words."""
        return self._is_listening
    
    async def set_sensitivity(self, sensitivity: float):
        """Update detection sensitivity."""
        self.config.sensitivity = max(0.0, min(1.0, sensitivity))
        logger.info(f"Wake word sensitivity updated to: {self.config.sensitivity}")
        # Note: Requires reinitializing Porcupine to take effect
    
    def get_info(self) -> dict:
        """Get detector information."""
        if not self.porcupine:
            return {}
        
        return {
            "version": pvporcupine.Porcupine.version,
            "sample_rate": self.porcupine.sample_rate,
            "frame_length": self.porcupine.frame_length,
            "keyword": self.config.keyword,
            "sensitivity": self.config.sensitivity,
            "is_listening": self._is_listening
        }
    
    async def cleanup(self):
        """Clean up Porcupine resources."""
        await self.stop_listening()
        
        if self.porcupine:
            try:
                # Run cleanup in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.porcupine.delete)
            except Exception as e:
                logger.warning(f"Error cleaning up Porcupine: {e}")
            self.porcupine = None
        
        logger.info("PorcupineDetector cleaned up")


class WakeWordDetector:
    """High-level wake word detector wrapper."""
    
    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.detector: Optional[PorcupineDetector] = None
        self._detection_callbacks: List[Callable[[WakeWordDetection], None]] = []
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the wake word detector."""
        try:
            self.detector = PorcupineDetector(self.config, self.audio_config)
            success = await self.detector.initialize()
            if success:
                self._is_initialized = True
                
                # Forward detection callbacks
                def forward_detection(detection):
                    for callback in self._detection_callbacks:
                        try:
                            callback(detection)
                        except Exception as e:
                            logger.error(f"Error in wake word detection callback: {e}")
                
                self.detector.add_detection_callback(forward_detection)
                logger.info("WakeWordDetector initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize WakeWordDetector: {e}")
            return False
    
    def add_detection_callback(self, callback: Callable[[WakeWordDetection], None]):
        """Add callback for wake word detections."""
        self._detection_callbacks.append(callback)
    
    def process_audio_chunk(self, chunk: AudioChunk):
        """Process audio chunk for wake word detection."""
        if self.detector:
            self.detector.process_audio_chunk(chunk)
    
    async def start_listening(self):
        """Start listening for wake words."""
        if self.detector:
            await self.detector.start_listening()
    
    async def stop_listening(self):
        """Stop listening for wake words."""
        if self.detector:
            await self.detector.stop_listening()
    
    def is_listening(self) -> bool:
        """Check if listening for wake words."""
        return self.detector.is_listening() if self.detector else False
    
    async def set_sensitivity(self, sensitivity: float):
        """Set wake word detection sensitivity."""
        if self.detector:
            await self.detector.set_sensitivity(sensitivity)
    
    def get_info(self) -> dict:
        """Get detector information."""
        return self.detector.get_info() if self.detector else {}
    
    @property
    def is_ready(self) -> bool:
        """Check if detector is ready."""
        return self._is_initialized and self.detector is not None
    
    async def cleanup(self):
        """Clean up wake word detector."""
        if self.detector:
            await self.detector.cleanup()
            self.detector = None
        self._is_initialized = False
        logger.info("WakeWordDetector cleaned up") 