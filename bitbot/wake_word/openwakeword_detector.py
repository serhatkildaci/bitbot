"""
OpenWakeWord Detector for BitBot
===============================

Fully offline wake word detection using OpenWakeWord.
Perfect for open source projects - no API keys, no internet required.
"""

import asyncio
import time
import threading
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import openwakeword
    from openwakeword.model import Model
    HAS_OPENWAKEWORD = True
except ImportError:
    logger.warning("OpenWakeWord not available. Install with: pip install openwakeword")
    HAS_OPENWAKEWORD = False

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


class OpenWakeWordDetector:
    """OpenWakeWord-based wake word detector - fully offline, no API keys."""
    
    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        if not HAS_OPENWAKEWORD:
            raise ImportError("OpenWakeWord not available. Install with: pip install openwakeword")
        
        self.config = config
        self.audio_config = audio_config
        self.model: Optional[Model] = None
        self._detection_callbacks: List[Callable[[WakeWordDetection], None]] = []
        self._is_listening = False
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        
        # OpenWakeWord expects 16kHz audio
        self.sample_rate = 16000
        self.frame_length = 1280  # 80ms at 16kHz (optimal for OpenWakeWord)
        
        logger.info(f"OpenWakeWord frame length: {self.frame_length} samples")
        logger.info(f"BitBot chunk size: {audio_config.chunk_size} samples")
        
        logger.info(f"OpenWakeWordDetector initialized for offline wake word detection")

    async def initialize(self) -> bool:
        """Initialize the OpenWakeWord detector."""
        try:
            # Choose wake word model based on config
            wake_word = self._get_best_available_model()
            
            logger.info(f"Initializing OpenWakeWord with model: '{wake_word}'")
            
            # Create OpenWakeWord model
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: Model(wakeword_models=[wake_word])
            )
            
            logger.info("OpenWakeWord detector initialized successfully")
            logger.info(f"✅ Fully offline wake word detection ready!")
            logger.info(f"✅ No API keys required - perfect for open source!")
            # Show proper wake word phrase instead of file path
            display_keyword = self.config.keyword if wake_word.endswith(('.tflite', '.onnx')) else wake_word
            logger.info(f"🎯 Say '{display_keyword}' to activate BitBot")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord: {e}")
            return False

    def _get_best_available_model(self) -> str:
        """Choose the best available wake word model."""
        # First check for custom "Hey BitBot" models
        custom_model_dir = Path(__file__).parent
        logger.info(f"🔍 Checking for custom models in: {custom_model_dir}")
        
        # Try ONNX model first (usually better performance)
        custom_onnx = custom_model_dir / "Hey_Bitbot.onnx"
        custom_tflite = custom_model_dir / "Hey_Bitbot.tflite"
        
        logger.info(f"🔍 ONNX model path: {custom_onnx}, exists: {custom_onnx.exists()}")
        logger.info(f"🔍 TFLite model path: {custom_tflite}, exists: {custom_tflite.exists()}")
        
        if custom_onnx.exists():
            logger.info(f"🎯 Using custom Hey BitBot ONNX model: {custom_onnx}")
            return str(custom_onnx)
        elif custom_tflite.exists():
            logger.info(f"🎯 Using custom Hey BitBot TFLite model: {custom_tflite}")
            return str(custom_tflite)
        
        # Fallback to built-in models if custom not found
        user_keyword = self.config.keyword.lower()
        logger.warning("Custom Hey BitBot model not found, falling back to built-in models")
        
        if "bitbot" in user_keyword or "hey bitbot" in user_keyword:
            return "hey jarvis"  # Closest built-in model
        elif "mycroft" in user_keyword:
            return "hey mycroft"  
        elif "alexa" in user_keyword:
            return "alexa"
        elif "jarvis" in user_keyword:
            return "hey jarvis"
        else:
            return "hey jarvis"

    def add_detection_callback(self, callback: Callable[[WakeWordDetection], None]):
        """Add callback for wake word detections."""
        self._detection_callbacks.append(callback)

    def process_audio_chunk(self, chunk: AudioChunk):
        """Process audio chunk for wake word detection."""
        # Debug: Log that we're receiving audio chunks
        if hasattr(self, '_chunk_counter'):
            self._chunk_counter += 1
        else:
            self._chunk_counter = 1
            
        # Log every 50th chunk to confirm audio is reaching wake word detector
        if self._chunk_counter % 50 == 0:
            logger.info(f"🎯 Wake word detector received chunk #{self._chunk_counter}")
            logger.info(f"🎯 Model initialized: {self.model is not None}, Listening: {self._is_listening}")
        
        if not self.model or not self._is_listening:
            if self._chunk_counter % 50 == 0:
                logger.warning(f"🚨 Wake word detector not ready - Model: {self.model is not None}, Listening: {self._is_listening}")
            return

        try:
            # Convert audio to the format expected by OpenWakeWord
            audio_data = self._prepare_audio_data(chunk)
            if audio_data is None:
                return

            with self._buffer_lock:
                # Add new audio to buffer
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_data])
                
                # Debug: Log buffer status every 50th chunk
                if self._chunk_counter % 50 == 0:
                    logger.info(f"🔧 Audio buffer: {len(self._audio_buffer)} samples, need {self.frame_length}, audio_data: {len(audio_data)}")
                
                # Process when we have enough frames (80ms chunks work best)
                while len(self._audio_buffer) >= self.frame_length:
                    frame = self._audio_buffer[:self.frame_length]
                    self._audio_buffer = self._audio_buffer[self.frame_length:]
                    
                    # Debug: Log frame processing attempt
                    if self._chunk_counter % 50 == 0:
                        logger.info(f"🔄 Processing frame: length={len(frame)}, dtype={frame.dtype}, shape={frame.shape}")
                    
                    try:
                        # Get prediction from OpenWakeWord
                        prediction = self.model.predict(frame)
                        
                        # Debug: Log successful prediction
                        if self._chunk_counter % 50 == 0:
                            logger.info(f"✅ Prediction successful: {prediction}")
                            
                    except Exception as pred_error:
                        logger.error(f"🚨 PREDICTION FAILED: {pred_error}")
                        logger.error(f"🚨 Frame details: shape={frame.shape}, dtype={frame.dtype}, min={frame.min():.3f}, max={frame.max():.3f}")
                        continue
                    
                    # Debug: Log prediction results for troubleshooting
                    if hasattr(self, '_debug_counter'):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 1
                        
                    # Log every 100th prediction to avoid spam
                    if self._debug_counter % 100 == 0:
                        logger.debug(f"🔍 Wake word prediction #{self._debug_counter}: {prediction}")
                        max_confidence = max(prediction.values()) if prediction else 0.0
                        logger.debug(f"🔍 Max confidence: {max_confidence:.3f}, Threshold: {self.config.sensitivity}")
                    
                    # Check for wake word detection
                    for wake_word, confidence in prediction.items():
                        # Debug: Log all predictions above 0.1 to see what's happening
                        if confidence > 0.1:
                            logger.info(f"🔍 Wake word '{wake_word}': {confidence:.3f} (threshold: {self.config.sensitivity})")
                        
                        if confidence > self.config.sensitivity:
                            self._handle_detection(wake_word, confidence, chunk)
                        
        except Exception as e:
            logger.error(f"Error processing audio for wake word: {e}")

    def _prepare_audio_data(self, chunk: AudioChunk) -> Optional[np.ndarray]:
        """Prepare audio data for OpenWakeWord processing."""
        try:
            audio_data = chunk.data
            
            # Flatten if multi-channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Resample if necessary (OpenWakeWord expects 16kHz)
            if chunk.sample_rate != self.sample_rate:
                # Simple resampling - in production use proper resampling
                ratio = self.sample_rate / chunk.sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Ensure float32 format for OpenWakeWord
            audio_data = audio_data.astype(np.float32)
            
            # Normalize audio data for OpenWakeWord (expects values roughly -1 to 1)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preparing audio data: {e}")
            return None

    def _handle_detection(self, keyword: str, confidence: float, chunk: AudioChunk):
        """Handle wake word detection."""
        try:
            detection = WakeWordDetection(
                keyword=keyword,
                confidence=confidence,
                timestamp=time.time(),
                audio_chunk=chunk
            )
            
            logger.info(f"🎯 Wake word detected: '{keyword}' (confidence: {confidence:.2f})")
            
            # Call all detection callbacks
            for callback in self._detection_callbacks:
                try:
                    callback(detection)
                except Exception as e:
                    logger.error(f"Error in wake word callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling wake word detection: {e}")

    async def start_listening(self):
        """Start listening for wake words."""
        if not self.model:
            logger.error("OpenWakeWord model not initialized")
            return
        
        self._is_listening = True
        logger.info("🎧 OpenWakeWord detection started - fully offline!")

    async def stop_listening(self):
        """Stop listening for wake words."""
        self._is_listening = False
        with self._buffer_lock:
            self._audio_buffer = np.array([], dtype=np.float32)
        logger.info("OpenWakeWord detection stopped")

    def is_listening(self) -> bool:
        """Check if currently listening for wake words."""
        return self._is_listening

    async def set_sensitivity(self, sensitivity: float):
        """Update detection sensitivity."""
        self.config.sensitivity = max(0.0, min(1.0, sensitivity))
        logger.info(f"Wake word sensitivity updated to: {self.config.sensitivity}")

    def get_info(self) -> dict:
        """Get detector information."""
        if not self.model:
            return {}
        
        return {
            "library": "OpenWakeWord",
            "version": openwakeword.__version__ if hasattr(openwakeword, '__version__') else "unknown",
            "sample_rate": self.sample_rate,
            "frame_length": self.frame_length,
            "keyword": self.config.keyword,
            "sensitivity": self.config.sensitivity,
            "is_listening": self._is_listening,
            "offline": True,
            "api_key_required": False,
            "open_source": True
        }

    async def cleanup(self):
        """Clean up OpenWakeWord resources."""
        await self.stop_listening()
        
        if self.model:
            # OpenWakeWord doesn't require explicit cleanup
            self.model = None
        
        logger.info("OpenWakeWordDetector cleaned up")


class WakeWordDetector:
    """High-level wake word detector wrapper using OpenWakeWord."""
    
    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.detector: Optional[OpenWakeWordDetector] = None
        self._detection_callbacks: List[Callable[[WakeWordDetection], None]] = []
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the wake word detector."""
        try:
            self.detector = OpenWakeWordDetector(self.config, self.audio_config)
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
                logger.info("🎯 WakeWordDetector initialized successfully - fully offline!")
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
        """Check if currently listening for wake words."""
        if self.detector:
            return self.detector.is_listening()
        return False

    async def set_sensitivity(self, sensitivity: float):
        """Update detection sensitivity."""
        if self.detector:
            await self.detector.set_sensitivity(sensitivity)

    def get_info(self) -> dict:
        """Get detector information."""
        if self.detector:
            return self.detector.get_info()
        return {}

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