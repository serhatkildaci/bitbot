"""
BitBot Speech-to-Text Engine
============================

Speech-to-text processing using Faster Whisper with streaming capability
for real-time transcription with asyncio integration.
"""

import asyncio
import threading
from typing import Optional, AsyncGenerator, Callable, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment
except ImportError:
    WhisperModel = None
    Segment = None

from loguru import logger
from ..config.settings import STTConfig
from ..audio.manager import AudioChunk


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    confidence: float
    start_time: float
    end_time: float
    language: str = "en"
    is_final: bool = True


class WhisperSTT:
    """Faster Whisper STT engine with streaming support."""
    
    def __init__(self, config: STTConfig):
        if WhisperModel is None:
            raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
        
        self.config = config
        self.model: Optional[WhisperModel] = None
        self._model_lock = threading.RLock()
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        self._min_chunk_duration = 1.0  # Minimum seconds of audio before processing
        self._max_chunk_duration = 10.0  # Maximum seconds to accumulate
        
        logger.info(f"WhisperSTT initialized with model: {config.model_name}")
    
    async def initialize(self) -> bool:
        """Initialize the Whisper model."""
        try:
            # Run model loading in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model
            )
            logger.info(f"Whisper model '{self.config.model_name}' loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            return False
    
    def _load_model(self) -> WhisperModel:
        """Load the Whisper model (blocking operation)."""
        return WhisperModel(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            download_root=None,  # Use default cache location
        )
    
    def add_audio_chunk(self, chunk: AudioChunk):
        """Add audio chunk to processing buffer."""
        with self._buffer_lock:
            # Convert to numpy array if needed and flatten
            audio_data = chunk.data
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            self._audio_buffer.extend(audio_data.tolist())
            
            # Check if we have enough audio for processing
            duration = len(self._audio_buffer) / chunk.sample_rate
            if duration >= self._min_chunk_duration:
                logger.debug(f"Audio buffer ready for processing: {duration:.2f}s")
    
    def _get_audio_for_processing(self) -> Optional[np.ndarray]:
        """Get accumulated audio for processing."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            
            # Convert to numpy array
            audio_array = np.array(self._audio_buffer, dtype=np.float32)
            
            # Clear buffer (or keep some overlap for better results)
            overlap_samples = int(0.5 * 16000)  # 0.5 second overlap
            if len(self._audio_buffer) > overlap_samples:
                self._audio_buffer = self._audio_buffer[-overlap_samples:]
            else:
                self._audio_buffer = []
            
            return audio_array
    
    async def transcribe_chunk(self) -> Optional[TranscriptionResult]:
        """Transcribe accumulated audio chunks."""
        if not self.model:
            logger.warning("Model not initialized")
            return None
        
        audio_data = self._get_audio_for_processing()
        if audio_data is None or len(audio_data) == 0:
            return None
        
        try:
            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_audio,
                audio_data
            )
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Perform the actual transcription (blocking operation)."""
        with self._model_lock:
            try:
                # Transcribe with Faster Whisper
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.config.language if self.config.language != "auto" else None,
                    beam_size=1,  # Faster beam size for real-time
                    best_of=1,    # Faster decoding
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                
                # Combine all segments into single result
                text_parts = []
                start_time = float('inf')
                end_time = 0
                avg_confidence = 0
                segment_count = 0
                
                for segment in segments:
                    text_parts.append(segment.text.strip())
                    start_time = min(start_time, segment.start)
                    end_time = max(end_time, segment.end)
                    
                    # Faster Whisper doesn't provide confidence per segment
                    # Use info.language_probability as proxy
                    avg_confidence += getattr(info, 'language_probability', 0.8)
                    segment_count += 1
                
                if not text_parts:
                    return None
                
                full_text = " ".join(text_parts).strip()
                if not full_text:
                    return None
                
                final_confidence = avg_confidence / segment_count if segment_count > 0 else 0.8
                
                return TranscriptionResult(
                    text=full_text,
                    confidence=final_confidence,
                    start_time=start_time if start_time != float('inf') else 0,
                    end_time=end_time,
                    language=info.language,
                    is_final=True
                )
                
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                return None
    
    async def transcribe_file(self, audio_file: Path) -> List[TranscriptionResult]:
        """Transcribe an entire audio file."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._transcribe_file_blocking,
                audio_file
            )
            return results
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return []
    
    def _transcribe_file_blocking(self, audio_file: Path) -> List[TranscriptionResult]:
        """Transcribe file in blocking mode."""
        with self._model_lock:
            segments, info = self.model.transcribe(
                str(audio_file),
                language=self.config.language if self.config.language != "auto" else None,
                beam_size=5,
                best_of=5,
                vad_filter=True,
            )
            
            results = []
            for segment in segments:
                results.append(TranscriptionResult(
                    text=segment.text.strip(),
                    confidence=getattr(info, 'language_probability', 0.8),
                    start_time=segment.start,
                    end_time=segment.end,
                    language=info.language,
                    is_final=True
                ))
            
            return results
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self._buffer_lock:
            self._audio_buffer = []
    
    async def cleanup(self):
        """Clean up resources."""
        with self._model_lock:
            self.model = None
        self.clear_buffer()
        logger.info("WhisperSTT cleaned up")


class STTEngine:
    """High-level STT engine wrapper."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.engine: Optional[WhisperSTT] = None
        self._transcription_callbacks: List[Callable[[TranscriptionResult], None]] = []
        self._processing_task: Optional[asyncio.Task] = None
        self._is_processing = False
    
    async def initialize(self) -> bool:
        """Initialize the STT engine."""
        try:
            self.engine = WhisperSTT(self.config)
            success = await self.engine.initialize()
            if success:
                logger.info("STTEngine initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize STTEngine: {e}")
            return False
    
    def add_transcription_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Add callback for transcription results."""
        self._transcription_callbacks.append(callback)
    
    def process_audio_chunk(self, chunk: AudioChunk):
        """Process incoming audio chunk."""
        if self.engine:
            self.engine.add_audio_chunk(chunk)
    
    async def start_processing(self):
        """Start continuous transcription processing."""
        if self._is_processing:
            logger.warning("STT processing already active")
            return
        
        self._is_processing = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("STT processing started")
    
    async def stop_processing(self):
        """Stop transcription processing."""
        self._is_processing = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        logger.info("STT processing stopped")
    
    async def _processing_loop(self):
        """Main processing loop for continuous transcription."""
        while self._is_processing:
            try:
                if self.engine:
                    result = await self.engine.transcribe_chunk()
                    if result and result.text.strip():
                        logger.debug(f"Transcribed: '{result.text}' (confidence: {result.confidence:.2f})")
                        
                        # Call all callbacks
                        for callback in self._transcription_callbacks:
                            try:
                                callback(result)
                            except Exception as e:
                                logger.error(f"Error in transcription callback: {e}")
                
                # Wait before next processing cycle
                await asyncio.sleep(0.5)  # Process every 500ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in STT processing loop: {e}")
                await asyncio.sleep(1.0)  # Wait longer on error
    
    async def transcribe_file(self, audio_file: Path) -> List[TranscriptionResult]:
        """Transcribe an audio file."""
        if self.engine:
            return await self.engine.transcribe_file(audio_file)
        return []
    
    def clear_buffer(self):
        """Clear the audio processing buffer."""
        if self.engine:
            self.engine.clear_buffer()
    
    async def cleanup(self):
        """Clean up STT engine."""
        await self.stop_processing()
        if self.engine:
            await self.engine.cleanup()
            self.engine = None
        logger.info("STTEngine cleaned up") 