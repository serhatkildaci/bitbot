"""
BitBot Audio Manager
====================

Audio input/output management using sounddevice with asyncio integration
for non-blocking real-time audio processing.
"""

import asyncio
import queue
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

from loguru import logger
from ..config.settings import AudioConfig


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int


class AudioBuffer:
    """Thread-safe audio buffer optimized for gentle M1 processing."""
    
    def __init__(self, maxsize: int = 500):  # Much larger default for M1 systems
        self._queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._dropped_chunks = 0
        self._last_warning_time = 0
    
    def put(self, chunk: AudioChunk, block: bool = True) -> bool:
        """Add audio chunk to buffer with gentle overflow handling."""
        try:
            self._queue.put(chunk, block=block)
            return True
        except queue.Full:
            self._dropped_chunks += 1
            # Only warn every 5 seconds to avoid spam
            import time
            current_time = time.time()
            if current_time - self._last_warning_time > 5.0:
                logger.warning(f"Audio buffer full, dropped {self._dropped_chunks} chunks (processing too slow for hardware)")
                self._last_warning_time = current_time
                self._dropped_chunks = 0
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """Get audio chunk from buffer."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear(self):
        """Clear all chunks from buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Signal buffer to stop."""
        self._stop_event.set()
    
    def is_stopped(self) -> bool:
        """Check if buffer is stopped."""
        return self._stop_event.is_set()


class AudioStream:
    """Manages audio input/output streams with asyncio integration."""
    
    def __init__(self, config: AudioConfig):
        if sd is None:
            raise ImportError("sounddevice not available. Install with: pip install sounddevice")
        
        self.config = config
        self.input_stream: Optional[Any] = None  # sd.InputStream when available
        self.output_stream: Optional[Any] = None  # sd.OutputStream when available
        # Use larger buffer sizes for M1 systems to prevent overflow
        self.input_buffer = AudioBuffer(maxsize=getattr(config, 'max_buffer_chunks', 500))
        self.output_buffer = AudioBuffer(maxsize=getattr(config, 'max_buffer_chunks', 500))
        self._input_callback: Optional[Callable] = None
        self._is_recording = False
        self._is_playing = False
        
        logger.info(f"AudioStream initialized (M1-optimized): {config.sample_rate}Hz, {config.channels} channels, {config.chunk_size} chunk size, {getattr(config, 'max_buffer_chunks', 500)} buffer chunks")
    
    def set_input_callback(self, callback: Callable[[AudioChunk], None]):
        """Set callback for processing input audio chunks."""
        self._input_callback = callback
    
    def _input_stream_callback(self, indata, frames, time, status):
        """Callback for input stream - runs in audio thread."""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if self._is_recording:
            chunk = AudioChunk(
                data=indata.copy(),
                timestamp=time.inputBufferAdcTime,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels
            )
            
            # Add to buffer
            self.input_buffer.put(chunk, block=False)
            
            # Call callback if set
            if self._input_callback:
                try:
                    self._input_callback(chunk)
                except Exception as e:
                    logger.error(f"Error in input callback: {e}")
    
    def _output_stream_callback(self, outdata, frames, time, status):
        """Callback for output stream - runs in audio thread."""
        if status:
            logger.warning(f"Audio output status: {status}")
        
        # Get audio from output buffer
        chunk = self.output_buffer.get(timeout=0.001)
        if chunk and chunk.data.shape[0] >= frames:
            outdata[:] = chunk.data[:frames]
        else:
            # Silence if no data available
            outdata.fill(0)
    
    async def start_recording(self) -> bool:
        """Start audio recording."""
        try:
            if self.input_stream is not None:
                logger.warning("Recording already active")
                return False
            
            self.input_stream = sd.InputStream(
                callback=self._input_stream_callback,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                device=self.config.input_device,
                dtype=np.float32
            )
            
            self.input_stream.start()
            self._is_recording = True
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    async def stop_recording(self):
        """Stop audio recording."""
        if self.input_stream is not None:
            self._is_recording = False
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            self.input_buffer.clear()
            logger.info("Audio recording stopped")
    
    async def start_playback(self) -> bool:
        """Start audio playback."""
        try:
            if self.output_stream is not None:
                logger.warning("Playback already active")
                return False
            
            self.output_stream = sd.OutputStream(
                callback=self._output_stream_callback,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                device=self.config.output_device,
                dtype=np.float32
            )
            
            self.output_stream.start()
            self._is_playing = True
            logger.info("Audio playback started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            return False
    
    async def stop_playback(self):
        """Stop audio playback."""
        if self.output_stream is not None:
            self._is_playing = False
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            self.output_buffer.clear()
            logger.info("Audio playback stopped")
    
    async def play_audio(self, audio_data: np.ndarray):
        """Queue audio data for playback."""
        if not self._is_playing:
            logger.warning("Playback not active")
            return
        
        chunk = AudioChunk(
            data=audio_data,
            timestamp=0,  # Will be set by output stream
            sample_rate=self.config.sample_rate,
            channels=self.config.channels
        )
        
        self.output_buffer.put(chunk, block=False)
    
    async def get_input_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get audio chunk from input buffer (async)."""
        # Run blocking operation in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.input_buffer.get,
            timeout
        )
    
    def get_available_devices(self) -> dict:
        """Get available audio input/output devices."""
        return {
            "input_devices": [
                {"id": i, "name": dev["name"], "channels": dev["max_input_channels"]}
                for i, dev in enumerate(sd.query_devices())
                if dev["max_input_channels"] > 0
            ],
            "output_devices": [
                {"id": i, "name": dev["name"], "channels": dev["max_output_channels"]}
                for i, dev in enumerate(sd.query_devices())
                if dev["max_output_channels"] > 0
            ]
        }
    
    async def cleanup(self):
        """Clean up audio streams."""
        await self.stop_recording()
        await self.stop_playback()
        self.input_buffer.stop()
        self.output_buffer.stop()


class AudioManager:
    """High-level audio manager for BitBot."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.stream: Optional[AudioStream] = None
        self._audio_callbacks = []
    
    async def initialize(self) -> bool:
        """Initialize audio manager."""
        try:
            self.stream = AudioStream(self.config)
            logger.info("AudioManager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AudioManager: {e}")
            return False
    
    def add_audio_callback(self, callback: Callable[[AudioChunk], None]):
        """Add callback for processing audio chunks."""
        self._audio_callbacks.append(callback)
        if self.stream:
            # Combine all callbacks
            def combined_callback(chunk):
                for cb in self._audio_callbacks:
                    try:
                        cb(chunk)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
            self.stream.set_input_callback(combined_callback)
    
    async def start_listening(self) -> bool:
        """Start listening for audio input."""
        if not self.stream:
            logger.error("AudioManager not initialized")
            return False
        
        return await self.stream.start_recording()
    
    async def stop_listening(self):
        """Stop listening for audio input."""
        if self.stream:
            await self.stream.stop_recording()
    
    async def start_speaking(self) -> bool:
        """Start audio output capability."""
        if not self.stream:
            logger.error("AudioManager not initialized")
            return False
        
        return await self.stream.start_playback()
    
    async def stop_speaking(self):
        """Stop audio output."""
        if self.stream:
            await self.stream.stop_playback()
    
    async def speak_audio(self, audio_data: np.ndarray):
        """Output audio data."""
        if self.stream:
            await self.stream.play_audio(audio_data)
    
    async def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get next audio chunk."""
        if self.stream:
            return await self.stream.get_input_chunk(timeout)
        return None
    
    def get_devices(self) -> dict:
        """Get available audio devices."""
        if self.stream:
            return self.stream.get_available_devices()
        return {"input_devices": [], "output_devices": []}
    
    async def cleanup(self):
        """Clean up audio manager."""
        if self.stream:
            await self.stream.cleanup()
            self.stream = None
        logger.info("AudioManager cleaned up") 