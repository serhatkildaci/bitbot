"""
BitBot Core Pipeline
====================

Main asyncio orchestration pipeline for BitBot that coordinates the
STT → LLM → TTS streaming pipeline with wake word detection.
"""

import asyncio
import signal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time

from loguru import logger
from ..config.settings import BitBotConfig, config
from ..audio.manager import AudioManager, AudioChunk
from ..stt.whisper_engine import STTEngine, TranscriptionResult
from ..llm.ollama_client import LLMEngine, LLMResponse
from ..tts.simple_engine import TTSEngine  # Use simple TTS engine instead
from ..wake_word.porcupine_detector import WakeWordDetector, WakeWordDetection


class PipelineState(Enum):
    """Pipeline states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    LISTENING = "listening"      # Waiting for wake word
    PROCESSING = "processing"    # STT → LLM → TTS active
    SPEAKING = "speaking"        # TTS output active
    ERROR = "error"


@dataclass
class PipelineStats:
    """Pipeline performance statistics."""
    wake_word_detections: int = 0
    transcriptions_processed: int = 0
    llm_responses_generated: int = 0
    tts_utterances_spoken: int = 0
    average_response_time: float = 0.0
    last_activity_time: float = 0.0
    uptime_seconds: float = 0.0


class BitBotPipeline:
    """Main BitBot pipeline orchestrator."""
    
    def __init__(self, config: BitBotConfig):
        self.config = config
        self.state = PipelineState.STOPPED
        self.stats = PipelineStats()
        self._start_time = 0.0
        
        # Core components
        self.audio_manager: Optional[AudioManager] = None
        self.stt_engine: Optional[STTEngine] = None
        self.llm_engine: Optional[LLMEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        
        # Pipeline control
        self._is_running = False
        self._conversation_active = False
        self._response_start_time = 0.0
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"BitBot pipeline initialized for tier: {config.tier.value}")
    
    async def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            self.state = PipelineState.INITIALIZING
            logger.info("Initializing BitBot pipeline...")
            
            # Initialize components in dependency order
            success = await self._initialize_components()
            if not success:
                self.state = PipelineState.ERROR
                return False
            
            # Set up component callbacks
            self._setup_callbacks()
            
            self.state = PipelineState.STOPPED
            self._start_time = time.time()
            logger.info("BitBot pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.state = PipelineState.ERROR
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            # Audio Manager
            logger.info("Initializing audio manager...")
            self.audio_manager = AudioManager(self.config.audio)
            if not await self.audio_manager.initialize():
                logger.error("Failed to initialize audio manager")
                return False
            
            # STT Engine
            logger.info("Initializing STT engine...")
            self.stt_engine = STTEngine(self.config.stt)
            if not await self.stt_engine.initialize():
                logger.error("Failed to initialize STT engine")
                return False
            
            # LLM Engine
            logger.info("Initializing LLM engine...")
            self.llm_engine = LLMEngine(self.config.llm)
            if not await self.llm_engine.initialize():
                logger.error("Failed to initialize LLM engine")
                return False
            
            # TTS Engine
            logger.info("Initializing TTS engine...")
            self.tts_engine = TTSEngine(self.config.tts, self.config.audio)
            if not await self.tts_engine.initialize():
                logger.error("Failed to initialize TTS engine")
                return False
            
            # Wake Word Detector
            logger.info("Initializing wake word detector...")
            self.wake_word_detector = WakeWordDetector(self.config.wake_word, self.config.audio)
            if not await self.wake_word_detector.initialize():
                logger.error("Failed to initialize wake word detector")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            return False
    
    def _setup_callbacks(self):
        """Set up callbacks between components."""
        # Audio → STT and Wake Word
        def audio_callback(chunk: AudioChunk):
            # Send to STT engine
            if self.stt_engine and self._conversation_active:
                self.stt_engine.process_audio_chunk(chunk)
            
            # Send to wake word detector
            if self.wake_word_detector and not self._conversation_active:
                self.wake_word_detector.process_audio_chunk(chunk)
        
        if self.audio_manager:
            self.audio_manager.add_audio_callback(audio_callback)
        
        # STT → LLM
        def transcription_callback(result: TranscriptionResult):
            logger.info(f"Transcription received: '{result.text}'")
            asyncio.create_task(self._process_transcription(result))
        
        if self.stt_engine:
            self.stt_engine.add_transcription_callback(transcription_callback)
        
        # LLM → TTS
        def llm_response_callback(response: LLMResponse):
            logger.info(f"LLM response received: '{response.content[:100]}...'")
            asyncio.create_task(self._process_llm_response(response))
        
        if self.llm_engine:
            self.llm_engine.add_response_callback(llm_response_callback)
        
        # TTS synthesis events
        def tts_callback(text: str):
            logger.debug(f"TTS started: '{text[:50]}...'")
        
        if self.tts_engine:
            self.tts_engine.add_synthesis_callback(tts_callback)
        
        # Wake word detection
        def wake_word_callback(detection: WakeWordDetection):
            logger.info(f"Wake word detected: '{detection.keyword}'")
            asyncio.create_task(self._handle_wake_word_detection(detection))
        
        if self.wake_word_detector:
            self.wake_word_detector.add_detection_callback(wake_word_callback)
    
    async def start(self) -> bool:
        """Start the BitBot pipeline."""
        if self.state != PipelineState.STOPPED:
            logger.warning("Pipeline already running or in error state")
            return False
        
        try:
            self._is_running = True
            self.state = PipelineState.LISTENING
            
            # Start audio input
            if not await self.audio_manager.start_listening():
                logger.error("Failed to start audio input")
                return False
            
            # Start audio output capability
            if not await self.audio_manager.start_speaking():
                logger.error("Failed to start audio output")
                return False
            
            # Start wake word detection
            await self.wake_word_detector.start_listening()
            
            logger.info("🤖 BitBot is now listening for 'Hey BitBot'...")
            await self._say_greeting()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.state = PipelineState.ERROR
            return False
    
    async def stop(self):
        """Stop the BitBot pipeline."""
        if not self._is_running:
            return
        
        logger.info("Stopping BitBot pipeline...")
        self._is_running = False
        self.state = PipelineState.STOPPED
        
        # Stop all components gracefully
        tasks = []
        
        if self.wake_word_detector:
            tasks.append(self.wake_word_detector.stop_listening())
        
        if self.stt_engine:
            tasks.append(self.stt_engine.stop_processing())
        
        if self.tts_engine:
            tasks.append(self.tts_engine.stop())
        
        if self.audio_manager:
            tasks.append(self.audio_manager.stop_listening())
            tasks.append(self.audio_manager.stop_speaking())
        
        # Wait for all to stop
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._shutdown_event.set()
        logger.info("BitBot pipeline stopped")
    
    async def _handle_wake_word_detection(self, detection: WakeWordDetection):
        """Handle wake word detection."""
        if self._conversation_active:
            logger.debug("Wake word detected but conversation already active")
            return
        
        self.stats.wake_word_detections += 1
        self.stats.last_activity_time = time.time()
        self._response_start_time = time.time()
        
        # Enter conversation mode
        self._conversation_active = True
        self.state = PipelineState.PROCESSING
        
        # Stop wake word detection temporarily
        await self.wake_word_detector.stop_listening()
        
        # Start STT processing
        await self.stt_engine.start_processing()
        
        # Play acknowledgment sound or brief response
        await self.tts_engine.speak("Yes?", interrupt=True)
        
        logger.info("🎙️ Conversation started - listening for user input...")
        
        # Set timeout for conversation
        asyncio.create_task(self._conversation_timeout())
    
    async def _process_transcription(self, result: TranscriptionResult):
        """Process STT transcription result."""
        if not self._conversation_active:
            return
        
        self.stats.transcriptions_processed += 1
        
        # Check for conversation end phrases
        if self._is_conversation_end(result.text):
            await self._end_conversation()
            return
        
        # Process with LLM
        response = await self.llm_engine.process_transcription(result)
        if not response:
            # Fallback response
            await self.tts_engine.speak("I didn't understand that. Could you please repeat?")
            return
    
    async def _process_llm_response(self, response: LLMResponse):
        """Process LLM response."""
        if not self._conversation_active:
            return
        
        self.stats.llm_responses_generated += 1
        
        # Speak the response
        self.state = PipelineState.SPEAKING
        await self.tts_engine.speak_response(response, interrupt=True)
        
        # Update response time stats
        if self._response_start_time > 0:
            response_time = time.time() - self._response_start_time
            self._update_response_time_stats(response_time)
        
        # After speaking, continue listening or end conversation
        await asyncio.sleep(1.0)  # Brief pause
        
        if self._conversation_active:
            self.state = PipelineState.PROCESSING
            logger.info("🎙️ Ready for next input...")
            # Reset for next interaction in this conversation
            self._response_start_time = time.time()
    
    def _is_conversation_end(self, text: str) -> bool:
        """Check if user wants to end conversation."""
        end_phrases = [
            "goodbye", "bye", "exit", "quit", "stop", "done", 
            "that's all", "thank you", "thanks"
        ]
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in end_phrases)
    
    async def _end_conversation(self):
        """End the current conversation."""
        if not self._conversation_active:
            return
        
        logger.info("🔚 Ending conversation")
        
        # Farewell message
        farewell_messages = [
            "Goodbye!", "See you later!", "Take care!", 
            "Have a great day!", "Until next time!"
        ]
        import random
        farewell = random.choice(farewell_messages)
        await self.tts_engine.speak(farewell, interrupt=True)
        
        # Reset conversation state
        self._conversation_active = False
        self.state = PipelineState.LISTENING
        
        # Stop STT processing
        await self.stt_engine.stop_processing()
        
        # Restart wake word detection
        await self.wake_word_detector.start_listening()
        
        logger.info("🤖 BitBot is listening for 'Hey BitBot'...")
    
    async def _conversation_timeout(self):
        """Handle conversation timeout."""
        await asyncio.sleep(30.0)  # 30 second timeout
        
        if self._conversation_active:
            logger.info("Conversation timeout - ending conversation")
            await self.tts_engine.speak("I'll be here when you need me.", interrupt=True)
            await self._end_conversation()
    
    async def _say_greeting(self):
        """Say initial greeting."""
        greeting = "Hello! I'm BitBot, your local AI assistant. Say 'Hey BitBot' to start a conversation."
        await self.tts_engine.speak(greeting)
        self.stats.tts_utterances_spoken += 1
    
    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics."""
        if self.stats.average_response_time == 0:
            self.stats.average_response_time = response_time
        else:
            # Moving average
            self.stats.average_response_time = (
                self.stats.average_response_time * 0.8 + response_time * 0.2
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        self.stats.uptime_seconds = time.time() - self._start_time if self._start_time > 0 else 0
        
        return {
            "state": self.state.value,
            "conversation_active": self._conversation_active,
            "config_tier": self.config.tier.value,
            "components_ready": {
                "audio": self.audio_manager is not None,
                "stt": self.stt_engine and self.stt_engine.is_ready,
                "llm": self.llm_engine and self.llm_engine.is_ready,
                "tts": self.tts_engine and self.tts_engine.is_ready,
                "wake_word": self.wake_word_detector and self.wake_word_detector.is_ready
            },
            "stats": {
                "wake_word_detections": self.stats.wake_word_detections,
                "transcriptions_processed": self.stats.transcriptions_processed,
                "llm_responses_generated": self.stats.llm_responses_generated,
                "tts_utterances_spoken": self.stats.tts_utterances_spoken,
                "average_response_time": round(self.stats.average_response_time, 2),
                "uptime_seconds": round(self.stats.uptime_seconds, 1)
            }
        }
    
    async def cleanup(self):
        """Clean up all pipeline resources."""
        logger.info("Cleaning up BitBot pipeline...")
        
        # Stop if running
        if self._is_running:
            await self.stop()
        
        # Clean up components
        cleanup_tasks = []
        for component in [
            self.audio_manager, self.stt_engine, self.llm_engine,
            self.tts_engine, self.wake_word_detector
        ]:
            if component and hasattr(component, 'cleanup'):
                cleanup_tasks.append(component.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("BitBot pipeline cleanup complete")


class BitBotCore:
    """High-level BitBot core interface."""
    
    def __init__(self, config_override: Optional[BitBotConfig] = None):
        self.config = config_override or config
        self.pipeline: Optional[BitBotPipeline] = None
        self._shutdown_handlers_set = False
    
    async def initialize(self) -> bool:
        """Initialize BitBot core."""
        try:
            logger.info(f"Initializing BitBot {self.config.tier.value}...")
            logger.info(f"Configuration: {self.config.get_config_summary()}")
            
            self.pipeline = BitBotPipeline(self.config)
            success = await self.pipeline.initialize()
            
            if success:
                self._setup_shutdown_handlers()
                logger.info("✅ BitBot core initialized successfully")
            else:
                logger.error("❌ BitBot core initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"BitBot core initialization error: {e}")
            return False
    
    async def start(self) -> bool:
        """Start BitBot."""
        if not self.pipeline:
            logger.error("BitBot not initialized")
            return False
        
        return await self.pipeline.start()
    
    async def stop(self):
        """Stop BitBot."""
        if self.pipeline:
            await self.pipeline.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get BitBot status."""
        if self.pipeline:
            return self.pipeline.get_status()
        return {"state": "not_initialized"}
    
    async def run_forever(self):
        """Run BitBot until shutdown signal."""
        if not self.pipeline:
            logger.error("BitBot not initialized")
            return
        
        if not await self.start():
            logger.error("Failed to start BitBot")
            return
        
        try:
            # Wait for shutdown signal
            await self.pipeline._shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.stop()
            await self.cleanup()
    
    def _setup_shutdown_handlers(self):
        """Set up graceful shutdown handlers."""
        if self._shutdown_handlers_set:
            return
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            if self.pipeline:
                asyncio.create_task(self.pipeline.stop())
        
        # Set up signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            self._shutdown_handlers_set = True
        except Exception as e:
            logger.warning(f"Could not set signal handlers: {e}")
    
    async def cleanup(self):
        """Clean up BitBot core."""
        if self.pipeline:
            await self.pipeline.cleanup()
            self.pipeline = None 