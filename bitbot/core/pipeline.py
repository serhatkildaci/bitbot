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
from ..wake_word.openwakeword_detector import WakeWordDetector, WakeWordDetection


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
    
    def __init__(self, config: BitBotConfig, debug_mode: bool = False, nowake_mode: bool = False):
        self.config = config
        self.debug_mode = debug_mode
        self.nowake_mode = nowake_mode
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
        
        # Debug mode logging
        self._audio_chunk_counter = 0
        self._audio_routing_counter = 0
        self._transcription_counter = 0
        
        # Chat mode attributes (for --nowake)
        self._chat_mode = False
        self._last_audio_time = 0.0
        self._silence_threshold = 1.0
        self._is_speaking = False
        self._pending_transcription = ""
        self._silence_timer = None
        
        if self.debug_mode:
            logger.info("🐛 DEBUG MODE ENABLED - Microphone input will be logged to screen")
        
        if self.nowake_mode:
            logger.info("🚀 NO-WAKE MODE ENABLED - Direct conversation mode")
            # In no-wake mode, use clean chat interface
            self._chat_mode = True
            self._last_audio_time = 0.0
            self._silence_threshold = 1.0  # 1 second of silence before processing
            self._is_speaking = False
        
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
            
            # Wake Word Detector (skip in no-wake mode)
            if not self.nowake_mode:
                logger.info("Initializing wake word detector...")
                self.wake_word_detector = WakeWordDetector(self.config.wake_word, self.config.audio)
                if not await self.wake_word_detector.initialize():
                    logger.error("Failed to initialize wake word detector")
                    return False
            else:
                logger.info("⏭️ Skipping wake word detector initialization (no-wake mode)")
                self.wake_word_detector = None
            
            return True
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            return False
    
    def _setup_callbacks(self):
        """Set up callbacks between components."""
        # Audio → STT and Wake Word
        def audio_callback(chunk: AudioChunk):
            # Debug: Log audio routing decisions
            if self.debug_mode and hasattr(self, '_audio_routing_counter'):
                self._audio_routing_counter += 1
            elif self.debug_mode:
                self._audio_routing_counter = 1
                
            # Log every 50th chunk to trace audio routing
            if self.debug_mode and self._audio_routing_counter % 50 == 0:
                print(f"🔄 Audio Routing #{self._audio_routing_counter}: "
                      f"STT={'✅' if (self.stt_engine and (self._conversation_active or self.debug_mode)) else '❌'}, "
                      f"WakeWord={'✅' if (self.wake_word_detector and not self._conversation_active) else '❌'}, "
                      f"ConversationActive={self._conversation_active}")
            
            # Send to STT engine (always in debug mode, no-wake mode, or when conversation active)
            # In chat mode, only process audio when not speaking
            if self.stt_engine and (self._conversation_active or self.debug_mode or self.nowake_mode):
                if not (self._chat_mode and self._is_speaking):
                    self.stt_engine.process_audio_chunk(chunk)
                    # Track audio activity for silence detection
                    if self._chat_mode:
                        import time
                        self._last_audio_time = time.time()
            
            # Send to wake word detector (skip in no-wake mode)
            if self.wake_word_detector and not self._conversation_active and not self.nowake_mode:
                if self.debug_mode and self._audio_routing_counter % 50 == 0:
                    print(f"🎯 Sending audio chunk to wake word detector...")
                self.wake_word_detector.process_audio_chunk(chunk)
        
        if self.audio_manager:
            self.audio_manager.add_audio_callback(audio_callback)
        
        # STT → LLM
        def transcription_callback(result: TranscriptionResult):
            # In chat mode, suppress logs and use silence detection
            if not self._chat_mode:
                logger.info(f"Transcription received: '{result.text}'")
            
            # Chat mode: Clean interface, silence detection
            if self._chat_mode and result.text.strip():
                self._pending_transcription = result.text.strip()
                # Start or reset silence timer
                self._start_silence_timer()
                return
            
            # Debug mode: Always show real-time transcription + conversation state
            if self.debug_mode:
                print(f"🗣️  YOU SAID: '{result.text}' (confidence: {result.confidence:.2f})")
                # Every 10th transcription, show conversation state
                if hasattr(self, '_transcription_counter'):
                    self._transcription_counter += 1
                else:
                    self._transcription_counter = 1
                    
                if self._transcription_counter % 10 == 0:
                    print(f"💬 Conversation State: Active={self._conversation_active}, Pipeline State={self.state.value}")
            
            # Only process with LLM if in conversation mode
            if self._conversation_active:
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
        
        # Wake word detection (skip in no-wake mode)
        if not self.nowake_mode:
            def wake_word_callback(detection: WakeWordDetection):
                logger.info(f"Wake word detected: '{detection.keyword}'")
                
                # Debug mode: Log wake word detection to screen
                if self.debug_mode:
                    print(f"🎯 WAKE WORD DETECTED: '{detection.keyword}' (confidence: {detection.confidence:.2f})")
                
                # Handle async task creation safely
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._handle_wake_word_detection(detection))
                    else:
                        # Schedule the task in a running loop
                        asyncio.ensure_future(self._handle_wake_word_detection(detection))
                except RuntimeError:
                    # No event loop, create a new thread for async handling
                    import threading
                    def run_async():
                        asyncio.run(self._handle_wake_word_detection(detection))
                    threading.Thread(target=run_async, daemon=True).start()
            
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
            
            # Start wake word detection (unless in no-wake mode)
            if not self.nowake_mode:
                await self.wake_word_detector.start_listening()
                
                # In debug mode, also start STT for real-time transcription
                if self.debug_mode:
                    await self.stt_engine.start_processing()
                    logger.info("🐛 DEBUG MODE: Real-time speech transcription enabled")
                
                logger.info("🤖 BitBot is now listening for 'Hey BitBot'...")
                await self._say_greeting()
            else:
                # No-wake mode: Start directly in conversation
                self._conversation_active = True
                self.state = PipelineState.PROCESSING
                
                # Start STT processing immediately
                await self.stt_engine.start_processing()
                
                logger.info("🚀 BitBot ready for conversation (no wake word needed)")
                await self.tts_engine.speak("Hello! I'm ready to help. What can I do for you?", interrupt=True)
            
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
        
        # Chat mode: Display clean response
        if self._chat_mode:
            print(f"🤖 BitBot: {response.content}")
            print("🔊 Speaking...")
        
        # Mark as speaking to disable microphone
        self._is_speaking = True
        
        # Speak the response
        self.state = PipelineState.SPEAKING
        await self.tts_engine.speak_response(response, interrupt=True)
        
        # Mark as not speaking to re-enable microphone
        self._is_speaking = False
        
        if self._chat_mode:
            print("🎧 Listening...")
        
        # Update response time stats
        if self._response_start_time > 0:
            response_time = time.time() - self._response_start_time
            self._update_response_time_stats(response_time)
        
        # After speaking, continue listening or end conversation
        await asyncio.sleep(1.0)  # Brief pause
        
        if self._conversation_active:
            self.state = PipelineState.PROCESSING
            if not self._chat_mode:
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
            logger.info("⏰ Conversation timeout - ending conversation")
            await self._end_conversation()
    
    async def _say_greeting(self):
        """Say initial greeting."""
        if self.tts_engine:
            await self.tts_engine.speak("Hello! Say 'Hey BitBot' to start a conversation.", interrupt=True)
    
    def _start_silence_timer(self):
        """Start or restart the silence detection timer."""
        # Cancel existing timer
        if self._silence_timer:
            self._silence_timer.cancel()
        
        # Start new timer
        self._silence_timer = asyncio.create_task(self._handle_silence_detection())
    
    async def _handle_silence_detection(self):
        """Handle silence detection after user stops talking."""
        try:
            await asyncio.sleep(self._silence_threshold)
            
            # If we have pending transcription, process it
            if self._pending_transcription and not self._is_speaking:
                # Display user message in chat mode
                if self._chat_mode:
                    print(f"\n🗣️  You: {self._pending_transcription}")
                
                # Create transcription result and process
                from ..stt.whisper_engine import TranscriptionResult
                result = TranscriptionResult(
                    text=self._pending_transcription,
                    confidence=1.0,
                    start_time=0.0,
                    end_time=0.0,
                    language="en",
                    is_final=True
                )
                
                # Clear pending transcription
                self._pending_transcription = ""
                
                # Process with LLM
                await self._process_transcription(result)
                
        except asyncio.CancelledError:
            # Timer was cancelled (new audio detected)
            pass
    
    def _chat_display(self, message: str, is_user: bool = False):
        """Display message in clean chat format."""
        if self._chat_mode:
            if is_user:
                print(f"\n🗣️  You: {message}")
            else:
                print(f"🤖 BitBot: {message}")

    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics."""
        if self.stats.llm_responses_generated == 1:
            self.stats.average_response_time = response_time
        else:
            # Moving average
            alpha = 0.3  # Weight for new sample
            self.stats.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.stats.average_response_time
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
    
    def __init__(self, config_override: Optional[BitBotConfig] = None, debug_mode: bool = False, nowake_mode: bool = False):
        self.config = config_override or config
        self.debug_mode = debug_mode
        self.nowake_mode = nowake_mode
        self.pipeline: Optional[BitBotPipeline] = None
        self._shutdown_handlers_set = False
    
    async def initialize(self) -> bool:
        """Initialize BitBot core."""
        try:
            logger.info(f"Initializing BitBot {self.config.tier.value}...")
            logger.info(f"Configuration: {self.config.get_config_summary()}")
            
            self.pipeline = BitBotPipeline(self.config, debug_mode=self.debug_mode, nowake_mode=self.nowake_mode)
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
            if self.pipeline:
                await self.pipeline.cleanup()
                self.pipeline = None 
    
    def _setup_shutdown_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        import signal
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            if self.pipeline:
                # Signal the pipeline to stop
                self.pipeline._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._shutdown_handlers_set = True
    
    async def cleanup(self):
        """Clean up BitBot core."""
        if self.pipeline:
            await self.pipeline.cleanup()
            self.pipeline = None 