"""
BitBot Configuration Settings
=============================

Configuration management for BitBot with automatic hardware tier detection
and component-specific settings optimized for gentle resource usage.
"""

import os
import psutil
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class HardwareTier(Enum):
    """Hardware configuration tiers optimized for gentle resource usage."""
    MINIMAL = "BitBotMin"    # Very limited resources, ultra-conservative
    SMALL = "BitBotS"        # Standard PCs, older laptops, M1 with 8GB
    MEDIUM = "BitBotM"       # Capable PCs, modern laptops with 16GB+
    LARGE = "BitBotL"        # High-end PCs, workstations with 32GB+


@dataclass
class ResourceLimits:
    """Resource usage limits to be gentle with hardware."""
    max_cpu_percent: float = 30.0    # Maximum CPU usage percentage
    max_memory_percent: float = 25.0  # Maximum memory usage percentage
    max_gpu_percent: float = 40.0     # Maximum GPU usage percentage
    threads: int = 2                  # Number of processing threads


@dataclass
class STTConfig:
    """Speech-to-Text configuration per hardware tier."""
    model_name: str
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1              # Smaller beam for less CPU usage
    best_of: int = 1                # Single pass for speed


@dataclass 
class LLMConfig:
    """Large Language Model configuration per hardware tier."""
    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024          # Reduced for faster processing
    stream: bool = True
    num_ctx: int = 2048             # Context window size
    num_thread: int = 2             # CPU threads for inference


@dataclass
class TTSConfig:
    """Text-to-Speech configuration per hardware tier."""
    engine: str = "pyttsx3"         # Use simple TTS by default
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 0.8


@dataclass
class AudioConfig:
    """Audio input/output configuration optimized per hardware tier."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024          # Larger chunks for M1 efficiency
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    buffer_size: int = 8            # Larger buffer for gentle processing
    max_buffer_chunks: int = 500    # Much larger audio buffer for M1 systems


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    keyword: str = "Hey BitBot"
    sensitivity: float = 0.0015     # Optimized threshold - above baseline noise (~0.0009) but allows detection


class BitBotConfig:
    """Main BitBot configuration with conservative hardware detection."""
    
    # Hardware tier configurations - optimized for gentle resource usage
    TIER_CONFIGS = {
        HardwareTier.MINIMAL: {
            "stt": STTConfig(
                model_name="tiny",  # Smallest possible model
                compute_type="int8",
                beam_size=1,
                best_of=1
            ),
            "llm": LLMConfig(
                model_name="gemma:2b",  # Very small model
                max_tokens=512,
                num_ctx=1024,
                num_thread=1
            ),
            "tts": TTSConfig(engine="pyttsx3"),
            "resources": ResourceLimits(
                max_cpu_percent=20.0,
                max_memory_percent=15.0,
                threads=1
            )
        },
        HardwareTier.SMALL: {
            "stt": STTConfig(
                model_name="tiny.en",
                compute_type="int8",
                beam_size=1
            ),
            "llm": LLMConfig(
                model_name="llama3.2:3b",  # Smaller, more efficient model
                max_tokens=768,
                num_ctx=2048,
                num_thread=2
            ),
            "tts": TTSConfig(engine="pyttsx3"),
            "resources": ResourceLimits(
                max_cpu_percent=30.0,
                max_memory_percent=25.0,
                threads=2
            )
        },
        HardwareTier.MEDIUM: {
            "stt": STTConfig(
                model_name="small.en", 
                compute_type="int8",
                beam_size=2
            ),
            "llm": LLMConfig(
                model_name="llama3.2:3b",  # Still conservative for 16GB systems
                max_tokens=1024,
                num_ctx=4096,
                num_thread=4
            ),
            "tts": TTSConfig(engine="pyttsx3"),
            "resources": ResourceLimits(
                max_cpu_percent=40.0,
                max_memory_percent=35.0,
                threads=4
            )
        },
        HardwareTier.LARGE: {
            "stt": STTConfig(
                model_name="small.en",  # Conservative even for large systems
                device="cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu",
                compute_type="float16" if (HAS_TORCH and torch and torch.cuda.is_available()) else "int8",
                beam_size=3
            ),
            "llm": LLMConfig(
                model_name="llama3.1:8b",
                max_tokens=2048,
                num_ctx=8192,
                num_thread=6
            ),
            "tts": TTSConfig(engine="pyttsx3"),
            "resources": ResourceLimits(
                max_cpu_percent=50.0,
                max_memory_percent=40.0,
                max_gpu_percent=60.0,
                threads=6
            )
        }
    }

    def __init__(self, tier: Optional[HardwareTier] = None):
        """Initialize configuration with optional manual tier override."""
        self.tier = tier or self._detect_hardware_tier()
        
        # Configure audio settings based on hardware tier for gentle processing
        if self.tier == HardwareTier.MINIMAL:
            # Ultra-gentle for M1 8GB systems
            self.audio = AudioConfig(
                sample_rate=16000,
                chunk_size=2048,         # Larger chunks = less frequent processing
                buffer_size=16,          # Larger buffers
                max_buffer_chunks=1000   # Even larger audio buffer
            )
        elif self.tier == HardwareTier.SMALL:
            # Gentle for small systems
            self.audio = AudioConfig(
                sample_rate=16000,
                chunk_size=1536,
                buffer_size=12,
                max_buffer_chunks=750
            )
        else:
            # Standard for medium+ systems
            self.audio = AudioConfig()
            
        self.wake_word = WakeWordConfig()
        
        # Load tier-specific configurations
        tier_config = self.TIER_CONFIGS[self.tier]
        self.stt = tier_config["stt"]
        self.llm = tier_config["llm"] 
        self.tts = tier_config["tts"]
        self.resources = tier_config["resources"]
        
        # Load environment overrides
        self._load_env_config()

    def _detect_hardware_tier(self) -> HardwareTier:
        """Conservatively detect appropriate hardware tier."""
        # Get system memory in GB
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Check for GPU memory if available
        gpu_memory_gb = 0
        if HAS_TORCH and torch and torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                gpu_memory_gb = 0
        
        # Special handling for Apple Silicon M1/M2 - be very conservative
        try:
            import platform
            if platform.processor() == 'arm' or 'arm64' in platform.machine().lower():
                # M1/M2 Macs - be extra gentle
                if memory_gb <= 8:
                    return HardwareTier.MINIMAL  # M1 8GB should use minimal
                elif memory_gb <= 16:
                    return HardwareTier.SMALL    # M1/M2 16GB
                else:
                    return HardwareTier.MEDIUM   # M1/M2 32GB+
        except:
            pass
        
        # Conservative tier detection for other systems
        if memory_gb and cpu_count and memory_gb >= 32 and cpu_count >= 8:
            return HardwareTier.LARGE
        elif memory_gb and cpu_count and memory_gb >= 16 and cpu_count >= 4:
            return HardwareTier.MEDIUM
        elif memory_gb and memory_gb >= 8:
            return HardwareTier.SMALL
        else:
            return HardwareTier.MINIMAL

    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Ollama base URL override
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.llm.base_url = ollama_url
        
        # Resource limit overrides for extra caution
        if max_cpu := os.getenv("BITBOT_MAX_CPU_PERCENT"):
            try:
                self.resources.max_cpu_percent = min(float(max_cpu), 50.0)
            except ValueError:
                pass
                
        if max_mem := os.getenv("BITBOT_MAX_MEMORY_PERCENT"):
            try:
                self.resources.max_memory_percent = min(float(max_mem), 40.0)
            except ValueError:
                pass

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "tier": self.tier.value,
            "stt_model": self.stt.model_name,
            "llm_model": self.llm.model_name,
            "tts_engine": self.tts.engine,
            "audio_sample_rate": self.audio.sample_rate,
            "wake_word": self.wake_word.keyword,
            "max_cpu_percent": self.resources.max_cpu_percent,
            "max_memory_percent": self.resources.max_memory_percent,
            "threads": self.resources.threads
        }

    def auto_download_models(self) -> Dict[str, str]:
        """Automatically download appropriate models for the detected tier."""
        models_to_download = {
            "stt_model": self.stt.model_name,
            "llm_model": self.llm.model_name
        }
        
        return models_to_download


# Global configuration instance
config = BitBotConfig() 