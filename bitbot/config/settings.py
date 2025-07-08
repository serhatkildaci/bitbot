"""
BitBot Configuration Settings
=============================

Configuration management for BitBot with automatic hardware tier detection
and component-specific settings for optimal performance.
"""

import os
import psutil
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch


class HardwareTier(Enum):
    """Hardware configuration tiers as defined in BitBot specifications."""
    SMALL = "BitBotS"    # Standard PCs, older laptops
    MEDIUM = "BitBotM"   # Capable PCs, modern laptops (4GB VRAM)
    LARGE = "BitBotL"    # High-end PCs, Apple Silicon (>8GB VRAM)


@dataclass
class STTConfig:
    """Speech-to-Text configuration per hardware tier."""
    model_name: str
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


@dataclass 
class LLMConfig:
    """Large Language Model configuration per hardware tier."""
    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True


@dataclass
class TTSConfig:
    """Text-to-Speech configuration per hardware tier."""
    engine: str  # "piper" or "kyutai"
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0


@dataclass
class AudioConfig:
    """Audio input/output configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    keyword: str = "Hey BitBot"
    sensitivity: float = 0.5
    access_key: Optional[str] = None  # Porcupine access key


class BitBotConfig:
    """Main BitBot configuration with automatic hardware detection."""
    
    # Hardware tier configurations
    TIER_CONFIGS = {
        HardwareTier.SMALL: {
            "stt": STTConfig(
                model_name="tiny.en",
                compute_type="int8"
            ),
            "llm": LLMConfig(
                model_name="mistral:7b-instruct"
            ),
            "tts": TTSConfig(
                engine="piper",
                voice="en_US-lessac-medium"
            )
        },
        HardwareTier.MEDIUM: {
            "stt": STTConfig(
                model_name="small.en", 
                compute_type="int8"
            ),
            "llm": LLMConfig(
                model_name="llama3.1:8b"
            ),
            "tts": TTSConfig(
                engine="kyutai",
                voice="default"
            )
        },
        HardwareTier.LARGE: {
            "stt": STTConfig(
                model_name="large-v3",
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "int8"
            ),
            "llm": LLMConfig(
                model_name="llama3.1:8b"
            ),
            "tts": TTSConfig(
                engine="kyutai",
                voice="premium"
            )
        }
    }

    def __init__(self, tier: Optional[HardwareTier] = None):
        """Initialize configuration with optional manual tier override."""
        self.tier = tier or self._detect_hardware_tier()
        self.audio = AudioConfig()
        self.wake_word = WakeWordConfig()
        
        # Load tier-specific configurations
        tier_config = self.TIER_CONFIGS[self.tier]
        self.stt = tier_config["stt"]
        self.llm = tier_config["llm"] 
        self.tts = tier_config["tts"]
        
        # Load environment overrides
        self._load_env_config()

    def _detect_hardware_tier(self) -> HardwareTier:
        """Automatically detect appropriate hardware tier."""
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check for GPU memory if available
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Tier detection logic
        if gpu_memory_gb >= 8 or memory_gb >= 16:
            return HardwareTier.LARGE
        elif gpu_memory_gb >= 4 or memory_gb >= 8:
            return HardwareTier.MEDIUM
        else:
            return HardwareTier.SMALL

    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Porcupine access key
        if access_key := os.getenv("PORCUPINE_ACCESS_KEY"):
            self.wake_word.access_key = access_key
            
        # Ollama base URL override
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.llm.base_url = ollama_url

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "tier": self.tier.value,
            "stt_model": self.stt.model_name,
            "llm_model": self.llm.model_name,
            "tts_engine": self.tts.engine,
            "audio_sample_rate": self.audio.sample_rate,
            "wake_word": self.wake_word.keyword
        }


# Global configuration instance
config = BitBotConfig() 