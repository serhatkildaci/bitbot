"""
BitBot Text-to-Speech Module
============================

Text-to-speech engines for BitBot with support for multiple backends
including RealtimeTTS and simple pyttsx3-based synthesis.
"""

# Export both TTS engines for flexibility
from .realtime_engine import TTSEngine as RealtimeTTSEngine
from .simple_engine import TTSEngine as SimpleTTSEngine

# Default export is the simple engine for MVP
TTSEngine = SimpleTTSEngine

__all__ = [
    "TTSEngine",
    "RealtimeTTSEngine", 
    "SimpleTTSEngine"
] 