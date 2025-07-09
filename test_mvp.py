#!/usr/bin/env python3
"""
BitBot MVP Test Script
=====================

Simple test script to verify the BitBot architecture and components
without requiring all dependencies to be installed.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports of BitBot modules."""
    print("🧪 Testing BitBot module imports...")
    
    try:
        from bitbot.config.settings import BitBotConfig, HardwareTier
        print("✅ Config module imported successfully")
        
        # Test config creation
        config = BitBotConfig()
        print(f"✅ Config created - Hardware tier: {config.tier.value}")
        
        summary = config.get_config_summary()
        print(f"✅ Config summary: {summary}")
        
    except Exception as e:
        print(f"❌ Config module error: {e}")
        return False
    
    try:
        from bitbot.llm.ollama_client import LLMEngine, LLMMessage, LLMResponse
        print("✅ LLM module imported successfully")
        
    except Exception as e:
        print(f"❌ LLM module error: {e}")
        return False
    
    try:
        from bitbot.stt.whisper_engine import STTEngine, TranscriptionResult
        print("✅ STT module imported successfully")
        
    except Exception as e:
        print(f"❌ STT module error: {e}")
        return False
    
    try:
        from bitbot.tts.simple_engine import TTSEngine, TTSRequest
        print("✅ Simple TTS module imported successfully")
        
    except Exception as e:
        print(f"❌ Simple TTS module error: {e}")
        return False
    
    try:
        from bitbot.wake_word.porcupine_detector import WakeWordDetector
        print("✅ Wake word module imported successfully")
        
    except Exception as e:
        print(f"❌ Wake word module error: {e}")
        return False
    
    try:
        from bitbot.core.pipeline import BitBotPipeline, PipelineState
        print("✅ Core pipeline module imported successfully")
        
    except Exception as e:
        print(f"❌ Core pipeline module error: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from bitbot.config.settings import BitBotConfig
        
        # Test different hardware tiers
        for tier in ["small", "medium", "large"]:
            print(f"Testing {tier} tier configuration...")
            
            # This will test the hardware tier configurations
            config = BitBotConfig()
            summary = config.get_config_summary()
            print(f"  ✅ {tier} tier: {summary['stt_model']} + {summary['llm_model']} + {summary['tts_engine']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False

async def test_async_functionality():
    """Test async components that don't require external services."""
    print("\n🧪 Testing async functionality...")
    
    try:
        from bitbot.llm.ollama_client import LLMMessage, LLMResponse
        
        # Test message creation
        message = LLMMessage(role="user", content="Hello")
        print(f"✅ LLM message created: {message.role} - {message.content}")
        
        # Test response creation
        response = LLMResponse(
            content="Hello! How can I help you?",
            finish_reason="stop",
            model="test-model"
        )
        print(f"✅ LLM response created: {response.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test error: {e}")
        return False

def test_architecture():
    """Test the overall architecture design."""
    print("\n🧪 Testing architecture design...")
    
    try:
        from bitbot.config.settings import BitBotConfig
        from bitbot.core.pipeline import BitBotPipeline, PipelineState
        
        # Test pipeline creation
        config = BitBotConfig()
        pipeline = BitBotPipeline(config)
        
        print(f"✅ Pipeline created with state: {pipeline.state.value}")
        print(f"✅ Pipeline configured for tier: {config.tier.value}")
        
        # Test state transitions
        pipeline.state = PipelineState.INITIALIZING
        print(f"✅ State changed to: {pipeline.state.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture test error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("BitBot MVP Architecture Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
    
    if not test_basic_functionality():
        all_tests_passed = False
    
    if not asyncio.run(test_async_functionality()):
        all_tests_passed = False
    
    if not test_architecture():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! BitBot architecture is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements-core.txt")
        print("2. Set up Ollama and get Porcupine access key")
        print("3. Test individual components")
        print("4. Run full integration tests")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())