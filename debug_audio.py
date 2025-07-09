#!/usr/bin/env python3
"""
Audio Debug Script for OpenWakeWord
===================================

Test OpenWakeWord directly with minimal setup.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import openwakeword
    from openwakeword.model import Model
    print("✅ OpenWakeWord imported successfully")
except ImportError as e:
    print(f"❌ OpenWakeWord import failed: {e}")
    sys.exit(1)

def test_openwakeword_direct():
    """Test OpenWakeWord with synthetic audio."""
    print("🧪 Testing OpenWakeWord directly...")
    
    try:
        # Create model with built-in wake word
        print("Creating model with 'hey jarvis'...")
        model = Model(wakeword_models=["hey jarvis"])
        print("✅ Model created successfully")
        
        # Test with dummy audio (should give low confidence)
        print("Testing with dummy audio...")
        dummy_audio = np.random.randn(1280).astype(np.float32) * 0.01  # Quiet random noise
        prediction = model.predict(dummy_audio)
        print(f"Dummy audio prediction: {prediction}")
        
        # Test with synthetic wake word-like audio (won't work but tests pipeline)
        print("Testing with synthetic audio...")
        synthetic_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.08, 1280)).astype(np.float32) * 0.1
        prediction = model.predict(synthetic_audio)
        print(f"Synthetic audio prediction: {prediction}")
        
        print("\n✅ OpenWakeWord basic functionality works")
        print("🎯 Issue is likely in audio pipeline or model compatibility")
        
    except Exception as e:
        print(f"❌ OpenWakeWord test failed: {e}")
        import traceback
        traceback.print_exc()

def check_audio_format():
    """Check our audio format compatibility."""
    print("\n🔍 Checking audio format...")
    
    # Simulate BitBot audio chunk
    sample_rate = 16000
    chunk_size = 2048
    duration = chunk_size / sample_rate
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk size: {chunk_size} samples")
    print(f"Chunk duration: {duration:.3f} seconds")
    print(f"Expected by OpenWakeWord: 16000 Hz, 1280 samples (0.08s)")
    
    if chunk_size != 1280:
        print("⚠️  Chunk size mismatch! BitBot uses 2048, OpenWakeWord expects 1280")
        print("   This might cause timing issues")
    
    print("✅ Sample rate matches (16kHz)")

if __name__ == "__main__":
    print("🐛 OpenWakeWord Debug Test")
    print("=" * 40)
    
    test_openwakeword_direct()
    check_audio_format()
    
    print("\n📋 Next steps:")
    print("1. If this test passes, the issue is in BitBot's audio pipeline")
    print("2. If this test fails, OpenWakeWord installation is broken")
    print("3. Check chunk size mismatch (2048 vs 1280)") 