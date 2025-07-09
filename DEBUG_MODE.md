# BitBot Debug Mode 🐛

Debug mode provides real-time logging of microphone input and voice interactions to help with troubleshooting and development.

## How to Enable Debug Mode

```bash
python main.py start --debug
```

Or use the short form:
```bash
python main.py start -d
```

## Debug Output Features

### 🎤 **Microphone Input Logging**
- Logs every 10th audio chunk to avoid flooding the screen
- Shows volume levels, peak amplitude, and sample size
- Helps verify microphone is working and audio levels are appropriate

**Example Output:**
```
🎤 Microphone Input #    10: Volume: 0.0123, Peak: 0.4567, Size: 2048 samples
🎤 Microphone Input #    20: Volume: 0.0089, Peak: 0.3210, Size: 2048 samples
```

### 🗣️ **Speech Transcription Logging**
- Displays exactly what you said with confidence scores
- Shows STT (Speech-to-Text) processing results
- Helps debug voice recognition accuracy

**Example Output:**
```
🗣️  YOU SAID: 'Hello BitBot, what time is it?' (confidence: 0.95)
🗣️  YOU SAID: 'How is the weather today?' (confidence: 0.87)
```

### 🎯 **Wake Word Detection Logging**
- Shows when your custom "Hey BitBot" wake word is detected
- Displays confidence scores for wake word detection
- Helps tune wake word sensitivity

**Example Output:**
```
🎯 WAKE WORD DETECTED: 'Hey BitBot' (confidence: 0.78)
```

## Debug Mode Benefits

### 🔧 **Troubleshooting**
- **No Microphone Input**: Check if volume levels appear
- **Poor Recognition**: Monitor confidence scores
- **Wake Word Issues**: Verify detection confidence
- **Audio Quality**: Monitor peak levels and volume

### 📊 **Performance Monitoring**
- Real-time audio processing statistics
- Voice recognition accuracy metrics
- Wake word detection sensitivity tuning
- Audio buffer and processing optimization

### 🎯 **Custom Wake Word Testing**
- Test your custom "Hey BitBot" model performance
- Monitor detection confidence to optimize sensitivity
- Verify model is working with your voice and accent

## Usage Examples

### Basic Debug Session
```bash
# Start BitBot with debug mode
python main.py start --debug

# You'll see:
🐛 DEBUG MODE ENABLED - Microphone input will be logged to screen
🎯 Say 'Hey BitBot' to activate BitBot

# Speak into microphone and watch debug output:
🎤 Microphone Input #    10: Volume: 0.0234, Peak: 0.5678, Size: 2048 samples
🎯 WAKE WORD DETECTED: 'Hey BitBot' (confidence: 0.82)
🗣️  YOU SAID: 'What is the capital of France?' (confidence: 0.91)
```

### Audio Level Testing
```bash
# Test microphone levels
python main.py start --debug

# Clap your hands or speak loudly - you should see:
🎤 Microphone Input #    30: Volume: 0.1234, Peak: 0.8901, Size: 2048 samples

# Whisper or speak quietly - you should see:
🎤 Microphone Input #    40: Volume: 0.0045, Peak: 0.1234, Size: 2048 samples
```

## Interpreting Debug Output

### Audio Volume Levels
- **Volume: 0.0000-0.0100**: Very quiet/background noise
- **Volume: 0.0100-0.0500**: Normal speaking volume
- **Volume: 0.0500-0.1000**: Loud speaking
- **Volume: 0.1000+**: Very loud/shouting

### Confidence Scores
- **0.90-1.00**: Excellent recognition
- **0.70-0.89**: Good recognition
- **0.50-0.69**: Fair recognition (might need adjustment)
- **0.00-0.49**: Poor recognition (check audio quality)

### Wake Word Detection
- **0.80-1.00**: Strong detection (reliable)
- **0.60-0.79**: Good detection (usually works)
- **0.40-0.59**: Weak detection (adjust sensitivity)
- **0.00-0.39**: Poor detection (check pronunciation/model)

## Normal vs Debug Mode

### Normal Mode (python main.py start)
```
🤖 BitBot is now listening for 'Hey BitBot'...
```

### Debug Mode (python main.py start --debug)
```
🐛 DEBUG MODE ENABLED - Microphone input will be logged to screen
🤖 BitBot is now listening for 'Hey BitBot'...
🎤 Microphone Input #    10: Volume: 0.0123, Peak: 0.4567, Size: 2048 samples
🎯 WAKE WORD DETECTED: 'Hey BitBot' (confidence: 0.78)
🗣️  YOU SAID: 'Hello there!' (confidence: 0.89)
```

## Tips for Using Debug Mode

1. **Use for Development**: Debug mode is perfect for testing and development
2. **Monitor Performance**: Watch for patterns in confidence scores
3. **Optimize Settings**: Use volume levels to adjust microphone sensitivity
4. **Test Custom Models**: Verify your custom "Hey BitBot" model performance
5. **Troubleshoot Issues**: Debug helps identify audio or recognition problems

## M1 Mac Optimizations

Debug mode is fully compatible with BitBot's M1 optimizations:
- ✅ **Gentle Resource Usage**: 20% CPU, 15% memory max
- ✅ **Large Audio Buffers**: 1000 chunks for smooth processing
- ✅ **Optimized Chunk Sizes**: 2048 samples for M1 efficiency
- ✅ **Custom Wake Word**: Uses your Hey_Bitbot.onnx model

## Disabling Debug Mode

Simply run BitBot normally without the --debug flag:

```bash
python main.py start
```

Debug mode only activates when explicitly requested with `--debug` or `-d`. 