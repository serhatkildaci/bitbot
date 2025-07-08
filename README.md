# BitBot

**Local Real-Time AI Assistant with Speech-to-Text, Large Language Model, and Text-to-Speech Integration**

## Overview

BitBot is a local, real-time, audio-enabled AI assistant designed to run entirely on consumer hardware without requiring cloud services or internet connectivity for core functionality. The project implements a streaming STT → LLM → TTS pipeline with wake word detection, optimized for responsive conversational interactions on daily-driver laptops and desktop computers.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** for local LLM serving
3. **Porcupine Access Key** for wake word detection

### Installation

1. **Clone and navigate to project:**
   ```bash
   git clone <repository-url>
   cd bitbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama:**
   ```bash
   # Download from https://ollama.com/
   ollama pull mistral:7b-instruct
   ollama serve
   ```

4. **Get Porcupine access key:**
   - Visit https://console.picovoice.ai/
   - Create account and get access key
   - Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env and add your PORCUPINE_ACCESS_KEY
   ```

5. **Test setup:**
   ```bash
   python main.py test
   ```

6. **Start BitBot:**
   ```bash
   python main.py start
   ```

## 🎙️ Usage

1. **Start BitBot:**
   ```bash
   python main.py start
   ```

2. **Wake BitBot:**
   - Say "Hey BitBot" (or "Hey Google" in MVP)
   - BitBot will respond with "Yes?" and start listening

3. **Have a conversation:**
   - Speak naturally to BitBot
   - BitBot will process your speech, generate responses, and speak back
   - Conversation continues until you say goodbye or timeout

4. **End conversation:**
   - Say "goodbye", "bye", "exit", or "thank you"

## 📋 Commands

### Main Commands
```bash
python main.py start          # Start BitBot assistant
python main.py config         # Show current configuration
python main.py test           # Test all components
python main.py setup          # Show setup guide
python main.py version        # Show version info
```

### Options
```bash
python main.py start --tier medium    # Force hardware tier
python main.py start --verbose        # Verbose logging
python main.py start --skip-validation # Skip environment validation
```

## ⚙️ Configuration

BitBot automatically detects your hardware and configures accordingly:

### Hardware Tiers

- **BitBotS (Small)**: Standard PCs, older laptops
  - STT: Whisper tiny.en
  - LLM: Mistral 7B (Q4_K_M)
  - TTS: Piper TTS

- **BitBotM (Medium)**: Modern laptops, 4GB+ VRAM
  - STT: Whisper small.en
  - LLM: Llama 3.1 8B (Q4_K_M)
  - TTS: Kyutai TTS

- **BitBotL (Large)**: High-end PCs, 8GB+ VRAM
  - STT: Whisper large-v3
  - LLM: Llama 3.1 8B (Q5_K_M)
  - TTS: Kyutai TTS

### Environment Variables

Edit `.env` file to customize:

```bash
# Required
PORCUPINE_ACCESS_KEY=your_key_here

# Optional
OLLAMA_BASE_URL=http://localhost:11434
AUDIO_INPUT_DEVICE=0
AUDIO_OUTPUT_DEVICE=0
LOG_LEVEL=INFO
```

## 🏗️ Architecture

BitBot follows a modular, asynchronous architecture:

### Core Components

- **Audio Manager**: Cross-platform audio I/O with sounddevice
- **STT Engine**: Faster Whisper for speech recognition
- **LLM Engine**: Ollama client with OpenAI-compatible API
- **TTS Engine**: RealtimeTTS with multiple backends
- **Wake Word**: Picovoice Porcupine for trigger detection
- **Pipeline**: Asyncio orchestration of STT → LLM → TTS

### Data Flow

```
Audio Input → Wake Word Detection → STT → LLM → TTS → Audio Output
     ↑                ↓                              ↓
   Microphone    "Hey BitBot"                   Speakers
```

## 🛠️ Development

### Project Structure

```
bitbot/
├── bitbot/                 # Main package
│   ├── audio/             # Audio I/O management
│   ├── stt/               # Speech-to-text engines
│   ├── llm/               # LLM integration
│   ├── tts/               # Text-to-speech engines
│   ├── wake_word/         # Wake word detection
│   ├── core/              # Pipeline orchestration
│   └── config/            # Configuration management
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Key Technologies

- **Python asyncio**: Non-blocking I/O and pipeline orchestration
- **Faster Whisper**: High-performance STT inference
- **Ollama**: Local LLM deployment platform
- **RealtimeTTS**: Low-latency text-to-speech synthesis
- **Picovoice Porcupine**: Accurate wake word detection
- **sounddevice**: Cross-platform audio handling

## 🚨 Troubleshooting

### Common Issues

1. **Ollama not running:**
   ```bash
   ollama serve
   ```

2. **Porcupine access key missing:**
   - Get key from https://console.picovoice.ai/
   - Add to `.env` file

3. **Audio device issues:**
   ```bash
   python main.py test  # Check audio devices
   ```

4. **Model download issues:**
   ```bash
   ollama pull mistral:7b-instruct
   ```

5. **Permission issues on Linux:**
   ```bash
   sudo usermod -a -G audio $USER  # Audio permissions
   ```

### Logs

Check logs in `logs/bitbot.log` for detailed debugging information.

## 🎯 MVP Limitations

This MVP includes core functionality with some limitations:

- **Wake Word**: Uses "Hey Google" instead of custom "Hey BitBot" (requires training)
- **TTS Backends**: Limited voice options in MVP
- **Tool Integration**: MCP and RAG features planned for future releases
- **Model Optimization**: Full quantization optimizations in progress

## 🔮 Roadmap

### Next Features
- [ ] Custom "Hey BitBot" wake word training
- [ ] Model Context Protocol (MCP) tool integration
- [ ] LanceDB vector storage for RAG
- [ ] Advanced TTS voice selection
- [ ] Streaming LLM responses
- [ ] Multi-language support
- [ ] Web interface for configuration

### Hardware Optimization
- [ ] Apple Silicon optimizations
- [ ] GPU acceleration improvements
- [ ] Memory usage optimization
- [ ] Real-time performance tuning

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This project is in early development. Contribution guidelines and development setup instructions will be provided as the codebase matures.

## 💡 Support

For issues, questions, or contributions:
- Check logs in `logs/bitbot.log`
- Run `python main.py test` to diagnose issues
- Review environment variables in `.env`

---

**BitBot**: Your local AI assistant, private by design. 🤖