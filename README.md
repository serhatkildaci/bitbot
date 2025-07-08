# BitBot

**Local Real-Time AI Assistant with Speech-to-Text, Large Language Model, and Text-to-Speech Integration**

## Overview

BitBot is a local, real-time, audio-enabled AI assistant designed to run entirely on consumer hardware without requiring cloud services or internet connectivity for core functionality. The project implements a streaming STT → LLM → TTS pipeline with wake word detection, optimized for responsive conversational interactions on daily-driver laptops and desktop computers.

## Project Status

**Development Phase: Initial Planning & Architecture**

This repository is in early development. The codebase, installation procedures and  implementations are not yet available.

## Architecture Overview

BitBot follows a modular, asynchronous architecture built on three core principles:

1. **Asynchronous Streaming Pipeline**: Real-time data flow between speech-to-text, language model processing, and text-to-speech synthesis
2. **Centralized Orchestration**: Python asyncio-based management of all I/O-bound operations for non-blocking execution
3. **Aggressive Model Quantization**: Optimized model formats (GGUF Q4_K_M) to ensure performance on consumer hardware

## Planned Technology Stack

### Core Components

- **Wake Word Detection**: Picovoice Porcupine for "Hey BitBot!" trigger phrase
- **Audio I/O**: sounddevice library for cross-platform audio handling
- **Speech-to-Text**: Faster Whisper with streaming implementation
- **LLM Platform**: Ollama for local model deployment and serving
- **Text-to-Speech**: RealtimeTTS with Kyutai TTS and Piper TTS backends
- **Tool Integration**: Model Context Protocol (MCP) with FastMCP framework
- **Vector Storage**: LanceDB for local retrieval-augmented generation

### Model Selection

**Language Models**:
- Mistral 7B Instruct (default)
- Llama 3.1 8B Instruct (alternative option)

**Speech Models**:
- Whisper models (tiny.en to large, depending on hardware tier)
- Quantized TTS models via RealtimeTTS backends

## Target Hardware Configurations

### BitBotS (Standard)
- Target: Standard PCs, older laptops

### BitBotM (Medium) 
- Target: Capable PCs, modern laptops (4GB VRAM)

### BitBotL (Large)
- Target: High-end PCs, Apple Silicon (>8GB VRAM)

## Design Principles

- **Local-First**: All processing occurs on-device without cloud dependencies
- **Real-Time Performance**: Sub-second response times for conversational flow
- **Modular Architecture**: Pluggable components for different hardware configurations
- **Open Source**: MIT-licensed with community contribution support
- **Privacy-Focused**: No data transmission to external services

## Contributing

This project is in early development. Contribution guidelines, code standards, and development setup instructions will be provided as the codebase matures.

## License

MIT License - see LICENSE file for details.