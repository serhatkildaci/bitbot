"""
BitBot Model Manager
===================

Automatically downloads and manages models based on hardware tier.
Includes resource monitoring to ensure gentle system usage.
"""

import asyncio
import psutil
import time
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    size_gb: float
    description: str
    recommended_ram_gb: float


class ResourceMonitor:
    """Monitors system resources to ensure gentle usage."""
    
    def __init__(self, max_cpu_percent: float = 30.0, max_memory_percent: float = 25.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self._monitoring = False
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring = True
        asyncio.create_task(self._monitor_loop())
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                if cpu_percent > self.max_cpu_percent:
                    logger.warning(f"CPU usage high: {cpu_percent:.1f}% (max: {self.max_cpu_percent}%)")
                    await asyncio.sleep(2)  # Cool down
                    
                if memory_info.percent > self.max_memory_percent:
                    logger.warning(f"Memory usage high: {memory_info.percent:.1f}% (max: {self.max_memory_percent}%)")
                    await asyncio.sleep(1)  # Brief pause
                    
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }


class ModelManager:
    """Manages model downloading and selection based on hardware tier."""
    
    # Model catalog with sizes and requirements
    MODEL_CATALOG = {
        # LLM Models
        "gemma:2b": ModelInfo("gemma:2b", 1.4, "Ultra-light Google model", 4.0),
        "llama3.2:3b": ModelInfo("llama3.2:3b", 2.0, "Efficient 3B parameter model", 6.0),
        "llama3.1:8b": ModelInfo("llama3.1:8b", 4.7, "Standard 8B parameter model", 12.0),
        "mistral:7b": ModelInfo("mistral:7b", 4.1, "Efficient 7B parameter model", 10.0),
        
        # Whisper Models (approximate sizes)
        "tiny": ModelInfo("tiny", 0.04, "Fastest, least accurate", 1.0),
        "tiny.en": ModelInfo("tiny.en", 0.04, "Fastest English-only", 1.0),
        "small.en": ModelInfo("small.en", 0.24, "Good balance for English", 2.0),
        "base.en": ModelInfo("base.en", 0.29, "Better accuracy English", 3.0),
    }
    
    def __init__(self, config):
        self.config = config
        self.resource_monitor = ResourceMonitor(
            max_cpu_percent=config.resources.max_cpu_percent,
            max_memory_percent=config.resources.max_memory_percent
        )
        
    async def ensure_models_available(self) -> bool:
        """Ensure required models are downloaded and available."""
        logger.info("🔄 Checking model availability...")
        
        try:
            # Start resource monitoring
            await self.resource_monitor.start_monitoring()
            
            # Check current resources
            usage = self.resource_monitor.get_current_usage()
            logger.info(f"Current usage - CPU: {usage['cpu_percent']:.1f}%, Memory: {usage['memory_percent']:.1f}%")
            
            # Get required models
            models_needed = self.config.auto_download_models()
            
            # Check and download LLM model
            llm_available = await self._ensure_llm_model(models_needed["llm_model"])
            
            # Check STT model (Whisper models are auto-downloaded by faster-whisper)
            stt_available = await self._ensure_stt_model(models_needed["stt_model"])
            
            return llm_available and stt_available
            
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False
        finally:
            self.resource_monitor.stop_monitoring()
    
    async def _ensure_llm_model(self, model_name: str) -> bool:
        """Ensure LLM model is available."""
        if not HAS_OLLAMA:
            logger.error("Ollama not available - cannot download LLM models")
            return False
            
        try:
            # Check if model exists
            existing_models = ollama.list()
            model_exists = any(model_name in model['name'] for model in existing_models.get('models', []))
            
            if model_exists:
                logger.info(f"✅ LLM model '{model_name}' already available")
                return True
            
            # Check if we have enough space
            model_info = self.MODEL_CATALOG.get(model_name)
            if model_info:
                usage = self.resource_monitor.get_current_usage()
                if usage['available_memory_gb'] < model_info.recommended_ram_gb:
                    logger.error(f"❌ Insufficient memory for {model_name} (need {model_info.recommended_ram_gb}GB)")
                    
                    # Suggest a smaller model
                    alternative = self._suggest_smaller_model(model_name)
                    if alternative:
                        logger.info(f"💡 Trying smaller alternative: {alternative}")
                        return await self._ensure_llm_model(alternative)
                    return False
            
            # Download the model
            logger.info(f"📥 Downloading LLM model '{model_name}'...")
            logger.info(f"💾 Size: ~{model_info.size_gb if model_info else 'unknown'}GB")
            
            try:
                # Download with progress monitoring
                response = ollama.pull(model_name, stream=True)
                for chunk in response:
                    if 'status' in chunk:
                        if chunk['status'] == 'downloading':
                            # Log progress every so often
                            if 'completed' in chunk and 'total' in chunk:
                                progress = (chunk['completed'] / chunk['total']) * 100
                                if int(progress) % 20 == 0:  # Log every 20%
                                    logger.info(f"📥 Download progress: {progress:.1f}%")
                
                logger.info(f"✅ Successfully downloaded '{model_name}'")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to download '{model_name}': {e}")
                
                # Try a smaller alternative
                alternative = self._suggest_smaller_model(model_name)
                if alternative and alternative != model_name:
                    logger.info(f"💡 Trying smaller alternative: {alternative}")
                    return await self._ensure_llm_model(alternative)
                    
                return False
                
        except Exception as e:
            logger.error(f"LLM model check failed: {e}")
            return False
    
    async def _ensure_stt_model(self, model_name: str) -> bool:
        """Ensure STT (Whisper) model is available."""
        try:
            # Whisper models are automatically downloaded by faster-whisper
            # We just need to verify the model name is valid
            valid_models = ["tiny", "tiny.en", "small", "small.en", "base", "base.en", "medium", "medium.en"]
            
            if model_name in valid_models:
                logger.info(f"✅ STT model '{model_name}' will be auto-downloaded when needed")
                return True
            else:
                logger.warning(f"⚠️ Unknown STT model '{model_name}', falling back to 'tiny.en'")
                return True
                
        except Exception as e:
            logger.error(f"STT model check failed: {e}")
            return False
    
    def _suggest_smaller_model(self, current_model: str) -> Optional[str]:
        """Suggest a smaller model alternative."""
        
        # Model size hierarchy (smallest to largest)
        llm_hierarchy = ["gemma:2b", "llama3.2:3b", "mistral:7b", "llama3.1:8b"]
        stt_hierarchy = ["tiny", "tiny.en", "small.en", "base.en"]
        
        try:
            if current_model in llm_hierarchy:
                current_index = llm_hierarchy.index(current_model)
                if current_index > 0:
                    return llm_hierarchy[current_index - 1]
            elif current_model in stt_hierarchy:
                current_index = stt_hierarchy.index(current_model)
                if current_index > 0:
                    return stt_hierarchy[current_index - 1]
                    
        except ValueError:
            pass
            
        return None
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Get model recommendations for current hardware."""
        usage = self.resource_monitor.get_current_usage()
        available_gb = usage['available_memory_gb']
        
        recommendations = {
            "status": "analysis_complete",
            "available_memory_gb": available_gb,
            "recommended_models": {},
            "warnings": []
        }
        
        # Recommend models based on available memory
        if available_gb < 4:
            recommendations["recommended_models"] = {
                "llm": "gemma:2b",
                "stt": "tiny.en"
            }
            recommendations["warnings"].append("Very limited memory - using ultra-light models")
        elif available_gb < 8:
            recommendations["recommended_models"] = {
                "llm": "llama3.2:3b",
                "stt": "tiny.en"
            }
            recommendations["warnings"].append("Limited memory - using efficient models")
        elif available_gb < 16:
            recommendations["recommended_models"] = {
                "llm": "llama3.2:3b",
                "stt": "small.en"
            }
        else:
            recommendations["recommended_models"] = {
                "llm": "llama3.1:8b",
                "stt": "small.en"
            }
            
        return recommendations 