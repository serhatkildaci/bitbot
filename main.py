#!/usr/bin/env python3
"""
BitBot - Local Real-Time AI Assistant
=====================================

Main entry point for BitBot with CLI interface and configuration options.
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import typer
    from loguru import logger
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

from bitbot.core.pipeline import BitBotCore
from bitbot.config.settings import BitBotConfig, HardwareTier


# Configure logging
def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    logger.remove()  # Remove default handler
    
    # Console logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "bitbot.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz"
    )


def setup_logging_quiet():
    """Set up minimal logging for chat mode."""
    logger.remove()  # Remove default handler
    
    # Only show ERROR level logs to keep chat clean
    logger.add(
        sys.stderr,
        level="ERROR",
        format="<red>{level}: {message}</red>",
        colorize=True
    )
    
    # Still log to file for debugging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "bitbot.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz"
    )


def load_environment():
    """Load environment variables from .env file."""
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")
    else:
        logger.info("No .env file found, using system environment")


def validate_environment():
    """Validate required environment variables and dependencies."""
    issues = []
    
    # Check for Porcupine access key
    if not os.getenv("PORCUPINE_ACCESS_KEY"):
        issues.append(
            "PORCUPINE_ACCESS_KEY not set. Get one from https://console.picovoice.ai/"
        )
    
    # Check if Ollama is running (basic check)
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            issues.append(
                "Ollama server not running on localhost:11434. Please start Ollama first."
            )
    except Exception:
        issues.append("Could not check Ollama server status")
    
    if issues:
        logger.warning("Environment validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    return True


async def run_bitbot(
    tier: Optional[HardwareTier] = None,
    verbose: bool = False,
    debug: bool = False,
    nowake: bool = False,
    validate_env: bool = True
):
    """Main BitBot execution function."""
    
    # Set up logging (quieter for no-wake mode)
    if nowake and not debug:
        # In no-wake mode, suppress most logs for clean chat interface
        setup_logging_quiet()
    else:
        setup_logging(verbose)
    
    # Load environment
    load_environment()
    
    # Validate environment
    if validate_env and not validate_environment():
        logger.error("Environment validation failed. Use --skip-validation to ignore.")
        return False
    
    # Create configuration
    config = BitBotConfig(tier=tier)
    if not (nowake and not debug):
        logger.info(f"BitBot Configuration: {config.get_config_summary()}")
    
    # Initialize BitBot core with mode settings
    core = BitBotCore(config, debug_mode=debug, nowake_mode=nowake)
    
    try:
        if not (nowake and not debug):
            logger.info("🚀 Starting BitBot initialization...")
        
        if not await core.initialize():
            logger.error("❌ BitBot initialization failed")
            return False
        
        if not (nowake and not debug):
            logger.info("✅ BitBot initialized successfully!")
            logger.info("🎤 Starting BitBot assistant...")
        elif nowake:
            # Clean startup for chat mode
            print("🤖 BitBot Chat Mode")
            print("================")
            print("🎧 Listening...")
        
        # Run BitBot
        await core.run_forever()
        
    except KeyboardInterrupt:
        if not (nowake and not debug):
            logger.info("👋 BitBot shutdown requested")
        else:
            print("\n👋 Chat session ended")
    except Exception as e:
        logger.error(f"💥 BitBot error: {e}")
        return False
    finally:
        if not (nowake and not debug):
            logger.info("🧹 Cleaning up...")
        await core.cleanup()
        if not (nowake and not debug):
            logger.info("✨ BitBot shutdown complete")
    
    return True


# CLI Interface using Typer
app = typer.Typer(
    name="bitbot",
    help="BitBot - Local Real-Time AI Assistant",
    add_completion=False
)


@app.command()
def start(
    tier: Optional[str] = typer.Option(
        None,
        "--tier", "-t",
        help="Hardware tier (small/medium/large). Auto-detected if not specified."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug mode with microphone input logging"
    ),
    nowake: bool = typer.Option(
        False,
        "--nowake",
        help="Start directly in conversation mode without wake word detection (production mode)"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip environment validation"
    )
):
    """Start BitBot assistant."""
    
    # Parse tier
    hardware_tier = None
    if tier:
        tier_map = {
            "small": HardwareTier.SMALL,
            "s": HardwareTier.SMALL,
            "medium": HardwareTier.MEDIUM,
            "m": HardwareTier.MEDIUM,
            "large": HardwareTier.LARGE,
            "l": HardwareTier.LARGE
        }
        hardware_tier = tier_map.get(tier.lower())
        if not hardware_tier:
            typer.echo(f"Invalid tier: {tier}. Use: small, medium, or large")
            raise typer.Exit(1)
    
    # Run BitBot
    try:
        success = asyncio.run(run_bitbot(
            tier=hardware_tier,
            verbose=verbose,
            debug=debug,
            nowake=nowake,
            validate_env=not skip_validation
        ))
        if not success:
            raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\n👋 BitBot stopped")
    except Exception as e:
        typer.echo(f"💥 Error: {e}")
        raise typer.Exit(1)


@app.command()
def config():
    """Show current BitBot configuration."""
    load_environment()
    
    config = BitBotConfig()
    summary = config.get_config_summary()
    
    typer.echo("🤖 BitBot Configuration:")
    typer.echo(f"  Hardware Tier: {summary['tier']}")
    typer.echo(f"  STT Model: {summary['stt_model']}")
    typer.echo(f"  LLM Model: {summary['llm_model']}")
    typer.echo(f"  TTS Engine: {summary['tts_engine']}")
    typer.echo(f"  Audio Sample Rate: {summary['audio_sample_rate']}Hz")
    typer.echo(f"  Wake Word: '{summary['wake_word']}'")


@app.command()
def setup():
    """Setup guide for BitBot."""
    typer.echo("🛠️  BitBot Setup Guide")
    typer.echo("")
    
    typer.echo("1. Install Dependencies:")
    typer.echo("   pip install -r requirements.txt")
    typer.echo("")
    
    typer.echo("2. Install and Start Ollama:")
    typer.echo("   Download from: https://ollama.com/")
    typer.echo("   ollama pull mistral:7b-instruct")
    typer.echo("   ollama serve")
    typer.echo("")
    
    typer.echo("3. Get Porcupine Access Key:")
    typer.echo("   Visit: https://console.picovoice.ai/")
    typer.echo("   Add to .env file: PORCUPINE_ACCESS_KEY=your_key_here")
    typer.echo("")
    
    typer.echo("4. Start BitBot:")
    typer.echo("   python main.py start")
    typer.echo("")
    
    typer.echo("5. Say 'Hey BitBot' to start a conversation!")


@app.command()
def test():
    """Test BitBot components."""
    setup_logging(verbose=True)
    load_environment()
    
    async def test_components():
        """Test component initialization."""
        config = BitBotConfig()
        
        typer.echo("🧪 Testing BitBot components...")
        
        # Test configuration
        typer.echo("✅ Configuration loaded")
        
        # Test hardware detection
        typer.echo(f"✅ Hardware tier detected: {config.tier.value}")
        
        # Test environment
        if validate_environment():
            typer.echo("✅ Environment validation passed")
        else:
            typer.echo("❌ Environment validation failed")
            return False
        
        # Test component initialization
        core = BitBotCore(config)
        
        typer.echo("🔧 Testing component initialization...")
        if await core.initialize():
            typer.echo("✅ All components initialized successfully")
            await core.cleanup()
            return True
        else:
            typer.echo("❌ Component initialization failed")
            return False
    
    try:
        success = asyncio.run(test_components())
        if success:
            typer.echo("🎉 All tests passed! BitBot is ready to run.")
        else:
            typer.echo("💥 Some tests failed. Check the logs.")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"💥 Test error: {e}")
        raise typer.Exit(1)


@app.command()
def chat():
    """Start text-based chat interface with BitBot."""
    
    async def run_chat():
        """Run the chat interface."""
        from bitbot.cli.chat_interface import ChatInterface
        from bitbot.config.settings import BitBotConfig
        
        setup_logging(verbose=False)
        load_environment()
        
        config = BitBotConfig()
        chat_interface = ChatInterface(config)
        
        typer.echo("🤖 Starting BitBot chat interface...")
        
        if await chat_interface.initialize():
            await chat_interface.start_chat()
        else:
            typer.echo("❌ Failed to initialize chat interface")
            raise typer.Exit(1)
        
        await chat_interface.cleanup()
    
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        typer.echo("\n👋 Chat session ended")
    except Exception as e:
        typer.echo(f"💥 Chat error: {e}")
        raise typer.Exit(1)


@app.command()
def webclient(
    host: str = typer.Option(
        "127.0.0.1",
        "--host", "-h",
        help="Host to bind the web server to"
    ),
    port: int = typer.Option(
        8080,
        "--port", "-p",
        help="Port to bind the web server to"
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development"
    )
):
    """Start the BitBot web client interface."""
    from bitbot.web_client.server import start_web_server
    
    typer.echo(f"🌐 Starting BitBot Web Client at http://{host}:{port}")
    typer.echo("Press Ctrl+C to stop the server")
    
    try:
        start_web_server(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        typer.echo("\n👋 BitBot Web Client stopped")
    except Exception as e:
        typer.echo(f"❌ Error starting web client: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show BitBot version information."""
    typer.echo("🤖 BitBot v1.0.0")
    typer.echo("Local Real-Time AI Assistant")


if __name__ == "__main__":
    # Handle direct execution
    if len(sys.argv) == 1:
        # No arguments, show help
        app()
    else:
        # Run with CLI
        app() 