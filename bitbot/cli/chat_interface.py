"""
BitBot CLI Chat Interface
=========================

Text-based chat interface for BitBot that provides an alternative
to voice interaction and allows for testing and debugging.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from loguru import logger
from ..config.settings import BitBotConfig
from ..llm.ollama_client import LLMEngine, LLMResponse


class SimpleChatInterface:
    """Simple text-based chat interface without rich formatting."""
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.conversation_history: List[Dict[str, Any]] = []
        
    async def start_chat(self):
        """Start the chat interface."""
        print("=" * 60)
        print("BitBot Text Chat Interface")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("BitBot: Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'clear':
                    self.llm_engine.clear_conversation()
                    self.conversation_history.clear()
                    print("BitBot: Conversation history cleared.")
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Process with LLM
                print("BitBot: [thinking...]", end="", flush=True)
                
                response = await self.llm_engine.process_user_input(user_input)
                
                print("\r" + " " * 20 + "\r", end="")  # Clear thinking message
                
                if response:
                    print(f"BitBot: {response.content}")
                    
                    # Add to history
                    self.conversation_history.append({
                        "timestamp": datetime.now(),
                        "user": user_input,
                        "assistant": response.content
                    })
                else:
                    print("BitBot: I'm sorry, I couldn't process that request.")
                
                print()
                
            except KeyboardInterrupt:
                print("\nBitBot: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat interface error: {e}")
                print("BitBot: Sorry, I encountered an error. Please try again.")
    
    def _show_help(self):
        """Show help information."""
        print("\nBitBot Chat Commands:")
        print("  help     - Show this help message")
        print("  clear    - Clear conversation history")
        print("  history  - Show conversation history")
        print("  quit     - Exit the chat")
        print()
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("BitBot: No conversation history yet.")
            return
        
        print("\nConversation History:")
        print("-" * 40)
        
        for i, entry in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            timestamp = entry["timestamp"].strftime("%H:%M:%S")
            print(f"{i}. [{timestamp}]")
            print(f"   You: {entry['user']}")
            print(f"   BitBot: {entry['assistant'][:100]}{'...' if len(entry['assistant']) > 100 else ''}")
            print()


class RichChatInterface:
    """Rich-formatted chat interface using the rich library."""
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.console = Console()
        self.conversation_history: List[Dict[str, Any]] = []
        
    async def start_chat(self):
        """Start the rich chat interface."""
        self.console.print(Panel.fit(
            "🤖 BitBot Text Chat Interface\n"
            "Type 'help' for commands, 'quit' to exit",
            style="bold blue"
        ))
        
        while True:
            try:
                # Get user input with rich prompt
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.console.print("[bold blue]BitBot:[/bold blue] Goodbye! 👋")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'clear':
                    self.llm_engine.clear_conversation()
                    self.conversation_history.clear()
                    self.console.print("[bold blue]BitBot:[/bold blue] Conversation history cleared. 🧹")
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                # Process with LLM
                with self.console.status("[bold blue]BitBot is thinking...", spinner="dots"):
                    response = await self.llm_engine.process_user_input(user_input)
                
                if response:
                    # Display response with rich formatting
                    self.console.print(f"[bold blue]BitBot:[/bold blue] {response.content}")
                    
                    # Add to history
                    self.conversation_history.append({
                        "timestamp": datetime.now(),
                        "user": user_input,
                        "assistant": response.content
                    })
                else:
                    self.console.print("[bold blue]BitBot:[/bold blue] [red]I'm sorry, I couldn't process that request.[/red]")
                
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print("\n[bold blue]BitBot:[/bold blue] Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Rich chat interface error: {e}")
                self.console.print("[bold blue]BitBot:[/bold blue] [red]Sorry, I encountered an error. Please try again.[/red]")
    
    def _show_help(self):
        """Show help information with rich formatting."""
        help_table = Table(title="BitBot Chat Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("help", "Show this help message")
        help_table.add_row("clear", "Clear conversation history")
        help_table.add_row("history", "Show recent conversation history")
        help_table.add_row("status", "Show BitBot status")
        help_table.add_row("quit", "Exit the chat")
        
        self.console.print(help_table)
        self.console.print()
    
    def _show_history(self):
        """Show conversation history with rich formatting."""
        if not self.conversation_history:
            self.console.print("[bold blue]BitBot:[/bold blue] No conversation history yet.")
            return
        
        history_table = Table(title="Recent Conversation History")
        history_table.add_column("Time", style="dim")
        history_table.add_column("You", style="green", width=30)
        history_table.add_column("BitBot", style="blue", width=40)
        
        for entry in self.conversation_history[-5:]:  # Show last 5
            timestamp = entry["timestamp"].strftime("%H:%M:%S")
            user_text = entry["user"][:27] + "..." if len(entry["user"]) > 30 else entry["user"]
            bot_text = entry["assistant"][:37] + "..." if len(entry["assistant"]) > 40 else entry["assistant"]
            
            history_table.add_row(timestamp, user_text, bot_text)
        
        self.console.print(history_table)
        self.console.print()
    
    def _show_status(self):
        """Show BitBot status."""
        summary = self.llm_engine.get_conversation_summary()
        
        status_table = Table(title="BitBot Status")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="white")
        
        status_table.add_row("Model", summary.get("model", "Unknown"))
        status_table.add_row("Messages", str(summary.get("message_count", 0)))
        status_table.add_row("System Prompt", "✅" if summary.get("system_prompt_set") else "❌")
        status_table.add_row("Session Messages", str(len(self.conversation_history)))
        
        self.console.print(status_table)
        self.console.print()


class ChatInterface:
    """Main chat interface that chooses between simple and rich modes."""
    
    def __init__(self, config: BitBotConfig):
        self.config = config
        self.llm_engine: Optional[LLMEngine] = None
        self.interface: Optional[object] = None
        
    async def initialize(self) -> bool:
        """Initialize the chat interface."""
        try:
            # Initialize LLM engine
            self.llm_engine = LLMEngine(self.config.llm)
            success = await self.llm_engine.initialize()
            
            if not success:
                logger.error("Failed to initialize LLM engine for chat")
                return False
            
            # Choose interface based on rich availability
            if RICH_AVAILABLE:
                self.interface = RichChatInterface(self.llm_engine)
                logger.info("Initialized rich chat interface")
            else:
                self.interface = SimpleChatInterface(self.llm_engine)
                logger.info("Initialized simple chat interface")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat interface: {e}")
            return False
    
    async def start_chat(self):
        """Start the chat interface."""
        if not self.interface:
            print("Chat interface not initialized")
            return
        
        await self.interface.start_chat()
    
    async def cleanup(self):
        """Clean up chat interface."""
        if self.llm_engine:
            await self.llm_engine.cleanup()
            self.llm_engine = None
        
        logger.info("Chat interface cleaned up")


async def main():
    """Main function for standalone chat interface."""
    from ..config.settings import BitBotConfig
    
    config = BitBotConfig()
    chat = ChatInterface(config)
    
    if await chat.initialize():
        await chat.start_chat()
    else:
        print("Failed to initialize chat interface")
    
    await chat.cleanup()


if __name__ == "__main__":
    asyncio.run(main())