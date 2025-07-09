"""
BitBot Web Client Server
========================

FastAPI server providing a web interface for BitBot with
real-time chat, status monitoring, and pipeline control.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from loguru import logger
from ..config.settings import BitBotConfig, HardwareTier
from ..core.pipeline import BitBotCore
from ..llm.ollama_client import LLMEngine, LLMMessage


# Pydantic models for API
class ChatMessage(BaseModel):
    content: str
    timestamp: Optional[str] = None


class ChatResponse(BaseModel):
    content: str
    timestamp: str
    success: bool


class SystemStatus(BaseModel):
    pipeline_state: str
    uptime: float
    stats: Dict[str, Any]
    config_summary: Dict[str, str]


class ConfigUpdate(BaseModel):
    tier: Optional[str] = None
    audio_settings: Optional[Dict[str, Any]] = None


# Global state
bitbot_core: Optional[BitBotCore] = None
connected_clients: List[WebSocket] = []
chat_history: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage BitBot lifecycle with FastAPI app."""
    global bitbot_core
    
    # Startup
    logger.info("Starting BitBot Web Server...")
    
    # Initialize BitBot if auto-start is enabled
    config = BitBotConfig()
    bitbot_core = BitBotCore(config)
    
    # Note: We don't auto-initialize here to allow user control
    logger.info("BitBot Web Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BitBot Web Server...")
    if bitbot_core:
        await bitbot_core.cleanup()
    logger.info("BitBot Web Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="BitBot Web Client",
    description="Web interface for BitBot AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing this file
current_dir = Path(__file__).parent
static_dir = current_dir / "static"

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main web interface."""
    html_file = current_dir / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    
    # Fallback terminal HTML if static files don't exist
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BitBot Terminal Interface</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: 'Courier New', monospace; 
                background: #0a0a0a; 
                color: #00ff00; 
                margin: 0; 
                padding: 20px; 
                min-height: 100vh;
            }
            .terminal { 
                background: #111; 
                border: 2px solid #00cc00; 
                padding: 20px; 
                border-radius: 8px;
                max-width: 1000px;
                margin: 0 auto;
            }
            .header { 
                color: #66ff66; 
                font-weight: bold; 
                margin-bottom: 20px; 
                text-align: center;
                font-size: 18px;
            }
            .output { 
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #00cc00; 
                background: rgba(0,0,0,0.5); 
                padding: 10px; 
                margin-bottom: 20px;
                font-size: 14px;
            }
            .input-line { 
                display: flex; 
                align-items: center; 
                gap: 8px;
            }
            .prompt { 
                color: #66ff66; 
                font-weight: bold; 
            }
            input { 
                flex: 1; 
                background: transparent; 
                border: none; 
                color: #00ff00; 
                font-family: inherit; 
                font-size: 14px;
                outline: none;
            }
            .status { 
                color: #ffaa00; 
                margin-bottom: 10px; 
            }
            .error { color: #ff0000; }
            .success { color: #00ff00; font-weight: bold; }
            .info { color: #0088ff; }
        </style>
    </head>
    <body>
        <div class="terminal">
            <div class="header">🤖 BITBOT TERMINAL INTERFACE v1.0.0</div>
            <div class="status" id="status">SYSTEM: OFFLINE | WS: DISCONNECTED</div>
            <div class="output" id="output">
                <div class="success">[SYSTEM] BitBot Terminal Interface initialized</div>
                <div class="info">[INFO] Type commands or chat messages</div>
                <div class="info">[INFO] Commands: /init, /start, /stop, /status, /help</div>
            </div>
            <div class="input-line">
                <span class="prompt">user@bitbot:~$</span>
                <input type="text" id="input" placeholder="Enter command or message..." autofocus>
            </div>
        </div>
        
        <script>
            let socket = null;
            const output = document.getElementById('output');
            const input = document.getElementById('input');
            const status = document.getElementById('status');
            
            function addLine(type, message) {
                const line = document.createElement('div');
                line.className = type;
                line.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
                output.appendChild(line);
                output.scrollTop = output.scrollHeight;
            }
            
            function connectWS() {
                socket = new WebSocket(`ws://${window.location.host}/ws`);
                socket.onopen = () => {
                    addLine('success', 'WebSocket connected');
                    status.textContent = 'SYSTEM: ONLINE | WS: CONNECTED';
                };
                socket.onmessage = (e) => {
                    const data = JSON.parse(e.data);
                    if (data.type === 'chat') {
                        addLine('', `[BITBOT] ${data.content}`);
                    }
                };
                socket.onclose = () => {
                    addLine('error', 'WebSocket disconnected');
                    status.textContent = 'SYSTEM: OFFLINE | WS: DISCONNECTED';
                    setTimeout(connectWS, 3000);
                };
            }
            
            input.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter') {
                    const cmd = input.value.trim();
                    if (!cmd) return;
                    
                    addLine('', `$ ${cmd}`);
                    input.value = '';
                    
                    if (cmd.startsWith('/')) {
                        const [command] = cmd.slice(1).split(' ');
                        switch (command) {
                            case 'init':
                                const initResp = await fetch('/api/initialize', {method: 'POST'});
                                const initResult = await initResp.json();
                                addLine(initResult.success ? 'success' : 'error', 
                                       initResult.success ? 'System initialized' : 'Init failed');
                                break;
                            case 'start':
                                const startResp = await fetch('/api/start', {method: 'POST'});
                                const startResult = await startResp.json();
                                addLine(startResult.success ? 'success' : 'error',
                                       startResult.success ? 'Pipeline started' : 'Start failed');
                                break;
                            case 'stop':
                                const stopResp = await fetch('/api/stop', {method: 'POST'});
                                const stopResult = await stopResp.json();
                                addLine(stopResult.success ? 'success' : 'error',
                                       stopResult.success ? 'Pipeline stopped' : 'Stop failed');
                                break;
                            case 'status':
                                const statusResp = await fetch('/api/status');
                                const statusResult = await statusResp.json();
                                addLine('info', `Pipeline: ${statusResult.pipeline_state} | Uptime: ${Math.round(statusResult.uptime)}s`);
                                break;
                            case 'help':
                                addLine('info', 'Commands: /init, /start, /stop, /status, /help');
                                addLine('info', 'Or type messages to chat with BitBot');
                                break;
                            default:
                                addLine('error', `Unknown command: ${command}`);
                        }
                    } else {
                        // Chat message
                        if (socket && socket.readyState === WebSocket.OPEN) {
                            socket.send(JSON.stringify({type: 'chat', content: cmd}));
                        } else {
                            addLine('error', 'WebSocket not connected');
                        }
                    }
                }
            });
            
            connectWS();
        </script>
    </body>
    </html>
    """)


@app.get("/api/status")
async def get_status() -> SystemStatus:
    """Get current system status."""
    global bitbot_core
    
    if not bitbot_core:
        return SystemStatus(
            pipeline_state="not_initialized",
            uptime=0.0,
            stats={},
            config_summary={}
        )
    
    status = bitbot_core.get_status()
    return SystemStatus(
        pipeline_state=status["pipeline_state"],
        uptime=status["uptime"],
        stats=status["stats"],
        config_summary=status["config_summary"]
    )


@app.post("/api/initialize")
async def initialize_bitbot():
    """Initialize BitBot core."""
    global bitbot_core
    
    try:
        if not bitbot_core:
            config = BitBotConfig()
            bitbot_core = BitBotCore(config)
        
        success = await bitbot_core.initialize()
        
        return {
            "success": success,
            "message": "BitBot initialized successfully" if success else "Initialization failed"
        }
    except Exception as e:
        logger.error(f"Failed to initialize BitBot: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/start")
async def start_pipeline():
    """Start the BitBot pipeline."""
    global bitbot_core
    
    if not bitbot_core:
        raise HTTPException(status_code=400, detail="BitBot not initialized")
    
    try:
        success = await bitbot_core.start()
        return {
            "success": success,
            "message": "Pipeline started" if success else "Failed to start pipeline"
        }
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/stop")
async def stop_pipeline():
    """Stop the BitBot pipeline."""
    global bitbot_core
    
    if not bitbot_core:
        raise HTTPException(status_code=400, detail="BitBot not initialized")
    
    try:
        await bitbot_core.stop()
        return {
            "success": True,
            "message": "Pipeline stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop pipeline: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/chat")
async def chat_text(message: ChatMessage) -> ChatResponse:
    """Process a text chat message."""
    global bitbot_core, chat_history
    
    if not bitbot_core or not bitbot_core.pipeline:
        raise HTTPException(status_code=400, detail="BitBot not initialized")
    
    if not bitbot_core.pipeline.llm_engine:
        raise HTTPException(status_code=400, detail="LLM engine not available")
    
    try:
        # Create LLM message
        llm_message = LLMMessage(role="user", content=message.content)
        
        # Process with LLM
        response = await bitbot_core.pipeline.llm_engine.process_user_input(message.content)
        
        if response:
            # Add to chat history
            timestamp = datetime.now().isoformat()
            chat_history.append({
                "timestamp": timestamp,
                "user": message.content,
                "assistant": response.content
            })
            
            # Broadcast to WebSocket clients
            await broadcast_message({
                "type": "chat",
                "content": response.content,
                "timestamp": timestamp
            })
            
            return ChatResponse(
                content=response.content,
                timestamp=timestamp,
                success=True
            )
        else:
            return ChatResponse(
                content="Sorry, I couldn't process that request.",
                timestamp=datetime.now().isoformat(),
                success=False
            )
    
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        return ChatResponse(
            content=f"Error: {str(e)}",
            timestamp=datetime.now().isoformat(),
            success=False
        )


@app.get("/api/chat/history")
async def get_chat_history():
    """Get chat history."""
    return {"history": chat_history[-50:]}  # Return last 50 messages


@app.delete("/api/chat/history")
async def clear_chat_history():
    """Clear chat history."""
    global chat_history
    chat_history.clear()
    return {"success": True, "message": "Chat history cleared"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Process chat message
                content = message_data.get("content", "")
                if content.strip():
                    # Process with BitBot
                    chat_message = ChatMessage(content=content)
                    response = await chat_text(chat_message)
                    
                    # Response is already broadcasted in chat_text function
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(connected_clients)}")


async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    if not connected_clients:
        return
    
    message_str = json.dumps(message)
    disconnected_clients = []
    
    for client in connected_clients:
        try:
            await client.send_text(message_str)
        except Exception as e:
            logger.warning(f"Failed to send message to client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        connected_clients.remove(client)


# Status broadcasting
async def broadcast_status_updates():
    """Periodically broadcast status updates to connected clients."""
    while True:
        try:
            if connected_clients:
                status = await get_status()
                await broadcast_message({
                    "type": "status",
                    "status": status.dict()
                })
        except Exception as e:
            logger.error(f"Status broadcast error: {e}")
        
        await asyncio.sleep(5)  # Broadcast every 5 seconds


def start_web_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False
):
    """Start the web server."""
    
    # Start status broadcasting task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(broadcast_status_updates())
    
    logger.info(f"Starting BitBot Web Client at http://{host}:{port}")
    
    uvicorn.run(
        "bitbot.web_client.server:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None  # Use loguru instead
    )


if __name__ == "__main__":
    start_web_server() 