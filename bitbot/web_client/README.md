# BitBot Terminal Interface

A badass terminal-style web interface for BitBot that provides a proper tech aesthetic and command-line experience.

## Features

- 🖥️ **Terminal Aesthetic**: Real terminal look with green text, scanlines, and CRT effects
- ⌨️ **Command-Line Interface**: Full command system with tab completion and history
- 🎯 **Tech Styling**: Monospace fonts, ASCII art, and retro terminal vibes
- 🔄 **Real-time Updates**: WebSocket-based communication for instant responses
- 📊 **System Monitoring**: Live status updates and performance metrics
- 🎮 **Keyboard Shortcuts**: Proper terminal controls (Ctrl+C, Ctrl+L, arrow keys)
- 📱 **Responsive**: Works on desktop and mobile with terminal scaling

## Quick Start

### 1. Install Dependencies

```bash
# Install web client dependencies
pip install fastapi uvicorn websockets

# Or install all BitBot dependencies
pip install -r requirements.txt
```

### 2. Start the Terminal Interface

```bash
# Start terminal interface on default port (8080)
python main.py webclient

# Or specify custom host/port
python main.py webclient --host 0.0.0.0 --port 3000
```

### 3. Open in Browser

Navigate to `http://127.0.0.1:8080` (or your custom host/port)

## Terminal Commands

### System Control Commands
```bash
/init              # Initialize BitBot components
/start             # Start voice pipeline
/stop              # Stop voice pipeline  
/restart           # Restart the pipeline
/status            # Show detailed system status
/config            # Show configuration
```

### Chat Commands
```bash
/clear             # Clear terminal output
/history           # Show command history
/save              # Save conversation log
```

### Utility Commands
```bash
/help              # Show help message
/version           # Show version information
/exit              # Disconnect from terminal
```

### Chat Mode
Simply type your message without '/' to chat with BitBot:
```bash
Hello BitBot, how are you?
What's the weather like?
Tell me a joke
```

## Terminal Features

### Keyboard Controls
- **Enter**: Execute command/send message
- **↑/↓ Arrow Keys**: Navigate command history
- **Tab**: Auto-complete commands
- **Ctrl+C**: Cancel current operation
- **Ctrl+L**: Clear terminal
- **Ctrl+D**: Exit (same as /exit)

### Command History
- All commands are saved in session history
- Use arrow keys to navigate through previous commands
- View history with `/history` command

### Tab Completion
- Type `/` and press Tab to see available commands
- Partial command completion (e.g., `/in` + Tab = `/init`)

### Status Indicators
- **Connection Status**: WebSocket connection state
- **System Status**: BitBot initialization state
- **Pipeline Status**: Voice pipeline state
- **Model Info**: Current LLM model
- **Voice Status**: Voice interaction state

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│ ○ ○ ○                BitBot Terminal v1.0.0            ●   │
├─────────────────────────────────────────────────────────────┤
│  ██████╗ ██╗████████╗██████╗  ██████╗ ████████╗             │
│  ██╔══██╗██║╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝             │
│  ██████╔╝██║   ██║   ██████╔╝██║   ██║   ██║                │
│  ██╔══██╗██║   ██║   ██╔══██╗██║   ██║   ██║                │
│  ██████╔╝██║   ██║   ██████╔╝╚██████╔╝   ██║                │
│  ╚═════╝ ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝                │
│                                                             │
│       LOCAL AI ASSISTANT TERMINAL v1.0.0                   │
│       ====================================                   │
│                                                             │
│ SYSTEM: ONLINE    │ PIPELINE: LISTENING │ MODEL: MISTRAL    │
│ UPTIME: 01:23:45  │ VOICE: ACTIVE       │ TIME: 14:30:25    │
│                                                             │
│ ┌─────────────────── TERMINAL OUTPUT ───────────────────┐   │
│ │ [14:30:01] BitBot Terminal Interface initialized       │   │
│ │ [14:30:01] Type '/help' for available commands        │   │
│ │ [14:30:15] $ /init                                     │   │
│ │ [14:30:16] ✓ BitBot system initialized successfully   │   │
│ │ [14:30:20] $ Hello BitBot!                             │   │
│ │ [14:30:21] [BITBOT] Hello! How can I help you today?  │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                             │
│ user@bitbot:~$ ▓                                            │
├─────────────────────────────────────────────────────────────┤
│ WS: CONNECTED │ CPU: 15% │ MEM: 512MB │ PORT: 8080 │ 14:30:25│
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### Frontend
- **Pure HTML/CSS/JavaScript**: No framework dependencies
- **Terminal Effects**: CSS scanlines, CRT glow, and flicker effects
- **Monospace Fonts**: JetBrains Mono and Share Tech Mono
- **WebSocket**: Real-time bidirectional communication
- **Responsive Design**: Scales properly on mobile devices

### Backend
- **FastAPI**: Modern, fast web framework
- **WebSocket Support**: Real-time updates and chat
- **Async/Await**: Non-blocking I/O for better performance
- **Integration**: Seamless integration with existing BitBot components

### Terminal Aesthetics
- **Green-on-Black Theme**: Classic terminal colors
- **ASCII Art Banner**: BitBot logo in terminal art
- **Scanline Effects**: Authentic CRT monitor simulation
- **Blinking Cursor**: Real terminal cursor animation
- **Sound Effects**: Optional terminal beep sounds

## Configuration

### Command Line Options
```bash
python main.py webclient --help
```

Options:
- `--host` - Host to bind the server to (default: 127.0.0.1)
- `--port` - Port to bind the server to (default: 8080)
- `--reload` - Enable auto-reload for development

### Environment Variables
The terminal interface respects all BitBot environment variables:
- `PORCUPINE_ACCESS_KEY` - For wake word detection
- `OLLAMA_HOST` - Ollama server URL
- And all other BitBot configuration variables

## Usage Examples

### Basic Workflow
```bash
# Initialize the system
/init

# Start voice pipeline  
/start

# Chat with BitBot
Hello BitBot, what can you do?

# Check system status
/status

# Stop voice pipeline
/stop

# Save conversation
/save

# Exit terminal
/exit
```

### Advanced Commands
```bash
# View configuration
/config

# Show command history
/history

# Clear terminal
/clear

# Get help
/help

# Check version
/version
```

## Troubleshooting

### Common Issues

**Terminal interface won't load**
- Ensure FastAPI dependencies are installed: `pip install fastapi uvicorn websockets`
- Check if port 8080 is available or use `--port` option
- Verify that static files exist in `bitbot/web_client/static/`

**Commands not working**
- Check WebSocket connection status in the interface
- Ensure BitBot core system is properly configured
- Verify Ollama is running: `ollama serve`

**Voice features not working**
- Initialize BitBot first using `/init` command
- Start the voice pipeline using `/start` command
- Check that `PORCUPINE_ACCESS_KEY` is set for wake word detection

**WebSocket connection fails**
- Check browser console for errors
- Try refreshing the page
- Ensure no firewall is blocking the connection
- Check that the web server is running

### Browser Compatibility

The terminal interface works with:
- ✅ Chrome/Chromium 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Performance Tips

- Use Chrome or Chromium for best terminal effects
- Enable hardware acceleration for smooth animations
- Use full-screen mode for authentic terminal experience
- Monospace font rendering works best at 14px and above

## Development

### Running in Development Mode

```bash
# Enable auto-reload for development
python main.py webclient --reload

# Or run the server directly
python -m bitbot.web_client.server
```

### File Structure

```
bitbot/web_client/
├── __init__.py          # Package initialization
├── server.py            # FastAPI server and API endpoints
├── static/
│   ├── index.html       # Terminal interface HTML
│   ├── styles.css       # Terminal CSS styling
│   └── app.js          # Terminal JavaScript functionality
└── README.md           # This file
```

### Customization

You can customize the terminal interface by modifying:
- `static/styles.css` - Terminal colors, effects, and styling
- `static/app.js` - Command handling and terminal behavior
- `static/index.html` - Terminal layout and structure
- `server.py` - API endpoints and server logic

### Adding New Commands

To add new terminal commands:

1. Add the command to the help text in `index.html`
2. Add the command handling logic in `app.js` in the `executeSystemCommand` function
3. Optionally add corresponding API endpoints in `server.py`

## Security

- **Local Only**: Designed for local use (127.0.0.1 by default)
- **No Authentication**: Suitable for single-user local deployment
- **CORS Configured**: Allows local development and testing
- **WebSocket Security**: Same-origin policy enforced

## License

Same license as BitBot main project. 