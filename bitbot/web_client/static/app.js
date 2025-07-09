/**
 * BitBot Terminal Interface JavaScript
 * ===================================
 * 
 * Terminal-style interface for BitBot with command processing,
 * real-time updates, and proper tech aesthetics.
 */

class BitBotTerminal {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.initialized = false;
        this.commandHistory = [];
        this.historyIndex = -1;
        this.startTime = Date.now();
        
        // Terminal state
        this.systemStatus = 'OFFLINE';
        this.pipelineStatus = 'NOT_INITIALIZED';
        this.currentModel = 'NONE';
        this.voiceStatus = 'INACTIVE';
        
        // Elements
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            systemStatus: document.getElementById('system-status'),
            pipelineStatus: document.getElementById('pipeline-status'),
            uptimeDisplay: document.getElementById('uptime-display'),
            modelInfo: document.getElementById('model-info'),
            voiceStatus: document.getElementById('voice-status'),
            terminalOutput: document.getElementById('terminal-output'),
            terminalInput: document.getElementById('terminal-input'),
            terminalCursor: document.getElementById('terminal-cursor'),
            wsStatus: document.getElementById('ws-status'),
            cpuUsage: document.getElementById('cpu-usage'),
            memUsage: document.getElementById('mem-usage'),
            currentTime: document.getElementById('current-time'),
            helpContent: document.getElementById('help-content'),
            terminalBeep: document.getElementById('terminal-beep')
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.startStatusUpdates();
        this.updateCursor();
        this.playBootSequence();
        
        console.log('[TERMINAL] BitBot Terminal Interface initialized');
    }
    
    setupEventListeners() {
        // Input handling
        this.elements.terminalInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.elements.terminalInput.addEventListener('input', (e) => this.updateCursor());
        
        // Focus management
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.terminal-output')) {
                this.elements.terminalInput.focus();
            }
        });
        
        // Keep input focused
        this.elements.terminalInput.addEventListener('blur', () => {
            setTimeout(() => this.elements.terminalInput.focus(), 100);
        });
        
        // Prevent context menu on right click
        document.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    handleKeyDown(e) {
        switch (e.key) {
            case 'Enter':
                e.preventDefault();
                this.processCommand();
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                this.navigateHistory(-1);
                break;
                
            case 'ArrowDown':
                e.preventDefault();
                this.navigateHistory(1);
                break;
                
            case 'Tab':
                e.preventDefault();
                this.handleTabCompletion();
                break;
                
            case 'c':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.cancelCurrentOperation();
                }
                break;
                
            case 'l':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.clearTerminal();
                }
                break;
        }
    }
    
    async processCommand() {
        const input = this.elements.terminalInput.value.trim();
        if (!input) return;
        
        // Add to command history
        this.commandHistory.push(input);
        this.historyIndex = this.commandHistory.length;
        
        // Echo command
        this.addOutput('command', `$ ${input}`);
        
        // Clear input
        this.elements.terminalInput.value = '';
        this.updateCursor();
        
        // Process command
        if (input.startsWith('/')) {
            await this.executeSystemCommand(input);
        } else {
            await this.sendChatMessage(input);
        }
        
        this.playBeep();
    }
    
    async executeSystemCommand(command) {
        const [cmd, ...args] = command.slice(1).split(' ');
        
        switch (cmd.toLowerCase()) {
            case 'help':
                this.showHelp();
                break;
                
            case 'init':
            case 'initialize':
                await this.initializeSystem();
                break;
                
            case 'start':
                await this.startPipeline();
                break;
                
            case 'stop':
                await this.stopPipeline();
                break;
                
            case 'restart':
                await this.restartPipeline();
                break;
                
            case 'status':
                this.showSystemStatus();
                break;
                
            case 'config':
                this.showConfiguration();
                break;
                
            case 'clear':
                this.clearTerminal();
                break;
                
            case 'history':
                this.showCommandHistory();
                break;
                
            case 'version':
                this.showVersion();
                break;
                
            case 'exit':
            case 'quit':
                this.exitTerminal();
                break;
                
            case 'save':
                this.saveConversation();
                break;
                
            default:
                this.addOutput('error', `Unknown command: ${cmd}`);
                this.addOutput('info', 'Type /help for available commands');
        }
    }
    
    async sendChatMessage(message) {
        this.addOutput('user', `[USER] ${message}`);
        
        try {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                // Send via WebSocket
                this.socket.send(JSON.stringify({
                    type: 'chat',
                    content: message
                }));
                this.addOutput('info', 'Message sent via WebSocket...');
            } else {
                // Fallback to HTTP API
                this.addOutput('info', 'Sending message via HTTP...');
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: message })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    this.addOutput('assistant', `[BITBOT] ${result.content}`);
                } else {
                    this.addOutput('error', 'Failed to get response from BitBot');
                }
            }
        } catch (error) {
            this.addOutput('error', `Communication error: ${error.message}`);
        }
    }
    
    async initializeSystem() {
        this.addOutput('info', 'Initializing BitBot system...');
        this.addOutput('info', 'Loading components<span class="loading-dots"></span>');
        
        try {
            const response = await fetch('/api/initialize', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.initialized = true;
                this.addOutput('success', '✓ BitBot system initialized successfully');
                this.addOutput('info', 'System ready for pipeline startup');
                this.updateSystemStatus('READY');
            } else {
                this.addOutput('error', `✗ Initialization failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            this.addOutput('error', `✗ Initialization error: ${error.message}`);
        }
    }
    
    async startPipeline() {
        if (!this.initialized) {
            this.addOutput('warning', 'System not initialized. Run /init first');
            return;
        }
        
        this.addOutput('info', 'Starting voice pipeline...');
        this.addOutput('info', 'Activating components<span class="loading-dots"></span>');
        
        try {
            const response = await fetch('/api/start', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.addOutput('success', '✓ Voice pipeline started successfully');
                this.addOutput('info', 'Listening for wake word...');
                this.updatePipelineStatus('LISTENING');
                this.updateVoiceStatus('ACTIVE');
            } else {
                this.addOutput('error', `✗ Pipeline start failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            this.addOutput('error', `✗ Pipeline start error: ${error.message}`);
        }
    }
    
    async stopPipeline() {
        this.addOutput('info', 'Stopping voice pipeline...');
        
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.addOutput('success', '✓ Voice pipeline stopped');
                this.addOutput('info', 'Text chat remains available');
                this.updatePipelineStatus('STOPPED');
                this.updateVoiceStatus('INACTIVE');
            } else {
                this.addOutput('error', `✗ Pipeline stop failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            this.addOutput('error', `✗ Pipeline stop error: ${error.message}`);
        }
    }
    
    async restartPipeline() {
        this.addOutput('info', 'Restarting voice pipeline...');
        await this.stopPipeline();
        setTimeout(() => this.startPipeline(), 1000);
    }
    
    showHelp() {
        const helpText = this.elements.helpContent.textContent;
        this.addOutput('info', helpText);
    }
    
    showSystemStatus() {
        this.addOutput('info', 'SYSTEM STATUS REPORT');
        this.addOutput('info', '===================');
        this.addOutput('info', `System Status: ${this.systemStatus}`);
        this.addOutput('info', `Pipeline Status: ${this.pipelineStatus}`);
        this.addOutput('info', `Current Model: ${this.currentModel}`);
        this.addOutput('info', `Voice Status: ${this.voiceStatus}`);
        this.addOutput('info', `WebSocket: ${this.isConnected ? 'CONNECTED' : 'DISCONNECTED'}`);
        this.addOutput('info', `Uptime: ${this.getUptime()}`);
        this.addOutput('info', `Commands Executed: ${this.commandHistory.length}`);
    }
    
    showConfiguration() {
        this.addOutput('info', 'BITBOT CONFIGURATION');
        this.addOutput('info', '===================');
        this.addOutput('info', 'Hardware Tier: AUTO-DETECTED');
        this.addOutput('info', 'Terminal Interface: v1.0.0');
        this.addOutput('info', 'WebSocket Port: 8080');
        this.addOutput('info', 'Audio Sample Rate: 16000 Hz');
        this.addOutput('info', 'Wake Word: "Hey BitBot"');
    }
    
    showVersion() {
        this.addOutput('info', 'BitBot Terminal Interface v1.0.0');
        this.addOutput('info', 'Local AI Assistant Platform');
        this.addOutput('info', 'Built with FastAPI + WebSockets');
    }
    
    clearTerminal() {
        this.elements.terminalOutput.innerHTML = '';
        this.addOutput('info', 'Terminal cleared');
    }
    
    showCommandHistory() {
        this.addOutput('info', 'COMMAND HISTORY');
        this.addOutput('info', '===============');
        
        if (this.commandHistory.length === 0) {
            this.addOutput('info', 'No commands in history');
            return;
        }
        
        this.commandHistory.slice(-10).forEach((cmd, index) => {
            const lineNum = this.commandHistory.length - 10 + index + 1;
            this.addOutput('info', `${lineNum.toString().padStart(3)} ${cmd}`);
        });
    }
    
    saveConversation() {
        const output = this.elements.terminalOutput.innerText;
        const blob = new Blob([output], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `bitbot-session-${new Date().toISOString().slice(0, 19)}.txt`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.addOutput('success', 'Conversation saved to file');
    }
    
    exitTerminal() {
        this.addOutput('warning', 'Disconnecting from BitBot Terminal...');
        if (this.socket) {
            this.socket.close();
        }
        setTimeout(() => {
            this.addOutput('info', 'Session terminated. Refresh page to reconnect.');
        }, 1000);
    }
    
    navigateHistory(direction) {
        if (this.commandHistory.length === 0) return;
        
        this.historyIndex += direction;
        
        if (this.historyIndex < 0) {
            this.historyIndex = 0;
        } else if (this.historyIndex >= this.commandHistory.length) {
            this.historyIndex = this.commandHistory.length;
            this.elements.terminalInput.value = '';
        } else {
            this.elements.terminalInput.value = this.commandHistory[this.historyIndex];
        }
        
        this.updateCursor();
    }
    
    handleTabCompletion() {
        const input = this.elements.terminalInput.value;
        const commands = [
            '/help', '/init', '/start', '/stop', '/restart', '/status', 
            '/config', '/clear', '/history', '/version', '/exit', '/save'
        ];
        
        const matches = commands.filter(cmd => cmd.startsWith(input));
        
        if (matches.length === 1) {
            this.elements.terminalInput.value = matches[0] + ' ';
            this.updateCursor();
        } else if (matches.length > 1) {
            this.addOutput('info', `Possible completions: ${matches.join(' ')}`);
        }
    }
    
    cancelCurrentOperation() {
        this.addOutput('warning', '^C Operation cancelled');
        this.elements.terminalInput.value = '';
        this.updateCursor();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                console.log('[WS] Connected to BitBot WebSocket');
                this.updateConnectionStatus(true);
                this.addOutput('success', 'WebSocket connection established');
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('[WS] Error parsing message:', error);
                }
            };
            
            this.socket.onclose = () => {
                console.log('[WS] Disconnected from BitBot WebSocket');
                this.updateConnectionStatus(false);
                this.addOutput('warning', 'WebSocket connection lost');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.addOutput('info', 'Attempting to reconnect...');
                        this.connectWebSocket();
                    }
                }, 3000);
            };
            
            this.socket.onerror = (error) => {
                console.error('[WS] WebSocket error:', error);
                this.updateConnectionStatus(false);
                this.addOutput('error', 'WebSocket connection error');
            };
            
        } catch (error) {
            console.error('[WS] Failed to create WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'chat':
                this.addOutput('assistant', `[BITBOT] ${data.content}`);
                break;
                
            case 'status':
                this.updateSystemInfo(data.status);
                break;
                
            case 'error':
                this.addOutput('error', `System error: ${data.message}`);
                break;
                
            default:
                console.log('[WS] Unknown message type:', data.type);
        }
    }
    
    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const indicator = this.elements.connectionStatus;
        
        if (connected) {
            indicator.className = 'connection-indicator connected';
            this.elements.wsStatus.textContent = 'CONNECTED';
            this.elements.wsStatus.className = 'status-value status-online';
        } else {
            indicator.className = 'connection-indicator disconnected';
            this.elements.wsStatus.textContent = 'DISCONNECTED';
            this.elements.wsStatus.className = 'status-value status-error';
        }
    }
    
    updateSystemInfo(status) {
        this.updateSystemStatus(status.pipeline_state?.toUpperCase() || 'UNKNOWN');
        this.updatePipelineStatus(status.pipeline_state?.toUpperCase() || 'UNKNOWN');
        
        if (status.config_summary?.llm_model) {
            this.updateModelInfo(status.config_summary.llm_model.toUpperCase());
        }
        
        // Update voice status based on pipeline state
        if (status.pipeline_state === 'listening') {
            this.updateVoiceStatus('LISTENING');
        } else if (status.pipeline_state === 'processing') {
            this.updateVoiceStatus('PROCESSING');
        } else if (status.pipeline_state === 'speaking') {
            this.updateVoiceStatus('SPEAKING');
        } else {
            this.updateVoiceStatus('INACTIVE');
        }
    }
    
    updateSystemStatus(status) {
        this.systemStatus = status;
        this.elements.systemStatus.textContent = status;
        this.elements.systemStatus.className = `value ${this.getStatusClass(status)}`;
    }
    
    updatePipelineStatus(status) {
        this.pipelineStatus = status;
        this.elements.pipelineStatus.textContent = status;
        this.elements.pipelineStatus.className = `value ${this.getStatusClass(status)}`;
    }
    
    updateModelInfo(model) {
        this.currentModel = model;
        this.elements.modelInfo.textContent = model;
    }
    
    updateVoiceStatus(status) {
        this.voiceStatus = status;
        this.elements.voiceStatus.textContent = status;
        this.elements.voiceStatus.className = `value ${this.getStatusClass(status)}`;
    }
    
    getStatusClass(status) {
        const statusMap = {
            'ONLINE': 'status-online',
            'READY': 'status-online',
            'LISTENING': 'status-online',
            'PROCESSING': 'status-warning',
            'SPEAKING': 'status-info',
            'ACTIVE': 'status-online',
            'OFFLINE': 'status-offline',
            'INACTIVE': 'status-offline',
            'STOPPED': 'status-warning',
            'ERROR': 'status-error',
            'UNKNOWN': 'status-error'
        };
        
        return statusMap[status] || 'status-offline';
    }
    
    addOutput(type, message) {
        const outputLine = document.createElement('div');
        outputLine.className = `output-line ${type}`;
        
        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = `[${this.getCurrentTime()}]`;
        
        const messageSpan = document.createElement('span');
        messageSpan.className = 'message';
        messageSpan.innerHTML = message;
        
        outputLine.appendChild(timestamp);
        outputLine.appendChild(messageSpan);
        
        this.elements.terminalOutput.appendChild(outputLine);
        this.scrollToBottom();
    }
    
    updateCursor() {
        const input = this.elements.terminalInput;
        const cursor = this.elements.terminalCursor;
        
        // Position cursor at end of input
        const inputRect = input.getBoundingClientRect();
        const textWidth = this.getTextWidth(input.value, input);
        
        cursor.style.left = `${textWidth + 8}px`;
    }
    
    getTextWidth(text, element) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const style = window.getComputedStyle(element);
        
        context.font = `${style.fontSize} ${style.fontFamily}`;
        return context.measureText(text).width;
    }
    
    scrollToBottom() {
        this.elements.terminalOutput.scrollTop = this.elements.terminalOutput.scrollHeight;
    }
    
    getCurrentTime() {
        return new Date().toLocaleTimeString('en-US', { hour12: false });
    }
    
    getUptime() {
        const uptime = Date.now() - this.startTime;
        const hours = Math.floor(uptime / 3600000);
        const minutes = Math.floor((uptime % 3600000) / 60000);
        const seconds = Math.floor((uptime % 60000) / 1000);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    startStatusUpdates() {
        // Update time every second
        setInterval(() => {
            this.elements.currentTime.textContent = this.getCurrentTime();
            this.elements.uptimeDisplay.textContent = this.getUptime();
        }, 1000);
        
        // Poll system status every 5 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                this.updateSystemInfo(status);
            } catch (error) {
                // Silently fail - WebSocket handles real-time updates
            }
        }, 5000);
        
        // Simulate CPU/Memory usage (placeholder)
        setInterval(() => {
            this.elements.cpuUsage.textContent = `${Math.floor(Math.random() * 20 + 10)}%`;
            this.elements.memUsage.textContent = `${Math.floor(Math.random() * 512 + 256)}MB`;
        }, 2000);
    }
    
    playBeep() {
        try {
            this.elements.terminalBeep.currentTime = 0;
            this.elements.terminalBeep.play().catch(() => {
                // Ignore audio play errors
            });
        } catch (error) {
            // Ignore audio errors
        }
    }
    
    playBootSequence() {
        const bootMessages = [
            { delay: 0, type: 'startup', message: 'BitBot Terminal Interface v1.0.0' },
            { delay: 500, type: 'info', message: 'Initializing terminal session...' },
            { delay: 1000, type: 'info', message: 'Loading system modules...' },
            { delay: 1500, type: 'success', message: '✓ Terminal ready' },
            { delay: 2000, type: 'info', message: 'Type /help for available commands' }
        ];
        
        bootMessages.forEach(({ delay, type, message }) => {
            setTimeout(() => {
                this.addOutput(type, message);
            }, delay);
        });
    }
}

// Initialize the terminal when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.bitbotTerminal = new BitBotTerminal();
}); 