#!/bin/bash
# Restore custom Hey BitBot model

echo "🔄 Restoring custom Hey BitBot model..."

# Stop current BitBot
pkill -f "python main.py start"
sleep 2

# Restore custom model
mv bitbot/wake_word/Hey_Bitbot.onnx.backup bitbot/wake_word/Hey_Bitbot.onnx

echo "✅ Custom model restored: bitbot/wake_word/Hey_Bitbot.onnx"
echo "🎯 Now say 'Hey BitBot' to test your custom model"
echo ""
echo "To test:"
echo "python main.py start --debug" 