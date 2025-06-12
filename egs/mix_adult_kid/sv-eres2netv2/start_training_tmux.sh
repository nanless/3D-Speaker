#!/bin/bash

# 设置脚本运行目录
cd "$(dirname "$0")"

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 设置tmux会话名称
SESSION_NAME="training_${timestamp}"

# 检查tmux是否已安装
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  CentOS/RHEL: sudo yum install tmux"
    exit 1
fi

# 检查是否已有训练会话在运行
if tmux list-sessions 2>/dev/null | grep -q "training_"; then
    echo "Warning: Found existing training sessions:"
    tmux list-sessions | grep "training_"
    echo ""
    read -p "Do you want to continue and create a new session? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting training in tmux session: $SESSION_NAME"
echo "Logs will be saved to logs/run_${timestamp}.log"

# 创建tmux会话并在其中运行训练
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" \
    "bash -c 'echo \"Training started at \$(date)\" | tee logs/run_${timestamp}.log; ./run.sh 2>&1 | tee -a logs/run_${timestamp}.log'"

# 保存会话信息
echo "$SESSION_NAME" > logs/current_session.txt
echo "Session name saved to logs/current_session.txt"

echo ""
echo "Training is now running in tmux session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  View training progress: tmux attach-session -t $SESSION_NAME"
echo "  Detach from session:    Ctrl+B, then D"
echo "  Kill training session: tmux kill-session -t $SESSION_NAME"
echo "  List all sessions:      tmux list-sessions"
echo "  Follow log file:        tail -f logs/run_${timestamp}.log"
echo ""
echo "The training will continue even if you disconnect from SSH" 