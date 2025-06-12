#!/bin/bash

# 设置脚本运行目录
cd "$(dirname "$0")"

show_help() {
    echo "Training Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start new training session"
    echo "  attach    - Attach to current training session"
    echo "  status    - Show training status"
    echo "  logs      - Show recent logs"
    echo "  stop      - Stop current training session"
    echo "  list      - List all tmux sessions"
    echo "  clean     - Clean old training sessions"
    echo "  help      - Show this help message"
    echo ""
}

get_current_session() {
    if [ -f "logs/current_session.txt" ]; then
        cat logs/current_session.txt
    else
        echo ""
    fi
}

case "$1" in
    "start")
        ./start_training_tmux.sh
        ;;
    "attach")
        SESSION=$(get_current_session)
        if [ -n "$SESSION" ] && tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Attaching to session: $SESSION"
            echo "Press Ctrl+B, then D to detach"
            tmux attach-session -t "$SESSION"
        else
            echo "No active training session found."
            echo "Available sessions:"
            tmux list-sessions 2>/dev/null | grep "training_" || echo "  None"
        fi
        ;;
    "status")
        SESSION=$(get_current_session)
        if [ -n "$SESSION" ] && tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Training session is running: $SESSION"
            echo "Session info:"
            tmux list-sessions | grep "$SESSION"
        else
            echo "No active training session found."
        fi
        echo ""
        echo "All training sessions:"
        tmux list-sessions 2>/dev/null | grep "training_" || echo "  None"
        ;;
    "logs")
        if [ -f "logs/current_session.txt" ]; then
            # 获取最新的日志文件
            LATEST_LOG=$(ls -t logs/run_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                echo "Following log file: $LATEST_LOG"
                echo "Press Ctrl+C to exit"
                tail -f "$LATEST_LOG"
            else
                echo "No log files found."
            fi
        else
            echo "No current training session found."
        fi
        ;;
    "stop")
        SESSION=$(get_current_session)
        if [ -n "$SESSION" ] && tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Stopping training session: $SESSION"
            tmux kill-session -t "$SESSION"
            rm -f logs/current_session.txt
            echo "Training session stopped."
        else
            echo "No active training session found."
        fi
        ;;
    "list")
        echo "All tmux sessions:"
        tmux list-sessions 2>/dev/null || echo "  No sessions running"
        ;;
    "clean")
        echo "Cleaning old training sessions..."
        OLD_SESSIONS=$(tmux list-sessions 2>/dev/null | grep "training_" | cut -d: -f1)
        if [ -n "$OLD_SESSIONS" ]; then
            echo "Found sessions to clean:"
            echo "$OLD_SESSIONS"
            read -p "Kill all these sessions? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "$OLD_SESSIONS" | while read session; do
                    tmux kill-session -t "$session"
                    echo "Killed session: $session"
                done
                rm -f logs/current_session.txt
                echo "Cleanup completed."
            else
                echo "Cleanup cancelled."
            fi
        else
            echo "No training sessions to clean."
        fi
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 