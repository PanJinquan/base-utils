# tmux命令

tmux new -s session_name    # 创建新会话
tmux ls                     # 列出所有会话
tmux attach -t session_name # 附加到指定会话
tmux detach                 # 分离当前会话（前缀键 d）
tmux kill-session -t session_name # 杀死指定会话
tmux switch -t session_name # 切换会话

```bash
tmux ls # 查看有多少个窗口
Crtl + B + { 或 Crtl + Shift + Alt + B + { # 浏览LOG记录[
Crtl + B + S 或 Crtl + Shift + Alt + B + S # 浏览窗口
```