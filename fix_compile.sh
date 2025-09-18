#!/bin/bash
# MuJoCo编译工具修复脚本
# 解决pthread_create符号查找错误

export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0

# 获取脚本所在目录的上级目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_BIN="$HOME/mujoco/build/bin"

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <输入文件> <输出文件> [其他参数...]"
    echo "示例: $0 resources/robots/xgo/meshes/xgo.urdf output.xml"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
shift 2

# 执行编译命令
"$MUJOCO_BIN/compile" "$INPUT_FILE" "$OUTPUT_FILE" "$@"