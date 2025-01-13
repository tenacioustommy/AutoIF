#!/bin/bash
start=${1:-1}
end=${2:-9}
echo "start: $start, end: $end"
# 记录开始时间
start_time=$(date +%s)
echo "开始时间: $(date -d @$start_time +%H:%M:%S)"

# 执行 code 目录下的所有 Python 文件
for file in code/*.py; do
    # 检查文件名是否以数字开头
    if [[ $(basename "$file") =~ ^[$start-$end] ]]; then
        echo "==============================================="
        echo "开始执行: $file"
        echo "==============================================="
        
        # 执行 Python 文件
        python "$file"
        
        # 检查执行状态
        if [ $? -eq 0 ]; then
            echo "✅ $file 执行成功"
        else
            echo "❌ $file 执行失败"
            # 可以选择在这里退出脚本
            exit 1
        fi
    fi
done

# 记录结束时间并计算总运行时间
end_time=$(date +%s)
echo "结束时间: $(date -d @$end_time +%H:%M:%S)"
duration=$((end_time - start_time))
echo "总运行时间: $duration 秒"

echo "所有 Python 文件执行完成"
