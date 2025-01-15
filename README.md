# Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-play-with-execution-feedback-improving/instruction-following-on-ifeval)](https://paperswithcode.com/sota/instruction-following-on-ifeval?p=self-play-with-execution-feedback-improving)

*Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou*

Qwen, Alibaba Inc.

---

## :sparkles: Overview


This is the repository contains core implementations of the **AutoIF**, proposed by [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](https://arxiv.org/abs/2406.13542).

**AutoIF** is the first scalable and reliable method for automatically generating instruction-following data and verifying its quality using code execution feedback.

![image](https://github.com/dongguanting/AutoIF/assets/60767110/6c222465-25a4-4dec-ade6-d3a5af80ba39)



## :rocket: Data Synthesis of AutoIF
We divided the AutoIF's data synthesis process into steps and provided 10-20 samples per step to facilitate your reproduction. Please remember to replace them with your own input.

# AutoIF: 自动指令生成和过滤工具

AutoIF 是一个用于自动生成高质量指令并通过执行反馈验证其有效性的工具。它提供了两种使用方式：通过命令行工具或作为 Python 库导入使用。

## 安装

### 安装步骤

1. 克隆仓库：
```bash
git clone https://gitlab.pjlab.org.cn/huangzihan/autoif.git
cd AutoIF/
```

2. 安装依赖：
```bash
pip install -e .
```

## 使用方法

### 方法一：使用命令行工具

1. 准备配置：
```bash
python cli.py --seed-num 10 \
    --model "Qwen2.5-72B-Instruct" \
    --api-key "YOUR_API_KEY" \
    --base-url "http://localhost:8000/v1" \
    --batch-size 256 \
    --process-num 16 \
    --output-dir "./output" \
    --cache-dir ".cache" \
    --resume True
```

参数说明：
- `seed-num`: 种子指令重复次数,唯一决定总指令数量
- `model`: 使用的模型名称
- `api-key`: API 密钥
- `base-url`: API 服务地址
- `seed-dir`: 参考文件目录路径，默认为 "./sample_data"
- `batch-size`: 批处理大小
- `process-num`: 进程数量
- `output-dir`: 输出目录
- `cache-dir`: 缓存目录
- `no-resume`: 是否不从继续

2. 运行特定步骤：
```bash
python cli.py --start-step 1 --end-step 3  # 运行步骤1到3
```

### 方法二：作为 Python 库使用

1. 基本用法：
```python
from autoif.core import AutoIF

# 初始化 AutoIF
autoif = AutoIF(
    N=10,                    # 种子指令重复次数
    model="Qwen2.5-72B-Instruct",  # 模型名称
    api_key="YOUR_API_KEY",  # API密钥
    base_url="http://localhost:8000/v1",  # API地址
    process_num=16,          # 进程数
    batch_size=256,          # 批处理大小
    seed_dir="./sample_data",  # 参考文件目录
    output_dir="./output",   # 输出目录
    cache_dir=".cache",      # 缓存目录
)

# 运行完整流程
autoif.run()

# 运行特定步骤
autoif.run(start_step=1, end_step=3)  # 只运行步骤1到3
```

2. 处理流程说明：
- 步骤1: RFT生成指令
- 步骤2: 生成验证函数和测试用例
- 步骤3: 交叉验证
- 步骤4: 反向翻译
- 步骤5: 反向验证过滤
- 步骤6: 拼接ShareGPT查询
- 步骤7: 查询验证
- 步骤8: 评分
- 步骤9: 过滤
- 步骤10: 构建SFT数据

### 输出文件

每个步骤会在 output_dir 目录下生成对应的输出文件：
- `augment_instructions.txt`: 扩展后的指令
- `verification_funcs_cases.jsonl`: 验证函数和测试用例
- `cross_validation.jsonl`: 交叉验证结果
- `backtranslator.jsonl`: 反向翻译结果
- `backtranslator_filter.jsonl`: 反向验证过滤结果
- `sharegpt_query.jsonl`: ShareGPT查询结果
- `score_quality.jsonl`: 评分结果
- `score_filter.jsonl`: 过滤后结果
- `sft_data.jsonl`: 最终的SFT训练数据

### 缓存机制

AutoIF 使用异步缓存机制提高性能：
- 定时将内存中的数据写入磁盘（默认5秒）
- 支持断点续传
- 自动清理已完成步骤的缓存

### ⚠️ 重要提醒

1. 关于 resume 参数：
   - 当 `no-resume` 设置时，会删除所有之前的缓存数据

2. 关于输出目录：
   - 如果使用相同的 `output_dir`，新的结果会覆盖之前的文件
   - 建议每次运行时使用不同的输出目录，或备份重要结果


## 许可证

本项目采用 MIT 许可证。

