import argparse
import os
from typing import Optional
from autoif.core import AutoIF

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoIF: 自动指令生成和过滤工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument("-n", "--seed_num", 
                       type=int, default=10,
                       help="种子指令重复次数")
    parser.add_argument("--model", 
                       type=str, default="Qwen2.5-72B-Instruct",
                       help="使用的模型名称")
    parser.add_argument("--api_key", 
                       type=str, default="EMPTY",
                       help="API认证密钥")
    parser.add_argument("--base_url", 
                       type=str, default="http://localhost:8000/v1",
                       help="API服务地址")
    
    # 性能配置
    parser.add_argument("--batch_size", 
                       type=int, default=256,
                       help="批处理大小")
    parser.add_argument("--process_num", 
                       type=int, default=16,
                       help="进程数量")
    
    # 流程控制
    parser.add_argument("--start_step",
                       type=int, default=None,
                       choices=range(1, 9),
                       help="起始步骤 (1-8)")
    parser.add_argument("--end_step",
                       type=int, default=None,
                       choices=range(1, 9),
                       help="结束步骤 (1-8)")
    
    # 输出配置
    parser.add_argument("--output_dir",
                       type=str, default="./output",
                       help="输出目录路径")
    
    return parser.parse_args()

def ensure_output_dir(output_dir: str) -> None:
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

def main():
    args = parse_args()
    
    # 确保输出目录存在
    ensure_output_dir(args.output_dir)
    
    # 创建AutoIF实例
    autoif = AutoIF(
        N=args.seed_num,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        process_num=args.process_num,
        batch_size=args.batch_size
    )
    
    try:
        # 运行流程
        autoif.run(
            start_step=args.start_step,
            end_step=args.end_step
        )
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
        raise
    
if __name__ == '__main__':
    main()
       