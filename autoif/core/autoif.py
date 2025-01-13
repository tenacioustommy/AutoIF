from typing import Optional
import asyncio
import time
from datetime import timedelta
from .base import BaseAutoIF
from .rft import RFTMixin
from .backtranslator import BackTranslatorMixin
from .query import QueryMixin

class AutoIF(BaseAutoIF, RFTMixin, BackTranslatorMixin, QueryMixin):
    """
    AutoIF主类，集成所有功能模块
    
    工作流程：
    1. RFT生成指令
    2. 生成验证函数和测试用例
    3. 交叉验证
    4. 反向翻译
    5. 反向验证过滤
    6. 拼接ShareGPT查询
    7. 查询验证
    8. 构建SFT数据
    """
    
    async def run_pipeline(self, 
                         start_step: Optional[int] = None,
                         end_step: Optional[int] = None) -> None:
        """
        运行完整的处理流程
        
        Args:
            start_step: 起始步骤（1-8），默认从头开始
            end_step: 结束步骤（1-8），默认运行到最后
        """
        pipeline_steps = [
            (1, self.RFT, "RFT生成指令"),
            (2, self.verification_funcs_cases_generation, "生成验证函数和测试用例"),
            (3, self.cross_validation, "交叉验证"),
            (4, self.eval_func_backtranslator, "反向翻译"),
            (5, self.eval_func_backtranslator_filter, "反向验证过滤"),
            (6, self.concat_sharegpt_query, "拼接ShareGPT查询"),
            (7, self.query_verification, "查询验证"),
            (8, self.construct_sft_data, "构建SFT数据")
        ]
        
        start_step = start_step or 1
        end_step = end_step or len(pipeline_steps)
        
        self.start_time = time.time()
        print(f"开始运行AutoIF流程 (步骤 {start_step} -> {end_step})")
        
        for step_num, func, desc in pipeline_steps:
            if start_step <= step_num <= end_step:
                print(f"\n=== 步骤 {step_num}: {desc} ===")
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                
                step_time = timedelta(seconds=int(time.time() - self.start_time))
                print(f"完成步骤 {step_num}，已用时: {step_time}")
        
        total_time = timedelta(seconds=int(time.time() - self.start_time))
        print(f"\n全部完成！总用时: {total_time}")

    def run(self, 
            start_step: Optional[int] = None,
            end_step: Optional[int] = None) -> None:
        """
        运行AutoIF流程的同步包装器
        
        Args:
            start_step: 起始步骤（1-8），默认从头开始
            end_step: 结束步骤（1-8），默认运行到最后
        """
        asyncio.run(self.run_pipeline(start_step, end_step)) 