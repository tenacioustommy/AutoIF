from typing import Optional
import asyncio
import time
from datetime import timedelta
from .base import BaseAutoIF
from .rft import RFTMixin
from .backtranslator import BackTranslatorMixin
from .query import QueryMixin
import os
import shutil
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
            (8, self.score_quality, "评分"),
            (9, self.score_filter, "过滤"),
            (10, self.construct_sft_data, "构建SFT数据")
        ]
        
        # 确定起始步骤
        if not start_step and self.resume:
            # 查找存在的缓存目录
            for i in range(1, len(pipeline_steps) + 1):
                cache_path = os.path.join(self.cache_dir, str(i))
                if os.path.exists(cache_path) and os.listdir(cache_path):
                    start_step = i  # 继续当前步骤
                    print(f"发现缓存: 步骤 {i}")
                    break
        else:
            shutil.rmtree(self.cache_dir)
        start_step = start_step or 1
        end_step = end_step or len(pipeline_steps)
        
        if start_step > end_step:
            print("end_step 不能小于 start_step")
            return
        
        self.start_time = time.time()
        print(f"开始运行AutoIF流程 (步骤 {start_step} -> {end_step})")
        if self.resume and start_step > 1:
            print(f"从断点继续: 步骤 {start_step}")
        
        try:
            for step_num, func, desc in pipeline_steps:
                if start_step <= step_num <= end_step:
                    self.current_step = step_num
                    print(f"\n=== 步骤 {step_num}: {desc} ===")
                    
                    # 创建当前步骤的缓存
                    self.set_step_cache(step_num)
                    
                    try:
                        if asyncio.iscoroutinefunction(func):
                            await func()
                        else:
                            func()
                        
                        # 步骤成功完成后清理缓存
                        self.clear_current_cache()
                        
                    except Exception as e:
                        print(f"步骤 {step_num} 执行出错: {e}")
                        raise
                    
                    step_time = timedelta(seconds=int(time.time() - self.start_time))
                    print(f"完成步骤 {step_num}，已用时: {step_time}")
        
        except Exception as e:
            print(f"\n执行出错: {e}")
            raise
        finally:
            total_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"\n运行结束！总用时: {total_time}")

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