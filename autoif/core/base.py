# 基础类和通用函数
from autoif.client.api_client import OpenAIClient   
import concurrent.futures
import asyncio
from datetime import timedelta
import time
import signal
import jsonlines
from tqdm import tqdm
import itertools
from typing import List, Protocol, TypeVar, Any
from functools import partial
from concurrent.futures import ProcessPoolExecutor

class BaseAutoIFProtocol(Protocol):
    batch_size: int
    N: int
    client: OpenAIClient
    process_num: int
    start_time: float | None

    async def batch_process_async(
        self, 
        messages: List[dict] | List[List[dict]], 
        total: int, 
        process_funcs: List[callable] | callable,
        **kwargs
    ) -> List[Any]: ...
    def get_process_pool(self) -> ProcessPoolExecutor: ...
    # 添加其他基础方法...

T = TypeVar('T', bound=BaseAutoIFProtocol)

class BaseAutoIF:
    def __init__(self, N, model, api_key, base_url, batch_size, process_num):
        self.batch_size = batch_size
        self.N = N
        self.client = OpenAIClient(base_url, api_key, model)
        self.process_num = process_num
        self.start_time = None

    
    async def batch_process_async(self, messages: List | List[List], total, process_funcs, **kwargs):
        """
        异步批处理通用函数
        
        Args:
            messages: 要发送的消息
            total: 需要处理的总数量
            process_funcs: 处理返回结果的函数或函数列表
            **kwargs: 传递给create_chat_completions的额外参数
        
        Returns:
            处理结果的列表
        """
        futures = []
        results = []
        next_index = 0
        completed_count = 0
        
        pbar = tqdm(total=total, desc="Processing")
        try:
            while completed_count < total:
                # 创建协程任务列表
                while len(futures) < self.batch_size and next_index < total:
                    msg = messages if isinstance(messages[0], dict) else messages[next_index]
                    process_func = process_funcs[next_index] if isinstance(process_funcs, list) else process_funcs
                    task = asyncio.create_task(
                        self._process_single_task(msg, process_func, **kwargs)
                    )
                    futures.append(task)
                    next_index += 1
                
                if not futures:
                    break
                    
                # 等待任意一个任务完成
                done, pending = await asyncio.wait(
                    futures,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                futures = list(pending)
                
                # 处理完成的任务
                for task in done:
                    try:
                        result = await task
                        if result is not None:
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                        
                    except Exception as e:
                        print(f"任务执行出错: {e}")
                    finally:
                        completed_count += 1
                        pbar.update(1)
                        
        finally:
            pbar.close()
            return results

    async def _process_single_task(self, message, process_func, **kwargs):
        """处理单个任务并保持索引对应关系"""
        result = await self.client.create_chat_completions(messages=message, **kwargs)
        processed_result = process_func(result)
        return processed_result

    # def run(self):
    #     self.start_time = time.time()
    #     print(f"开始运行")
    #     asyncio.run(self.RFT())
    #     asyncio.run(self.verification_funcs_cases_generation())
    #     self.cross_validation()
    #     asyncio.run(self.eval_func_backtranslator())
    #     asyncio.run(self.eval_func_backtranslator_filter())
    #     asyncio.run(self.concat_sharegpt_query())
    #     asyncio.run(self.query_verification())
    #     end_time = time.time()
    #     duration = timedelta(seconds=int(end_time - self.start_time))
    #     print(f"总用时: {duration}") 

    def get_process_pool(self):
        return concurrent.futures.ProcessPoolExecutor(max_workers=self.process_num) 