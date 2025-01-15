# 基础类和通用函数
from autoif.client.api_client import OpenAIClient   
import concurrent.futures
import asyncio
from diskcache import Index
from tqdm import tqdm
from typing import List, Protocol, TypeVar, Any
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
from autoif.utils import AsyncCache, md5, ensure_output_dir

class BaseAutoIFProtocol(Protocol):
    batch_size: int
    N: int
    client: OpenAIClient
    process_num: int
    start_time: float | None
    cache_dir: str
    current_step: int
    output_dir: str
    seed_dir: str
    resume: bool
    _current_cache: AsyncCache
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
    def __init__(self, N, model, api_key, base_url, batch_size, process_num, seed_dir, output_dir='./output', cache_dir='.cache', resume=True):
        self.batch_size = batch_size
        self.N = N
        self.seed_dir = seed_dir
        self.client = OpenAIClient(base_url, api_key, model)
        self.process_num = process_num
        self.start_time = None
        self.output_dir = os.path.join(output_dir, f"{model}-{md5(seed_dir)}")
        ensure_output_dir(self.output_dir)
        self.cache_dir = cache_dir
        self.resume = resume
        self.current_step = 0
        self._current_cache = None

    def set_step_cache(self, step: int):
        """获取指定步骤的缓存"""
        cache_path = os.path.join(self.cache_dir, str(step))
        os.makedirs(cache_path, exist_ok=True)
        self._current_cache = AsyncCache(cache_path)
    
    def clear_current_cache(self) -> None:
        """清除当前步骤的缓存"""
        if self._current_cache is not None:
            self._current_cache.stop()  # 停止缓存线程
            self._current_cache = None
            cache_path = os.path.join(self.cache_dir, str(self.current_step))
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
    
    async def batch_process_async(self, messages: List | List[List], total, process_funcs, **kwargs):
        futures = []
        next_index = 0
        completed_count = 0
        
        pbar = tqdm(total=total, desc="Processing")
        try:
            while completed_count < total:
                results = {}
                while len(futures) < self.batch_size and next_index < total:
                    if next_index in self._current_cache:
                        pbar.update(1)
                        next_index += 1
                        continue
                    
                    msg = messages if isinstance(messages[0], dict) else messages[next_index]
                    process_func = process_funcs[next_index] if isinstance(process_funcs, list) else process_funcs
                    task = asyncio.create_task(
                        self._process_single_task(msg, next_index, process_func, **kwargs)
                    )
                    futures.append(task)
                    next_index += 1
                
                if not futures:
                    break
                    
                done, pending = await asyncio.wait(
                    futures,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                futures = list(pending)
                
                for task in done:
                    try:
                        index, result = await task
                        if result is not None:
                            results[index] = result
                    except Exception as e:
                        print(f"任务执行出错: {e}")
                    finally:
                        completed_count += 1
                        pbar.update(1)
                
                if results:
                    self._current_cache.async_update(results)
                
        finally:
            pbar.close()
            self._current_cache.stop()

    async def _process_single_task(self, message, index, process_func, **kwargs):
        """处理单个任务并保持索引对应关系"""
        result = await self.client.create_chat_completions(messages=message, **kwargs)
        processed_result = process_func(result)
        return index, processed_result


    def get_process_pool(self):
        return concurrent.futures.ProcessPoolExecutor(max_workers=self.process_num) 