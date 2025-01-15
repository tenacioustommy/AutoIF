# 查询相关函数
import re
import random
import copy
from tqdm import tqdm
from functools import partial
import signal
from concurrent.futures import as_completed
import json
import numpy as np
from typing import Generic, Dict, List
from .base import T, BaseAutoIFProtocol
from autoif.client.api_client import OpenAIClient
from autoif.utils import (
    save_jsonl, 
    load_jsonl, 
    contains_chinese, 
    with_timeout
)
import os

class QueryMixin(Generic[T]):
    def __init__(self: T):
        self: BaseAutoIFProtocol
    
    async def concat_sharegpt_query(self: T):
        print("开始拼接ShareGPT查询")
        
        # 读取过滤后的结果
        filter_results = load_jsonl(os.path.join(self.output_dir, "backtranslator_filter.jsonl"))
        
        # 读取并处理ShareGPT数据
        sft_data = load_jsonl("/cpfs01/user/huangzihan/AutoIF/sample_data/void_condor.jsonl")
        queries = [each['dialogs'][0]['content'] for each in sft_data]
        
        # 只保留长度在20-300之间且不包含中文的问题
        queries = [each for each in queries if len(each) > 20 and not contains_chinese(each)]
        
        # 构建输入数据
        inputs = []
        for instruction in tqdm(filter_results, desc="Preparing inputs"):
            ins_queries = random.sample(queries, 16)  # 拼16个
            for q in ins_queries:
                prompt = f"Please answer the query strictly following the instruction.\n[instruction] {instruction['instruction']}\n[Query] {q}"
                item = copy.deepcopy(instruction)
                item['prompt'] = prompt
                inputs.append(item)
        
        def process_result(result: List[str], item: Dict) -> Dict:
            """处理单个结果"""
            return {}
            # responses = [each.strip() for each in result]
            # item['gpt-answer'] = responses
            # return item
        
        print(f"开始生成回复，共 {len(inputs)} 个查询")
        
        # 批量处理生成回复
        await self.batch_process_async(
            messages=[self.client.build_messages(item['prompt']) for item in inputs],
            total=len(inputs),
            process_funcs=[partial(process_result, item=item) for item in inputs],
            n=4  # 每个query生成4个回复
        )
        
        print(f"生成完成，共 {len(self._current_cache)} 个结果")
        save_jsonl(list(self._current_cache.values()), os.path.join(self.output_dir, "sharegpt_query.jsonl"))
    
    @staticmethod
    @with_timeout
    def process_single_result(result: Dict) -> List[Dict]:
        """处理单个结果的函数"""
        eval_funcs = []
        for func, score in result['eval_func']:
            local_vars = {}
            try:
                exec(func, {}, local_vars) 
            except Exception as e:
                print(e)
                continue
            if 'evaluate' in local_vars:
                eval_funcs.append(local_vars['evaluate'])

        filter_responses = []
        for response in result['gpt-answer']:
            acc = []
            for eval_func in eval_funcs:
                try:
                    res = eval_func(response)
                    if res is not None:
                        acc.append(int(res))
                except:
                    continue
            acc = np.mean(acc) if acc else 0

            if acc > 0:
                filter_responses.append(response)

        samples = []
        for each in filter_responses:
            try:
                samples.append({
                    'instruction': result['instruction'],
                    'query': re.findall(r'\[Query\](.*)$', result['prompt'], re.DOTALL)[0].strip(),
                    'response': each
                })
            except IndexError:
                print(result['prompt'])
        return samples
    
    async def query_verification(self: T):
        print("开始查询验证")
        results = load_jsonl(os.path.join(self.output_dir, "sharegpt_query.jsonl"))
        all_samples = []
        
        # 使用进程池处理结果
        print(f"开始处理 {len(results)} 个结果")
        batch_size = self.process_num * 4096
        
        # 使用上下文管理器创建进程池
        with self.get_process_pool() as process_pool:
            for i in range(0, len(results), batch_size):
                batch_results = results[i:i+batch_size]
                futures = [process_pool.submit(QueryMixin.process_single_result, result) 
                          for result in batch_results]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"Processing batch {i//batch_size + 1}"):
                    try:
                        samples = future.result()
                        if samples:
                            all_samples.extend(samples)
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue

        print(f"初始样本数: {len(all_samples)}")
        # 去重
        all_samples = list(map(json.loads, set(map(json.dumps, all_samples))))
        print(f"去重后样本数: {len(all_samples)}")
        save_jsonl(all_samples, os.path.join(self.output_dir, "query_verification.jsonl"))
    
    
    async def score_quality(self: T):
        all_samples = load_jsonl(os.path.join(self.output_dir, "query_verification.jsonl"))
        # 构建评分prompt
        prompt_template = """You are an expert that is good at judging whether a response is following the instruction and query.
        [Instruction] {instruction}
        [Query] {query}
        [Response] {response}
        Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
        You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
        Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
        Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""

        # 添加评分prompt
        for sample in all_samples:
            sample['prompt'] = prompt_template.format(
                instruction=sample['instruction'],
                query=sample['query'],
                response=sample['response']
            )

        def process_score_result(result: List[str], item: Dict) -> Dict | None:
            """处理评分结果"""
            score_text = result[0].strip()
            score = re.findall(r'Score: (\d+?)$', score_text)
            if score:
                item['gen'] = [score_text]
                return item
            return None

        print("开始生成质量评分")
        # 使用异步批处理进行评分
        await self.batch_process_async(
            messages=[self.client.build_messages(item['prompt']) for item in all_samples],
            total=len(all_samples),
            process_funcs=[partial(process_score_result, item=item) for item in all_samples],
        )
        
        # 过滤None结果
        scored_results = []
        for result in list(self._current_cache.values()):
            if result is not None:
                scored_results.append(result)
        print(f"评分完成，共 {len(scored_results)} 个有效结果")
        save_jsonl(scored_results, os.path.join(self.output_dir, "score_quality.jsonl"))
      
    def score_filter(self: T):
        print("开始查询评分过滤")
        filter_results = []
        scored_results = load_jsonl(os.path.join(self.output_dir, "score_quality.jsonl"))
        print(f"初始结果数: {len(scored_results)}")
        
        for result in tqdm(scored_results, desc="Filtering results"):
            scores = []
            for each in result['gen']:
                score = re.findall(r'Score: (\d+?)$', each)
                if score:
                    scores.append(int(score[0]))
            score = np.mean(scores) if scores else 0
            if score > 8:  # quality score
                filter_results.append(result)
        
        print(f"过滤后结果数: {len(filter_results)}")
        
        # 统计唯一指令数
        unique_instructions = set()
        for each in filter_results:
            unique_instructions.add(each['instruction'])
        print(f"唯一指令数: {len(unique_instructions)}")
        
        save_jsonl(filter_results, os.path.join(self.output_dir, "score_filter.jsonl")) 

    def construct_sft_data(self: T):
        """
        构建SFT训练数据
        将query_score_filter.jsonl转换为标准的对话格式
        """
        print("开始构建SFT数据")
        data = load_jsonl(os.path.join(self.output_dir, "score_filter.jsonl"))
        
        processed_data = []
        for item in data:
            # 首字母大写处理
            query = item['query'][0].upper() + item['query'][1:]
            instruction = item['instruction'][0].upper() + item['instruction'][1:]
            
            # 构建输入文本
            if "?" in query:
                inputs = f"{query} {instruction}."
            elif "." in query:
                inputs = f"{query} {instruction}."
            else:
                inputs = f"{query}. {instruction}."

            # 构建对话格式数据
            new_item = {
                "dialogs": [
                    {
                        "role": "user",
                        "content": inputs
                    },
                    {
                        "role": "assistant",
                        "content": item['response']
                    }
                ]
            }
            processed_data.append(new_item)
        
        print(f"生成SFT数据 {len(processed_data)} 条, 保存到 {os.path.join(self.output_dir, 'sft_data.jsonl')}")
        save_jsonl(processed_data, os.path.join(self.output_dir, "sft_data.jsonl")) 