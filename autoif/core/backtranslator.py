# 反向翻译相关函数
import re
from tqdm import tqdm
import jsonlines
from functools import partial
from typing import Generic
from .base import T, BaseAutoIFProtocol
from autoif.client.api_client import OpenAIClient
from autoif.utils import save_jsonl, load_jsonl
import os

class BackTranslatorMixin(Generic[T]):
    def __init__(self: T):
        self: BaseAutoIFProtocol
    
    async def eval_func_backtranslator(self: T):
        print("开始反向翻译")
        results = load_jsonl(os.path.join(self.output_dir, "cross_validation.jsonl"))
        
        # 构建翻译prompt
        translate_prompt = """Please translate the following instruction into Chinese, and then translate it back to English. Please make sure the back-translation maintains the original meaning but uses different wording.
        Instruction: {instruction}
        Please respond in the following format:
        Chinese: {{Chinese translation}}
        Back: {{back translation to English}}
        Back: {{another back translation}}
        Back: {{another back translation}}"""
        
        def process_result(result, item):
            translations = []
            for line in result[0].split('\n'):
                if line.startswith('Back:'):
                    trans = line[5:].strip()
                    if trans:
                        translations.append(trans)
            item['back_instruction'] = translations
            return item
        
        print(f"开始处理 {len(results)} 个指令")
        await self.batch_process_async(
            messages=[self.client.build_messages(translate_prompt.format(instruction=result['instruction'])) 
                     for result in results],
            total=len(results),
            process_funcs=[partial(process_result, item=result) for result in results]
        )
        
        outputs = list(self._current_cache.values())
        
        print(f"翻译完成，保存结果")
        save_jsonl(outputs, os.path.join(self.output_dir, "backtranslator.jsonl"))
        
    async def eval_func_backtranslator_filter(self: T):
        print("开始反向验证过滤")
        data = load_jsonl(os.path.join(self.output_dir, "backtranslator.jsonl"))
        
        filter_results = []
        filter_count = 0
        count = 0
        
        def build_nli_prompt(ori_ins, back_ins):
            return [{"role": "system", "content": f"""Please determine the relationship between the following two sentences - whether it is entailment, neutral, or contradiction.
            Sentence 1: {ori_ins}
            Sentence 2: {back_ins}
            Please only respond with one of these words: entailment, neutral, or contradiction."""}]
        
        def process_nli_result(result):
            content = result[0].strip().lower()
            if 'entailment' in content:
                return 'entailment'
            elif 'neutral' in content:
                return 'neutral'
            return 'contradiction'  # 默认返回contradiction
        
        for line in tqdm(data, desc="Processing lines"):
            back_instructions = line["back_instruction"]
            ori_ins = line['instruction']
            
            # 为每个back_instruction构建prompt
            messages = [build_nli_prompt(ori_ins, back_ins) 
                       for back_ins in back_instructions[:3]]
            
            # 批量处理NLI判断
            await self.batch_process_async(
                messages=messages,
                total=len(messages),
                process_funcs=process_nli_result,
                n=8  # 每个prompt生成8个回复
            )
            
            line["nli_scores"] = list(self._current_cache.values())
            
            if "contradiction" in line["nli_scores"]:
                filter_count += 1
                continue
            
            filter_results.append(line)
            count += 1
        
        print(f"过滤后剩余: {count}, 过滤掉: {filter_count}")
        save_jsonl(filter_results, os.path.join(self.output_dir, "backtranslator_filter.jsonl")) 