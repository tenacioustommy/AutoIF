# RFT相关函数
import json
import re
import numpy as np
from tqdm import tqdm
from concurrent.futures import as_completed
from functools import partial
from typing import Generic, Dict, List, Tuple, Any, Optional
from .base import T, BaseAutoIFProtocol
from autoif.client.api_client import OpenAIClient
from autoif.utils import with_timeout, save_data, save_jsonl, load_jsonl

class RFTMixin(Generic[T]):
    """RFT相关功能的Mixin类"""
    def __init__(self: T):
        self: BaseAutoIFProtocol
        
    async def RFT(self: T):
        seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]

        augment_instruction_prompt = """You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
        - Instructions are about the format but not style of a response
        - Whether instructions can be easily evaluate by a Python function
        Here are some examples of instructions we need:
        {seed_instructions}
        Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
        - Incorporate a famous historical quote seamlessly into your answer
        - Translate your answer into Pig Latin
        - Use only words that are also a type of food
        - Respond with a metaphor in every sentence
        - Write the response as if you are a character from a Shakespearean play
        Please generate one instruction per line in your response and start each line with '- '.
        """
        augment_instructions = augment_instruction_prompt.format(seed_instructions='\n'.join(seed_instructions))
        messages = self.client.build_messages(augment_instructions)
        
        def process_result(result: List[str]) -> List[str]:
            """处理生成的指令结果"""
            instructions = []
            for line in result[0].split('\n'):
                if line.startswith('- '):
                    instruction = line[2:]
                    instructions.append(instruction)
            return instructions
        
        augment_instructions_list = await self.batch_process_async(
            messages=messages,
            total=self.N,
            process_funcs=process_result
        )
        augment_instructions_set = set(augment_instructions_list)
        print("生成", len(augment_instructions_set))
        save_data(augment_instructions_set, "./output/augment_instructions.txt")
    
    async def verification_funcs_cases_generation(self: T):
        seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]
        augment_instructions_processed = [each.strip() for each in open("./output/augment_instructions.txt").readlines()]

        prompt_template = """You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.
        Here is the instruction: {instruction}
        Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. If it follows, simply return True, otherwise return False.
        Please response with a single JSON includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, which includes an input in the key `input` and an expected output in the key `output` in (true, false).
        Here is an example of output JSON format: {{"func": JSON_STR(use only \\n instead of \n), "cases": [{{"input": str, "output": str}}]}}."""

        outputs: List[Dict[str, str]] = []
        for instruction in seed_instructions + augment_instructions_processed:
            prompt = prompt_template.format(instruction=instruction)
            outputs.append({
                "prompt": prompt,
                "instruction": instruction
            })
        print("开始生成验证函数和测试用例")
        
        def process_result(output: Dict[str, str], result: List[str]) -> Dict[str, Any]:
            """处理生成的函数和测试用例"""
            output["gpt-answer"] = [each.strip() for each in result]  
            return output
            
        outputs = await self.batch_process_async(
            messages=[self.client.build_messages(output["prompt"]) for output in outputs],
            total=len(outputs),
            process_funcs=[partial(process_result, output) for output in outputs],
            n=8
        )
        print("生成", len(outputs))
        save_jsonl(outputs, "./output/eval_func_rft.jsonl")
        
    @staticmethod
    @with_timeout
    def process_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理和验证生成的函数和测试用例"""
        def is_safe_code(code: str) -> bool:
            """检查代码是否安全"""
            dangerous_keywords = [
                'exit', 'quit', 'os.', 'sys.', 'subprocess', 'eval(', 'exec(', 
                'open(', '__import__', 'import os', 'import sys', 'import subprocess',
                'shutil', 'pathlib', 'remove', 'rmdir', 'unlink', 'delete'
            ]
            return not any(keyword in code for keyword in dangerous_keywords)

        res = result['gpt-answer']
        eval_funcs: List[str] = []
        test_cases: List[Tuple[str, bool]] = []

        # 处理每个生成的结果
        for each in res:
            try:
                json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
                res_dict = json.loads(json_dict)
            except (IndexError, json.JSONDecodeError):
                continue

            func = res_dict['func'].strip()
            if not is_safe_code(func):
                continue
                
            func = '\n'.join([line for line in func.split('\n') 
                            if 'download' not in line and 'requests' not in line])
            if '\\n' in func:
                func = func.replace('\\n', '\n')

            try:
                exec(func, {}, {})
                eval_funcs.append(func)
            except Exception:
                continue

            for case in res_dict['cases']:
                try:
                    test_cases.append((case['input'], case['output']))
                except KeyError:
                    print(case)
                    
        eval_funcs = list(set(eval_funcs))
        test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))
        
        if len(eval_funcs) < 3 or len(test_cases) < 10:
            return None

        # 过滤和评分测试用例
        filtered_test_cases = []
        for test_case in test_cases:
            if any(RFTMixin._validate_test_case(func, test_case) for func in eval_funcs):
                filtered_test_cases.append(test_case)

        # 评分函数
        scored_funcs = []
        for func in eval_funcs:
            score = RFTMixin._score_function(func, filtered_test_cases)
            if score >= 0.8:
                scored_funcs.append((func, score))

        if not scored_funcs:
            return None

        return {
            "instruction": result['instruction'],
            "eval_func": scored_funcs,
            "cases": filtered_test_cases
        }

    @staticmethod
    def _validate_test_case(func: str, test_case: Tuple[str, bool]) -> bool:
        """验证单个测试用例"""
        local_vars = {}
        try:
            exec(func, {}, local_vars)
            if 'evaluate' not in local_vars:
                return False
            eval_func = local_vars['evaluate']
            res = eval_func(test_case[0])
            return res is not None and res == test_case[1]
        except Exception:
            return False

    @staticmethod
    def _score_function(func: str, test_cases: List[Tuple[str, bool]]) -> float:
        """评分单个函数"""
        local_vars = {}
        try:
            exec(func, {}, local_vars)
            if 'evaluate' not in local_vars:
                return 0.0
            eval_func = local_vars['evaluate']
            
            scores = []
            for inp, out in test_cases:
                try:
                    res = eval_func(inp)
                    scores.append(1 if res is not None and res == out else 0)
                except Exception:
                    scores.append(0)
            
            return np.mean(scores) if scores else 0.0
        except Exception:
            return 0.0
        
    def cross_validation(self: T):   
        results = load_jsonl("./output/eval_func_rft.jsonl")
        print(f"total results: {len(results)}")
        print("cross validation for functions and cases")
        
        batch_size = self.process_num * 4096
        filter_results = []
        
        with self.get_process_pool() as process_pool:
            for i in range(0, len(results), batch_size):
                batch_results = results[i:i+batch_size]
                futures = [process_pool.submit(RFTMixin.process_result, result) 
                          for result in batch_results]
                
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        result = future.result()
                        if result is not None:
                            filter_results.append(result)
                    except Exception as e:
                        print(f"Error processing result: {e}")
        
        save_jsonl(filter_results, "./output/cross_validation.jsonl") 