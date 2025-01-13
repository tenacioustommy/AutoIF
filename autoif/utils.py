import signal
from typing import Callable, TypeVar, Any, List, Dict
from functools import wraps
import jsonlines
import re

T = TypeVar('T')

def save_data(data: List[str], path: str) -> None:
    """保存文本数据到文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for each in data:
            f.write(each + '\n')
                
def save_jsonl(data: List[Dict], path: str) -> None:
    """保存JSON数据到JSONL文件"""
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(data)

def load_jsonl(path: str) -> List[Dict]:
    """从JSONL文件加载数据"""
    with jsonlines.open(path) as reader:
        return list(reader)

def contains_chinese(text: str) -> bool:
    """判断字符串是否包含中文"""
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))
        
def timeout_handler(signum, frame):
    """处理超时的信号处理器"""
    raise TimeoutError("Function execution timed out")

def with_timeout(func: Callable[..., T], timeout: int = 3) -> Callable[..., T | None]:
    """
    装饰器：为函数添加超时限制
    
    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
    
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T | None:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
        finally:
            signal.alarm(0)
    return wrapper
