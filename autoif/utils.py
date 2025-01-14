import signal
from typing import Callable, TypeVar, Any, List, Dict
from functools import wraps
import jsonlines
import re
from diskcache import Index

T = TypeVar('T')

class MyCache(Index):
    def __len__(self):
        # 计算长度时排除元数据
        return super().__len__() - 1 if 'current_step' in self else super().__len__()

def save_data(data: List[str], path: str, mode: str = 'w') -> None:
    """保存文本数据到文件"""
    with open(path, mode, encoding='utf-8') as f:
        for each in data:
            f.write(each + '\n')
                
def save_jsonl(data: List[Dict], path: str, mode: str = 'w') -> None:
    """保存JSON数据到JSONL文件"""
    with jsonlines.open(path, mode=mode) as writer:
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

def with_timeout(func=None, *, timeout=2):
    """超时装饰器，支持带参数和不带参数两种方式
    
    可以这样使用:
        @with_timeout  # 使用默认超时时间
        def func(): pass
        
        @with_timeout(timeout=5)  # 自定义超时时间
        def func(): pass
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                return f(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper

    # 如果直接使用 @with_timeout 而不带参数
    if func is not None:
        return decorator(func)
    
    # 如果使用 @with_timeout(timeout=xxx)
    return decorator
