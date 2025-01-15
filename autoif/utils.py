import signal
from typing import Callable, TypeVar, Any, List, Dict
from functools import wraps
import jsonlines
import re
from diskcache import Index
import threading
from queue import Queue

T = TypeVar('T')

class AsyncCache(Index):
    """异步缓存类，继承自diskcache.Index，提供定时写入功能"""
    def __init__(self, directory, flush_interval=5, **kwargs):
        super().__init__(directory, **kwargs)
        self._cache_buffer: Dict = {}
        self._cache_lock = threading.Lock()
        self._flush_interval = flush_interval
        self._timer = None
        self._start_timer()
    
    def _start_timer(self):
        """启动定时器"""
        if self._timer is None:
            self._timer = threading.Timer(self._flush_interval, self._flush_buffer)
            self._timer.daemon = True
            self._timer.start()

    def _flush_buffer(self):
        """将缓冲区数据写入磁盘"""
        try:
            with self._cache_lock:
                if self._cache_buffer:
                    self.update(self._cache_buffer)
                    self._cache_buffer.clear()
        except Exception as e:
            print(f"缓存写入出错: {e}")
        finally:
            # 重置定时器
            if self._timer is not None:
                self._timer.cancel()  # 取消当前定时器
                self._timer = None
            self._start_timer()

    def async_update(self, other: dict):
        """异步更新缓存"""
        if other:
            with self._cache_lock:
                self._cache_buffer.update(other)

    def stop(self):
        """停止定时器并确保数据写入"""
        if self._timer:
            self._timer.cancel()
            self._timer = None
            # 最后一次写入
            with self._cache_lock:
                try:
                    self.update(self._cache_buffer)
                    self._cache_buffer.clear()
                except Exception as e:
                    print(f"最终缓存写入出错: {e}")

    def __getitem__(self, key):
        """获取数据时先检查缓冲区"""
        with self._cache_lock:
            if key in self._cache_buffer:
                return self._cache_buffer[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        """检查键是否存在时同时检查缓冲区"""
        with self._cache_lock:
            if key in self._cache_buffer:
                return True
        return super().__contains__(key)

    def __len__(self):
        """计算长度时排除元数据和缓冲区"""
        with self._cache_lock:
            total = super().__len__()
            return total + len(self._cache_buffer)

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
