# conversation_manager.py
from collections import defaultdict
from typing import Dict, List, Any
import threading

class ConversationManager:
    """
    管理不同客户端的对话历史记录 - 完全同步实现
    """
    
    def __init__(self):
        # 使用defaultdict自动为新客户端创建空列表
        self._history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        # 使用线程锁，而不是asyncio锁
        self._lock = threading.Lock()
    
    def add_message(self, client_id: str, role: str, content: str) -> None:
        """向对话历史添加一条消息"""
        with self._lock:
            self._history[client_id].append({"role": role, "content": content})
    
    def add_partial_response(self, client_id: str, content: str) -> None:
        """在流式响应过程中添加或更新助手的部分回复"""
        with self._lock:
            # 如果最后一条消息来自助手，则更新它
            if self._history[client_id] and self._history[client_id][-1]["role"] == "assistant":
                self._history[client_id][-1]["content"] += content
            else:
                # 否则，创建一条新的助手消息
                self._history[client_id].append({"role": "assistant", "content": content})
    
    def get_history(self, client_id: str, include_system: bool = False) -> List[Dict[str, str]]:
        """获取客户端的对话历史"""
        with self._lock:
            if client_id not in self._history:
                return []
            
            # 根据需要过滤系统消息
            if not include_system:
                return [msg for msg in self._history[client_id] if msg["role"] != "system"]
            return self._history[client_id].copy()
    
    def clear_history(self, client_id: str) -> bool:
        """清除客户端的对话历史"""
        with self._lock:
            if client_id in self._history:
                self._history[client_id] = []
                return True
            return False
    
    def get_formatted_history(self, client_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """获取格式化的最近对话历史，用于模型输入"""
        with self._lock:
            if client_id not in self._history:
                return []
                
            # 获取所有非系统消息
            history = [msg for msg in self._history[client_id] if msg["role"] != "system"]
            
            # 最多返回最近的max_turns条消息
            return history[-max_turns:] if len(history) > max_turns else history