import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any, Callable
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.bridge.pydantic import Field, PrivateAttr

class CustomEmbeddingService(BaseEmbedding):
    """
    自定义嵌入服务，适配LlamaIndex框架，使用sentence-transformers模型
    """
    
    model_name: str = Field(description="嵌入模型名称或路径")
    device: Optional[str] = Field(default=None, description="计算设备，如果为None，将自动检测")
    
    _model: SentenceTransformer = PrivateAttr()
    
    def __init__(
        self,
        model: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        embed_batch_size: int = 32,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化嵌入服务
        
        Args:
            model: 预训练模型的路径或名称
            device: 计算设备，如果为None，将自动检测
            embed_batch_size: 批处理大小
            callback_manager: 回调管理器
            additional_kwargs: 额外的参数
        """
        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
        
        # 初始化父类
        model_name = model if isinstance(model, str) else model.__class__.__name__
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            additional_kwargs=additional_kwargs or {},
            **kwargs,
        )
        
        # 初始化模型
        self.model_name = model
        self.device = device
        self._model = SentenceTransformer(model, device=device)
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名"""
        return "CustomEmbeddingService"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询文本的嵌入向量（当前实现是同步的）
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        # 当前使用同步实现，sentence-transformers不直接支持异步
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 文本
            
        Returns:
            List[float]: 嵌入向量
        """
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        异步获取文本的嵌入向量（当前实现是同步的）
        
        Args:
            text: 文本
            
        Returns:
            List[float]: 嵌入向量
        """
        # 当前使用同步实现
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.embed_batch_size,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量获取文本的嵌入向量（当前实现是同步的）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 当前使用同步实现
        return self._get_text_embeddings(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        Returns:
            int: 嵌入向量的维度
        """
        return self._model.get_sentence_embedding_dimension()

"""
# 在builder_script.py中使用:
from embedding_service import CustomEmbeddingService

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 替换OpenAIEmbedding的初始化代码
embed_model = CustomEmbeddingService(
    model="BAAI/bge-large-zh-v1.5", 
    device=device
)

# 设置全局配置
Settings.embed_model = embed_model

# 其余代码保持不变
"""