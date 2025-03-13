#!/usr/bin/env python3
import os
import json
import yaml
import openai
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import traceback
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from embedding_service import CustomEmbeddingService


class RAGService:
    """
    RAG服务：提供类似OpenAI的接口，但在内部使用RAG增强回答质量
    """
    
    def __init__(
        self,
        vector_db_path: str,
        config_path: str = "config.yaml",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        top_k: int = 5
    ):
        """
        初始化RAG服务
        
        Args:
            vector_db_path: 向量数据库路径
            config_path: 配置文件路径
            openai_api_key: OpenAI API密钥(默认从环境变量获取)
            openai_base_url: OpenAI API基础URL(可选)
            openai_model: OpenAI模型名称
            embedding_model: 嵌入模型路径
            device: 计算设备(自动选择)
            top_k: 默认检索文档数量
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置默认参数
        self.top_k = top_k
        self.default_similarity_threshold = self.config.get("similarity_threshold", 0.7)
        
        # 设置OpenAI
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("请提供OpenAI API密钥或设置OPENAI_API_KEY环境变量")
            
        self.openai_model = openai_model
        
        # 初始化OpenAI客户端
        try:
            self.client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=openai_base_url
            )
        except Exception as e:
            print(f"初始化OpenAI客户端错误: {e}")
            traceback.print_exc()
            raise
        
        # 初始化嵌入模型
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
        self.embed_model = CustomEmbeddingService(
            model=embedding_model,
            device=device
        )
        Settings.embed_model = self.embed_model
        
        # 加载向量数据库
        self._load_vector_db(vector_db_path)
        
        print(f"RAG服务初始化完成，使用模型: {openai_model}, 向量库: {vector_db_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置参数
        """
        # 默认配置
        default_config = {
            "system_prompt": "你是一个有用的AI助手。请基于提供的上下文回答问题。如果上下文中没有相关信息，请直接说明不知道。",
            "similarity_threshold": 0.7,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config is None:  # 文件存在但为空
                        print(f"配置文件 {config_path} 为空，使用默认配置")
                        return default_config
                    return config
            else:
                print(f"配置文件 {config_path} 不存在，使用默认配置")
                return default_config
        except Exception as e:
            print(f"加载配置文件错误: {e}, 使用默认配置")
            return default_config
    
    def _load_vector_db(self, vector_db_path: str) -> None:
        """
        加载向量数据库
        
        Args:
            vector_db_path: 向量数据库路径
        """
        try:
            if not os.path.exists(vector_db_path):
                raise ValueError(f"向量数据库路径 {vector_db_path} 不存在")
                
            # 加载持久化索引
            storage_context = StorageContext.from_defaults(
                vector_store=FaissVectorStore.from_persist_dir(vector_db_path),
                persist_dir=vector_db_path
            )
            self.index = load_index_from_storage(storage_context)
            print(f"成功加载向量数据库: {vector_db_path}")
            
        except Exception as e:
            print(f"加载向量数据库错误: {e}")
            traceback.print_exc()
            raise RuntimeError(f"加载向量数据库错误: {e}")
    
    def _retrieve_context(
        self, 
        query: str, 
        similarity_threshold: float,
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        检索相关上下文
        
        Args:
            query: 查询文本
            similarity_threshold: 相似度阈值
            top_k: 检索结果数量
            
        Returns:
            Tuple[List[Dict], str]: 相关上下文列表和格式化后的上下文
        """
        # 获取检索器
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # 检索相似节点
        nodes: List[NodeWithScore] = retriever.retrieve(query)
        
        # 提取文本、元数据和分数
        results = []
        for node_with_score in nodes:
            node = node_with_score.node
            score = node_with_score.get_score(raise_error=False)
            
            # 应用相似度阈值
            if score < similarity_threshold:
                continue
            
            # 处理不同类型的节点
            if hasattr(node, 'text'):
                text = node.text
            else:
                text = str(node)
            
            # 获取元数据
            metadata = getattr(node, 'metadata', {})
            
            result = {
                "text": text,
                "metadata": metadata,
                "score": score
            }
            results.append(result)
        
        # 格式化上下文
        formatted_context = self._format_context(results)
        
        return results, formatted_context
    
    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        将上下文列表格式化为提示词中的上下文部分
        
        Args:
            contexts: 上下文列表
            
        Returns:
            str: 格式化后的上下文字符串
        """
        if not contexts:
            return "没有找到相关上下文。"
            
        formatted_context = "以下是与查询相关的上下文信息：\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("source", "未知来源")
            text = ctx.get("text", "")
            formatted_context += f"上下文 {i}（来源：{source}）：\n{text}\n\n"
            
        return formatted_context
    
    def _create_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        创建消息列表
        
        Args:
            query: 用户查询
            context: 格式化后的上下文
            
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        system_prompt = self.config.get("system_prompt", "你是一个有用的AI助手。请基于提供的上下文回答问题。如果上下文中没有相关信息，请直接说明不知道。")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\n用户查询：{query}\n\n请基于提供的上下文回答上述查询。"}
        ]
        
        return messages
    
    def create_completion(
        self,
        query: str,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        创建RAG增强的完成（类似OpenAI的接口）
        
        Args:
            query: 用户查询
            similarity_threshold: 相似度阈值（可选，默认使用配置值）
            top_k: 检索结果数量（可选，默认使用初始化值）
            temperature: 温度参数（可选，默认使用配置值）
            max_tokens: 最大生成token数（可选，默认使用配置值）
            stream: 是否流式返回
            return_context: 是否返回检索的上下文
            
        Returns:
            Dict[str, Any]: 回复结果，格式类似OpenAI的返回
        """
        # 使用提供的参数或默认值
        actual_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 1024)
        
        # 检索相关上下文
        contexts, formatted_context = self._retrieve_context(
            query=query,
            similarity_threshold=actual_threshold,
            top_k=actual_top_k
        )
        
        # 创建消息
        messages = self._create_messages(query, formatted_context)
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=actual_temperature,
            max_tokens=actual_max_tokens,
            stream=stream
        )
        
        # 构建返回结果
        result = {
            "completion": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        # 如果需要返回上下文信息
        if return_context:
            result["contexts"] = contexts
            result["formatted_context"] = formatted_context
        
        return result


# 使用示例
if __name__ == "__main__":
    # 初始化服务
    service = RAGService(
        vector_db_path="./vector_db",
        openai_model="gpt-3.5-turbo"
    )
    
    # 创建完成
    result = service.create_completion(
        query="量子计算机的工作原理是什么？",
        similarity_threshold=0.7,
        return_context=True
    )
    
    # 打印回复
    print("\n====== 回复 ======")
    print(result["completion"])
    
    # 打印上下文（如果请求）
    if "contexts" in result:
        print("\n====== 检索到的上下文 ======")
        for i, ctx in enumerate(result["contexts"], 1):
            print(f"上下文 {i} (相似度: {ctx['score']:.4f})")
            print(f"来源: {ctx['metadata'].get('source', '未知')}")
            print(f"文本: {ctx['text'][:150]}..." if len(ctx['text']) > 150 else f"文本: {ctx['text']}")
            print()
