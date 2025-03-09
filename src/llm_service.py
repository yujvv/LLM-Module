#!/usr/bin/env python3
import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from querier_script import query_vector_db

class RAGQueryProcessor:
    """
    RAG查询处理器：处理查询、获取相关上下文并调用语言模型生成回复
    """
    
    def __init__(
        self,
        persist_dir: str = "./vector_db",
        similarity_threshold: float = 0.7,
        top_k: int = 5,
        local_llm_url: str = "http://localhost:8000/v1/chat/completions",
        model_name: str = "local-model",
        embed_model: str = "text-embedding-3-small",
        system_prompt: str = "你是一个有用的AI助手。请基于提供的上下文回答问题。如果上下文中没有相关信息，请直接说明不知道。"
    ):
        """
        初始化RAG查询处理器
        
        Args:
            persist_dir: 向量数据库存储目录
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤
            top_k: 检索的最大结果数量
            local_llm_url: 本地语言模型API地址
            model_name: 本地模型名称
            embed_model: 使用的OpenAI嵌入模型
            system_prompt: 系统提示词
        """
        self.persist_dir = persist_dir
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.local_llm_url = local_llm_url
        self.model_name = model_name
        self.embed_model = embed_model
        self.system_prompt = system_prompt
    
    def get_relevant_context(self, query: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        获取与查询相关的上下文，并应用相似度阈值过滤
        
        Args:
            query: 用户查询文本
            verbose: 是否打印详细信息
            
        Returns:
            过滤后的相关上下文列表
        """
        # 调用现有的query_vector_db函数获取相似文本块
        results = query_vector_db(
            query_text=query,
            persist_dir=self.persist_dir,
            embed_model=self.embed_model,
            top_k=self.top_k,
            verbose=verbose
        )
        
        # 应用相似度阈值过滤
        filtered_results = [r for r in results if r["score"] >= self.similarity_threshold]
        
        if verbose:
            print(f"检索到 {len(results)} 个结果，过滤后剩余 {len(filtered_results)} 个结果")
            
        return filtered_results
    
    def format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        将上下文列表格式化为提示词中的上下文部分
        
        Args:
            contexts: 上下文列表
            
        Returns:
            格式化后的上下文字符串
        """
        if not contexts:
            return "没有找到相关上下文。"
            
        formatted_context = "以下是与查询相关的上下文信息：\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("source", "未知来源")
            text = ctx.get("text", "")
            formatted_context += f"上下文 {i}（来源：{source}）：\n{text}\n\n"
            
        return formatted_context
    
    def build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        构建提示词
        
        Args:
            query: 用户查询
            context: 格式化后的上下文
            
        Returns:
            消息列表，用于发送给语言模型
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{context}\n\n用户查询：{query}\n\n请基于提供的上下文回答上述查询。"}
        ]
        
        return messages
    
    def call_local_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用本地语言模型获取回复
        
        Args:
            messages: 消息列表
            
        Returns:
            模型生成的回复
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.local_llm_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                error_msg = f"调用本地模型失败：HTTP {response.status_code}, {response.text}"
                print(error_msg)
                return f"错误：{error_msg}"
                
        except Exception as e:
            error_msg = f"调用本地模型时发生错误：{str(e)}"
            print(error_msg)
            return f"错误：{error_msg}"
    
    def process_query(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        处理查询的主要方法：获取上下文、构建提示词、调用模型、返回结果
        
        Args:
            query: 用户查询
            verbose: 是否打印详细信息
            
        Returns:
            包含响应和上下文信息的字典
        """
        if verbose:
            print(f"处理查询: '{query}'")
            
        # 获取相关上下文
        contexts = self.get_relevant_context(query, verbose)
        
        # 格式化上下文
        formatted_context = self.format_context(contexts)
        
        # 构建提示词
        messages = self.build_prompt(query, formatted_context)
        
        if verbose:
            print("发送到模型的消息:")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            
        # 调用本地语言模型
        response = self.call_local_llm(messages)
        
        # 返回结果
        result = {
            "query": query,
            "response": response,
            "contexts": contexts,
            "formatted_context": formatted_context
        }
        
        return result


# 使用示例
if __name__ == "__main__":
    # 创建RAG处理器实例
    processor = RAGQueryProcessor(
        persist_dir="./vector_db",
        similarity_threshold=0.75,
        top_k=3,
        local_llm_url="http://localhost:8000/v1/chat/completions",
        model_name="local-model"
    )
    
    # 处理查询
    query = "量子计算机的工作原理是什么？"
    result = processor.process_query(query, verbose=True)
    
    # 打印回复
    print("\n====== 最终回复 ======")
    print(result["response"])