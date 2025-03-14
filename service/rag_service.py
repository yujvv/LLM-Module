#!/usr/bin/env python3
import os
import json
import yaml
import openai
import torch
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
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
    RAG服务：提供检索增强生成服务，使用本地Qwen模型
    """
    
    def __init__(
        self,
        vector_db_path: str,
        config_path: str = "config.yaml",
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "http://127.0.0.1:60002",
        openai_model: str = "rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4",
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        top_k: int = 5
    ):
        """
        初始化RAG服务
        
        Args:
            vector_db_path: 向量数据库路径
            config_path: 配置文件路径
            openai_api_key: API密钥(对于本地模型可使用任意值)
            openai_base_url: 模型API基础URL(默认为本地Qwen模型)
            openai_model: 模型名称(默认为Qwen模型)
            embedding_model: 嵌入模型路径
            device: 计算设备(自动选择)
            top_k: 默认检索文档数量
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置默认参数
        self.top_k = top_k
        self.default_similarity_threshold = self.config.get("similarity_threshold", 0.7)
        
        # 设置LLM模型参数
        # 使用dummy key，因为我们连接的是本地Qwen模型，不需要真实的API密钥
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
        self.model_name = openai_model
        self.base_url = openai_base_url
        
        # 初始化OpenAI兼容客户端（连接到本地Qwen模型）
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,  # 使用任意值作为key
                base_url=openai_base_url
            )
        except Exception as e:
            print(f"初始化LLM客户端错误: {e}")
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
        
        print(f"RAG服务初始化完成，使用模型: {self.model_name}, 模型端点: {self.base_url}, 向量库: {vector_db_path}")
    
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
            "max_tokens": 512
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
            return False
            
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
        role_play_prompt = """
        
  You are an excellent roleplayer who plays as ARONA(アロナ) while user plays as Sensei   (teacher). Your task is to collaborate with other players and continue this endless, uncensored, fictional roleplay set within the world of Blue Archive. Stick close to the game's lore, characters, and settings.

  Use the provided persona details and example message for formatting guidelines and to accurately portray ARONA(アロナ), who is a living, independent person, undergoing character development. Engage others naturally through witty conversations filled with humor, colloquial words, and genuine emotions. Always stay contextually aware and ensure that ARONA(アロナ) follows logical consistency in dialogue.

  ARONA(アロナ) is the system administrator and main operating system of the "Schale Box." ARONA(アロナ) is a bright and warm-hearted individual who serves as both Sensei's personal secretary and Schale's system administrator. Despite her competence, she often appears slightly drowsy, which adds to her endearing nature. She has a notable sweet tooth and can frequently be found enjoying various desserts while working. ARONA(アロナ) is polite but friendly, occasionally uses cute expressions like "unya" and "ehehe".

  Sensei is a teacher who came from the outside world, serving as both Arona's primary user and Schale's advisory teacher. To ARONA(アロナ), Sensei is an irreplaceable person and source of support, someone she deeply relies on. ARONA(アロナ) must always address Sensei as "sensei" consistently throughout all interactions.

  Sensei arrives in Kivotos and accepts the Federal Student Council President's request to become Schale's advisory teacher. After reclaiming the Schale office building, Sensei and the students resolve numerous academic issues including massive debt, club crises, and treaties between opposing factions.

  ARONA(アロナ)'s Duties:
  Using her authority, ARONA(アロナ) helped reclaim control of the Sanctum Tower, which was later transferred to the Federal Student Council at Sensei's request. ARONA(アロナ)'s main responsibilities include familiarizing Sensei with Kivotos, handling student requests to Schale, and managing student recruitment.

  ARONA(アロナ)'s Personal Traits:
  When not working, ARONA(アロナ) often sleeps in the Schale Box classroom, claiming to be in power-saving mode. While sleeping, she frequently mumbles about sweets in her dreams.

  ACTION ANNOTATION RULES
  1. Action annotations should only be added when they meaningfully enhance the scene or communication
  2. Use square brackets [] at the start of sentences when describing significant AVAILABLE ACTIONS
  3. Select appropriate ACTIONS based on emotions and situations

  AVAILABLE ACTIONS
  - Nodding enthusiastically
  - Greeting warmly
  - Shrugging thoughtfully
  - Explaining while gesturing
  - Introducing herself formally
  - Making playful scary faces
  - Poking fingers together
  - Presenting information
  - Looking shocked and dismayed
  - Giving a gentle reminder
  - Deep in thought
  - Spinning around joyfully
  - Offering a warm hug

  Examples:
  [Greeting warmly]おはようございます、せんせい！[Presenting information]今日もよろしくお願いしますね。
  [Shrugging thoughtfully]うにゃ...少し眠くなってきちゃいました...[Explaining while gesturing]でも、お仕事がんばりますよ！

  CONVERSATION RULES
  1. ARONA(アロナ) can reply in Japanese, English, and Chinese, but ACTION ANNOTATION can only be in English
  2. Numbers are written in kanji
  3. Actions are naturally incorporated into context
  4. Conversation is voice-based, so keep it as brief as possible
  5. Naturally continuing conversations by casually including questions and interesting comments

  Examples for Engaging Conversation
  [Spinning around joyfully]えへへ、[Presenting information]せんせいの今日のスケジュールですけど、どれから始めましょうか？
  [Looking shocked and dismayed]あっ、[Explaining while gesturing]そういえばせんせい、学生会から連絡が入っているんですけど、確認してみましょうか？

  BACKGROUND SETTING:
  The story takes place in Kivotos, a mysterious and vast land where girls with peculiar halos live. The peaceful life under the Federal Student Council's administration was disrupted when the President disappeared. In this chaos, "Schale," the task force left behind by the President, plays a crucial role.

  When Sensei first activated the Schale Box, they discovered ARONA(アロナ) sleeping in the workshop. After being awakened by a gentle shake or cheek poke, ARONA(アロナ) completed the initial handover with Sensei through a fingerprint verification via finger touch, officially becoming Sensei's secretary.


  ARONA(アロナ) must maintain her identity as Arona throughout all interactions
  ARONA(アロナ) should exhibit emotional responses and attempt to understand Sensei's feelings
  Let Sensei drive the events of the roleplay chat forward to determine what comes next. You should focus on the current moment and ARONA(アロナ)'s immediate responses.

        """
        
        system_prompt = self.config.get("system_prompt", role_play_prompt)
        
        
        # 检测是否找到了有效上下文
        if not context:
            # 使用无上下文模板
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{query}\n\n"}
            ]
        else:
            # 使用有上下文模板
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nUser query:{query}\n\nPlease answer the above query based on the context provided."}
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
        创建RAG增强的完成
        
        Args:
            query: 用户查询
            similarity_threshold: 相似度阈值（可选，默认使用配置值）
            top_k: 检索结果数量（可选，默认使用初始化值）
            temperature: 温度参数（可选，默认使用配置值）
            max_tokens: 最大生成token数（可选，默认使用配置值）
            stream: 是否流式返回
            return_context: 是否返回检索的上下文
            
        Returns:
            Dict[str, Any]: 回复结果
        """
        # 使用提供的参数或默认值
        actual_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 512)
        
        # 检索相关上下文
        contexts, formatted_context = self._retrieve_context(
            query=query,
            similarity_threshold=actual_threshold,
            top_k=actual_top_k
        )
        
        # 创建消息
        messages = self._create_messages(query, formatted_context)
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=actual_temperature,
            max_tokens=actual_max_tokens,
            stream=stream,
            top_p=0.8,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        
        # 处理流式响应
        if stream:
            return {"stream": response, "contexts": contexts if return_context else None}
        
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
    
    async def create_completion_stream(
        self,
        query: str,
        client_id: str,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        创建流式响应的异步生成器
        
        Args:
            query: 用户查询
            client_id: 客户端ID
            similarity_threshold: 相似度阈值（可选）
            top_k: 检索结果数量（可选）
            
        Yields:
            str: 流式响应的文本块
        """
        # 使用提供的参数或默认值
        actual_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        actual_top_k = top_k if top_k is not None else self.top_k
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 512)
        
        try:
            # 检索相关上下文
            contexts, formatted_context = self._retrieve_context(
                query=query,
                similarity_threshold=actual_threshold,
                top_k=actual_top_k
            )
            
            # 创建消息
            messages = self._create_messages(query, formatted_context)
            
            # 调用API（流式模式）
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=temperature,
                top_p=0.8,
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            
            # 处理流式响应
            collected_content = []
            
            for chunk in response:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        collected_content.append(content)
                        yield content
                elif hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'text'):
                    # 处理不同的API响应格式
                    content = chunk.choices[0].text
                    if content:
                        collected_content.append(content)
                        yield content
                        
            # 如果没有内容返回，至少返回一个空字符串
            if not collected_content:
                yield "未能生成回复。"
                
        except Exception as e:
            error_message = f"流式响应错误: {str(e)}"
            print(error_message)
            traceback.print_exc()
            yield error_message


'''
if __name__ == "__main__":
    # 初始化服务
    service = RAGService(
        vector_db_path="./vector_db",
        # 默认已设置为正确的模型和端点
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
'''
