#!/usr/bin/env python3
from collections import defaultdict
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
from conversation_manager import ConversationManager


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
        openai_model: str = "default",
        embedding_model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        top_k: int = 5,
        require_vector_db: bool = True  # Add parameter to control whether vector DB is required
    ):
        """
        初始化RAG服务
        
        Args:
            vector_db_path: 向量数据库路径
            config_path: 配置文件路径
            openai_api_key: API密钥(对于本地模型可使用任意值)
            openai_base_url: 模型API基础URL(默认为本地Qwen模型)
            openai_model: 模型名称(默认为default)
            embedding_model: 嵌入模型路径
            device: 计算设备(自动选择)
            top_k: 默认检索文档数量
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置默认参数
        self.top_k = top_k
        self.default_similarity_threshold = self.config.get("similarity_threshold", 2.0)
        
        # 设置LLM模型参数
        # 使用dummy key，因为我们连接的是本地Qwen模型，不需要真实的API密钥
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
        
        # 优先使用参数传入的模型名称，如果没有则使用配置文件中的，都没有则使用默认值
        self.model_name = openai_model or self.config.get("model_name", "default")
        
        # 确保base_url以/v1结尾
        self.base_url = openai_base_url
        if not self.base_url.endswith('/v1'):
            self.base_url = f"{self.base_url}/v1"
            
        # 初始化OpenAI兼容客户端（连接到本地Qwen模型）
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,  # 使用任意值作为key
                base_url=self.base_url
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
        
        # 检查向量数据库是否成功加载
        if require_vector_db and self.index is None:
            raise RuntimeError(f"向量数据库加载失败，无法提供RAG服务。请检查向量数据库路径: {vector_db_path}")
        

        # Initialize conversation manager
        self.conversation_manager = ConversationManager()

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
        # 先设置一个默认值，防止后续代码出错
        self.index = None
        
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
            # 不立即抛出异常，而是在服务初始化时检查
            print(f"警告: 向量数据库加载失败。RAG功能将不可用。")
    
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
        # 检查索引是否存在
        if self.index is None:
            print("警告: 向量数据库未加载，无法进行检索。")
            return [], ""
        
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
            
        formatted_context = "The following is contextual information related to the query:\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("source", "Unknown source")
            text = ctx.get("text", "")
            formatted_context += f"Context {i}(Source:{source}):\n{text}\n\n"
            
        return formatted_context
    
    # def _create_messages(self, query: str, context: str) -> List[Dict[str, str]]:
    def _create_messages(self, query: str, context: str, client_id: str = None, include_history: bool = True) -> List[Dict[str, str]]:
        """
        创建消息列表，包含历史记录
        
        Args:
            query: 用户查询
            context: 格式化后的上下文
            client_id: 客户端ID，用于获取历史记录
            include_history: 是否包含历史记录
            
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
    
        # Create base messages
        if not context:
            base_messages = [
                {"role": "system", "content": system_prompt}
            ]
        else:
            base_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\n以下是我的查询:\n{query}\n\n请根据提供的上下文回答问题。"}
            ]
        
        # If we don't need history or don't have a client ID, return just the base messages
        if not include_history or not client_id:
            # For a fresh conversation without history, add the query as user message
            if not context:
                base_messages.append({"role": "user", "content": query})
            return base_messages
        
        # Get conversation history
        if hasattr(self.conversation_manager, 'get_formatted_history'):
            history = self.conversation_manager.get_formatted_history(client_id)
        else:
            history = []
        
        # Combine base messages with history
        # Start with system message
        complete_messages = [base_messages[0]]
        
        # Add history messages
        complete_messages.extend(history)
        
        # Add the current query as the last user message if not using context
        # and it's not a duplicate of the last message
        if not context:
            # Check if we need to add the current query
            should_add_query = True
            
            # Check if last message in history is from user with the same content
            if history and history[-1]["role"] == "user" and history[-1]["content"] == query:
                should_add_query = False
                
            if should_add_query:
                complete_messages.append({"role": "user", "content": query})
        
        return complete_messages
    
    
    # Add this synchronous helper method to RAGService class:
    def _get_history_sync(self, client_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        同步获取对话历史 - 避免事件循环问题
        """
        # Directly access the history data structure
        if not hasattr(self.conversation_manager, '_history') or client_id not in self.conversation_manager._history:
            return []
            
        # Get history without using async
        history = [msg for msg in self.conversation_manager._history[client_id] 
                if msg["role"] != "system"]
        
        # Return at most the last max_turns messages
        return history[-max_turns:] if len(history) > max_turns else history
 
    # Modify create_completion method to store message history
    # Also modify the create_completion method to avoid running coroutines directly:
    def create_completion(
        self,
        query: str,
        client_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        return_context: bool = False,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        创建RAG增强的完成
        
        Args:
            query: 用户查询
            client_id: 客户端ID，用于存储和获取历史记录
            similarity_threshold: 相似度阈值（可选，默认使用配置值）
            top_k: 检索结果数量（可选，默认使用初始化值）
            temperature: 温度参数（可选，默认使用配置值）
            max_tokens: 最大生成token数（可选，默认使用配置值）
            stream: 是否流式返回
            return_context: 是否返回检索的上下文
            include_history: 是否包含历史记录
            
        Returns:
            Dict[str, Any]: 回复结果
        """
        # Use provided parameters or defaults
        actual_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 512)
        
        # If we have a client ID, store the query in history
        # NOTE: We no longer need to add the user message here since it will be added
        # in the _create_messages method to prevent duplication
        
        # Skip RAG for empty queries
        if not query or query.strip() == "":
            # Create messages without context
            messages = self._create_messages("", "", client_id, include_history)
            formatted_context = ""
            contexts = []
        else:
            # Retrieve relevant context for non-empty queries
            contexts, formatted_context = self._retrieve_context(
                query=query,
                similarity_threshold=actual_threshold,
                top_k=actual_top_k
            )
            
            # Create messages with context and history if available
            messages = self._create_messages(query, formatted_context, client_id, include_history)
        
        # Call the API
        try:
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
        except Exception as e:
            error_message = f"LLM API 错误: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            
            # Return error information
            return {
                "completion": f"生成回复时发生错误: {str(e)}",
                "error": str(e),
                "contexts": contexts if return_context else None
            }
        
        # Handle streaming response
        if stream:
            return {"stream": response, "contexts": contexts if return_context else None}
        
        # Get the completion text
        completion_text = response.choices[0].message.content
        
        # Store the response in conversation history if we have a client ID
        if client_id:
            # First, add the user's query if it wasn't added before
            if not self._message_exists(client_id, "user", query):
                self.conversation_manager.add_message(client_id, "user", query)
                
            # Then add the assistant's response
            self.conversation_manager.add_message(client_id, "assistant", completion_text)
        
        # Build result
        result = {
            "completion": completion_text,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        # Include context information if requested
        if return_context:
            result["contexts"] = contexts
            result["formatted_context"] = formatted_context
        
        return result
    
    # Modify create_completion_stream to store streamed responses in history
    async def create_completion_stream(
        self,
        query: str,
        client_id: str,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        include_history: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        创建流式响应的异步生成器，并记录历史
        
        Args:
            query: 用户查询
            client_id: 客户端ID
            similarity_threshold: 相似度阈值（可选）
            top_k: 检索结果数量（可选）
            include_history: 是否包含历史记录
            
        Yields:
            str: 流式响应的文本块
        """
        # Store the user query in history
        await self.conversation_manager.add_message(client_id, "user", query)
        
        # Use provided parameters or defaults
        actual_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        actual_top_k = top_k if top_k is not None else self.top_k
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 512)
        
        try:
            # Skip RAG for empty queries
            if not query or query.strip() == "":
                formatted_context = ""
            else:
                # Check if index exists
                if self.index is None:
                    yield "错误: 向量数据库未加载，无法进行RAG检索。将仅使用用户查询生成回复。"
                    formatted_context = ""
                else:
                    # Retrieve relevant context
                    contexts, formatted_context = self._retrieve_context(
                        query=query,
                        similarity_threshold=actual_threshold,
                        top_k=actual_top_k
                    )
            
            # Create messages with context and history
            messages = self._create_messages(query, formatted_context, client_id, include_history)
            
            # Call API (stream mode)
            print(f"使用模型: {self.model_name} 进行流式生成")
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
            
            # Process streaming response and accumulate in history
            content_yielded = False
            accumulated_text = ""
            
            for chunk in response:
                content = None
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                elif hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'text'):
                    content = chunk.choices[0].text
                    
                if content:
                    content_yielded = True
                    accumulated_text += content
                    
                    # Store partial response in history
                    await self.conversation_manager.add_partial_response(client_id, content)
                    
                    # Yield to client
                    yield content
                    
            # If no content was yielded, provide a fallback message
            if not content_yielded:
                fallback_message = "未能生成回复。"
                await self.conversation_manager.add_message(client_id, "assistant", fallback_message)
                yield fallback_message
                    
        except Exception as e:
            error_message = f"流式响应错误: {str(e)}"
            print(error_message)
            traceback.print_exc()
            
            # Add error message to history
            await self.conversation_manager.add_message(client_id, "assistant", error_message)
            
            yield error_message
            
    # Add these synchronous helper methods to RAGService class:
    def _add_message_sync(self, client_id: str, role: str, content: str) -> None:
        """同步添加消息到历史记录"""
        if not hasattr(self.conversation_manager, '_history'):
            self.conversation_manager._history = defaultdict(list)
        self.conversation_manager._history[client_id].append({"role": role, "content": content})

    def _add_partial_response_sync(self, client_id: str, content: str) -> None:
        """同步添加部分响应到历史记录"""
        if not hasattr(self.conversation_manager, '_history'):
            self.conversation_manager._history = defaultdict(list)
            
        if self.conversation_manager._history[client_id] and self.conversation_manager._history[client_id][-1]["role"] == "assistant":
            self.conversation_manager._history[client_id][-1]["content"] += content
        else:
            self.conversation_manager._history[client_id].append({"role": "assistant", "content": content})

    # Add this helper method to check if a message already exists in history
    def _message_exists(self, client_id: str, role: str, content: str) -> bool:
        """
        检查消息是否已存在于历史记录中
        """
        if not hasattr(self.conversation_manager, '_history'):
            return False
            
        if client_id not in self.conversation_manager._history:
            return False
            
        # Check if this exact message already exists
        for msg in self.conversation_manager._history[client_id]:
            if msg["role"] == role and msg["content"] == content:
                return True
                
        return False

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