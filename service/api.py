#!/usr/bin/env python3
import os
import traceback
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag_service import RAGService

# ----- 定义API模型 -----

class CompletionRequest(BaseModel):
    """完成请求模型 - 新格式"""
    prompt: str = Field(..., description="用户查询文本")
    id: str = Field(..., description="客户端ID")

# ----- 创建应用 -----

app = FastAPI(
    title="RAG API服务",
    description="提供RAG增强的流式响应API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- 全局服务实例 -----
# 延迟初始化RAG服务
rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """获取或初始化RAG服务实例"""
    global rag_service
    if rag_service is None:
        # 从环境变量获取配置
        vector_db_path = os.environ.get("VECTOR_DB_PATH", "./vector_db")
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        
        # 明确设置为localhost/127.0.0.1，与直接调用中使用的一致
        model_base_url = "http://127.0.0.1:60002"
        
        # 确保base_url以/v1结尾
        if not model_base_url.endswith('/v1'):
            model_base_url = f"{model_base_url}/v1"
            
        # 使用llm_sample_code.py中的模型名称，确保一致性
        model_name = "rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4"
        
        # 初始化服务，但允许向量数据库加载失败
        try:
            print(f"正在初始化RAG服务，使用向量库: {vector_db_path}, 模型: {model_name}, 端点: {model_base_url}")
            rag_service = RAGService(
                vector_db_path=vector_db_path,
                config_path=config_path,
                openai_api_key="dummy-key",  # 使用虚拟密钥，因为我们连接的是本地Qwen模型
                openai_base_url=model_base_url,  # 强制使用localhost作为端点
                openai_model=model_name,  # 使用指定的模型
                require_vector_db=False  # 即使向量数据库加载失败，也继续初始化服务
            )
            print("RAG服务初始化完成")
        except Exception as e:
            error_detail = f"服务初始化失败: {str(e)}"
            print(f"错误: {error_detail}")
            # traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_detail)
    
    return rag_service

# Add these models to the existing models in api.py
class HistoryRequest(BaseModel):
    """History request model"""
    id: str = Field(..., description="客户端ID")

class HistoryResponse(BaseModel):
    """History response model"""
    history: List[Dict[str, str]] = Field(default_factory=list, description="对话历史")
    success: bool = Field(True, description="操作是否成功")
    message: str = Field("", description="操作消息")

# Add these new API endpoints to api.py
@app.post("/clear_history")
async def clear_history(
    request: HistoryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """清除客户端的对话历史"""
    try:
        success = service.conversation_manager.clear_history(request.id)
        
        if success:
            return HistoryResponse(
                success=True,
                message="历史记录已清除"
            )
        else:
            return HistoryResponse(
                success=False,
                message="未找到该客户端的历史记录"
            )
    except Exception as e:
        print(f"清除历史记录错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the get_history endpoint:
@app.post("/get_history")
async def get_history(
    request: HistoryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """获取客户端的对话历史"""
    try:
        history = service.conversation_manager.get_history(request.id)
        
        return HistoryResponse(
            history=history,
            success=True,
            message="获取历史记录成功"
        )
    except Exception as e:
        print(f"获取历史记录错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# In api.py, update the stream_response function:

async def stream_response(client_id: str, query: str, service: RAGService):
    """生成流式响应，并记录历史"""
    try:
        print(f"处理来自客户端 {client_id} 的请求: {query[:50]}..." if len(query) > 50 else f"处理来自客户端 {client_id} 的请求: {query}")
        
        # Add user message to history if it doesn't exist yet
        if not service._message_exists(client_id, "user", query):
            service.conversation_manager.add_message(client_id, "user", query)
        
        # 使用标准参数
        actual_threshold = service.default_similarity_threshold
        actual_top_k = service.top_k
        temperature = service.config.get("temperature", 0.7)
        max_tokens = service.config.get("max_tokens", 512)
        
        # Skip RAG for empty queries
        if not query or query.strip() == "":
            formatted_context = ""
        else:
            # Check if index exists
            if service.index is None:
                error_message = "错误: 向量数据库未加载，无法进行RAG检索。将仅使用用户查询生成回复。"
                yield error_message.encode('utf-8') + b'\n'
                formatted_context = ""
            else:
                # Retrieve relevant context
                contexts, formatted_context = service._retrieve_context(
                    query=query,
                    similarity_threshold=actual_threshold,
                    top_k=actual_top_k
                )
        
        # Create messages with context and history
        messages = service._create_messages(query, formatted_context, client_id, True)
        
        # Call API (stream mode)
        print(f"使用模型: {service.model_name} 进行流式生成")
        response = service.client.chat.completions.create(
            model=service.model_name,
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
        
        # Create a placeholder for the assistant's response
        assistant_message_index = None
        
        for chunk in response:
            # Initialize content to None for each chunk
            content = None
            
            # Try to extract content based on the response format
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                # OpenAI-style response format
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:  # Check explicitly for None
                    content = delta_content
            elif hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'text'):
                # Alternative response format
                text_content = chunk.choices[0].text
                if text_content is not None:  # Check explicitly for None
                    content = text_content
                
            # Only process and yield if we got actual content
            if content is not None and content != "":
                content_yielded = True
                accumulated_text += content
                
                # Update or create the assistant's message in history
                if assistant_message_index is None:
                    # Add a new message
                    service.conversation_manager.add_message(client_id, "assistant", content)
                    # Get the index of this message
                    assistant_message_index = len(service.conversation_manager._history[client_id]) - 1
                else:
                    # Update existing message
                    service.conversation_manager._history[client_id][assistant_message_index]["content"] += content
                
                # Yield to client
                yield content.encode('utf-8') + b'\n'
                
        # If no content was yielded, provide a fallback message
        if not content_yielded:
            fallback_message = "未能生成回复。"
            service.conversation_manager.add_message(client_id, "assistant", fallback_message)
            yield fallback_message.encode('utf-8') + b'\n'
                
    except Exception as e:
        error_message = f"流式响应错误: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        
        # Add error message to history
        service.conversation_manager.add_message(client_id, "assistant", error_message)
        
        yield error_message.encode('utf-8') + b'\n'

# ----- 非流式API端点 -----

# Modify the non-streaming completions endpoint to include client_id
@app.post("/v1/completions")
async def create_completion(
    request: Request,
    service: RAGService = Depends(get_rag_service)
):
    """
    非流式完成API
    """
    try:
        # 获取请求数据
        data = await request.json()
        query = data.get("query", "")
        client_id = data.get("id", None)  # Add client_id parameter
        similarity_threshold = data.get("similarity_threshold", 2.0)
        top_k = data.get("top_k", 5)
        temperature = data.get("temperature", 0.7)
        return_context = data.get("return_context", False)
        include_history = data.get("include_history", True)  # Add history parameter
        
        # 使用标准的(非流式)完成方法
        result = service.create_completion(
            query=query,
            client_id=client_id,  # Pass client_id
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            temperature=temperature,
            return_context=return_context,
            include_history=include_history  # Use history parameter
        )
        
        return result
    except Exception as e:
        print(f"API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----- 流式响应生成器 -----

# async def stream_response(client_id: str, query: str, service: RAGService):
#     """生成流式响应"""
#     try:
#         print(f"处理来自客户端 {client_id} 的请求: {query[:50]}..." if len(query) > 50 else f"处理来自客户端 {client_id} 的请求: {query}")
#         # 调用RAG服务的流式完成
#         async for chunk in service.create_completion_stream(
#             query=query,
#             client_id=client_id
#         ):
#             yield chunk.encode('utf-8') + b'\n'
#     except Exception as e:
#         error_message = f"流式响应生成错误: {str(e)}"
#         print(error_message)
#         # traceback.print_exc()
#         yield error_message.encode('utf-8')

# ----- 流式API端点 -----

@app.post("/")
async def root(
    request: CompletionRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    主要API端点，接收prompt和id，返回流式响应
    """
    return StreamingResponse(
        stream_response(request.id, request.prompt, service),
        media_type="text/plain"
    )

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

# ----- 启动服务器 -----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)