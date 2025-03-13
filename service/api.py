#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_service import RAGService

# ----- 定义API模型 -----

class CompletionRequest(BaseModel):
    """完成请求模型"""
    query: str = Field(..., description="用户查询文本")
    similarity_threshold: Optional[float] = Field(None, description="相似度阈值，低于此值的结果将被过滤")
    top_k: Optional[int] = Field(None, description="检索的最大结果数量")
    temperature: Optional[float] = Field(None, description="温度参数，控制输出的随机性")
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    stream: bool = Field(False, description="是否流式返回")
    return_context: bool = Field(False, description="是否返回检索的上下文")

class CompletionResponse(BaseModel):
    """完成响应模型"""
    completion: str = Field(..., description="生成的回复")
    model: str = Field(..., description="使用的模型")
    usage: Dict[str, int] = Field(..., description="token使用情况")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="检索的上下文")
    formatted_context: Optional[str] = Field(None, description="格式化后的上下文")

# ----- 创建应用 -----

app = FastAPI(
    title="RAG API服务",
    description="提供类似OpenAI的API接口，但在内部使用RAG增强回答质量",
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
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        
        # 初始化服务
        try:
            rag_service = RAGService(
                vector_db_path=vector_db_path,
                config_path=config_path,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_model=openai_model,
                embedding_model=embedding_model
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"服务初始化失败: {str(e)}")
    
    return rag_service

# ----- API端点 -----

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    创建RAG增强的完成
    
    返回结果类似于OpenAI的API响应，但内部使用RAG增强了回答质量
    """
    try:
        # 调用服务
        result = service.create_completion(
            query=request.query,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            return_context=request.return_context
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

# ----- 启动服务器 -----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)