#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag_service import RAGService

# ----- 定义API模型 -----

class CompletionRequest(BaseModel):
    """完成请求模型"""
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
        openai_api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")  # 使用虚拟密钥，因为我们将直接连接到本地模型
        
        # 初始化服务
        try:
            rag_service = RAGService(
                vector_db_path=vector_db_path,
                config_path=config_path,
                openai_api_key=openai_api_key,
                openai_base_url="http://127.0.0.1:60002",  # 直接连接到指定模型服务
                openai_model="rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4"  # 使用指定模型
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"服务初始化失败: {str(e)}")
    
    return rag_service

# ----- 流式响应生成器 -----

async def stream_response(client_id: str, query: str, service: RAGService):
    """生成流式响应"""
    try:
        # 调用RAG服务的流式完成
        async for chunk in service.create_completion_stream(
            query=query,
            client_id=client_id
        ):
            yield chunk.encode('utf-8') + b'\n'
    except Exception as e:
        yield f"错误: {str(e)}".encode('utf-8')

# ----- API端点 -----

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

# 保留旧的API端点以兼容性
@app.post("/v1/completions")
async def create_completion(
    request: Request,
    service: RAGService = Depends(get_rag_service)
):
    """旧版API端点，保留以兼容性"""
    try:
        data = await request.json()
        query = data.get("query", "")
        client_id = data.get("id", "default-client")
        
        return StreamingResponse(
            stream_response(client_id, query, service),
            media_type="text/plain"
        )
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
    
'''    
url = "http://127.0.0.1:8000"
data = {"prompt": prompt, "id": self.client_id}
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=data, headers=headers, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            yield decoded_line
else:
    print(f"请求失败，状态码：{response.status_code}")
'''
