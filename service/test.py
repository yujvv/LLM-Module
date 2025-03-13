#!/usr/bin/env python3
import os
import requests
import json
from pprint import pprint

# 直接使用RAGService类
def use_rag_service_directly():
    """直接使用RAGService类的示例"""
    from rag_service import RAGService
    
    # 设置OpenAI API密钥（如果未设置环境变量）
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # 初始化服务
    service = RAGService(
        vector_db_path="./vector_db",     # 向量数据库路径
        config_path="config.yaml",        # 配置文件路径
        openai_model="gpt-3.5-turbo"      # 可选：指定OpenAI模型
    )
    
    # 创建完成
    result = service.create_completion(
        query="量子计算机的工作原理是什么？",    # 用户查询
        similarity_threshold=0.75,            # 可选：相似度阈值
        top_k=3,                              # 可选：检索结果数量
        temperature=0.7,                      # 可选：温度参数
        return_context=True                   # 可选：返回检索的上下文
    )
    
    # 打印回复
    print("\n====== 回复内容 ======")
    print(result["completion"])
    
    # 打印使用情况
    print("\n====== 令牌使用情况 ======")
    print(f"提示词令牌: {result['usage']['prompt_tokens']}")
    print(f"完成令牌: {result['usage']['completion_tokens']}")
    print(f"总令牌: {result['usage']['total_tokens']}")
    
    # 打印上下文（如果请求）
    if "contexts" in result:
        print("\n====== 检索到的上下文 ======")
        for i, ctx in enumerate(result["contexts"], 1):
            print(f"上下文 {i} (相似度: {ctx['score']:.4f})")
            print(f"来源: {ctx['metadata'].get('source', '未知')}")
            print(f"文本: {ctx['text'][:150]}..." if len(ctx['text']) > 150 else f"文本: {ctx['text']}")
            print()

# 通过API使用服务
def use_rag_api():
    """通过API使用服务的示例"""
    # API端点URL
    api_url = "http://localhost:8000/v1/completions"
    
    # 请求数据
    payload = {
        "query": "量子计算机的工作原理是什么？",
        "similarity_threshold": 0.75,
        "top_k": 3,
        "temperature": 0.7,
        "return_context": True
    }
    
    # 发送POST请求
    response = requests.post(api_url, json=payload)
    
    # 检查响应
    if response.status_code == 200:
        result = response.json()
        
        # 打印回复
        print("\n====== API回复内容 ======")
        print(result["completion"])
        
        # 打印使用情况
        print("\n====== API令牌使用情况 ======")
        print(f"提示词令牌: {result['usage']['prompt_tokens']}")
        print(f"完成令牌: {result['usage']['completion_tokens']}")
        print(f"总令牌: {result['usage']['total_tokens']}")
        
        # 打印上下文（如果请求）
        if "contexts" in result:
            print("\n====== API检索到的上下文 ======")
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"上下文 {i} (相似度: {ctx['score']:.4f})")
                print(f"来源: {ctx['metadata'].get('source', '未知')}")
                print(f"文本: {ctx['text'][:150]}..." if len(ctx['text']) > 150 else f"文本: {ctx['text']}")
                print()
    else:
        print(f"API请求失败: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # 选择一种方式运行
    print("=== 直接使用RAGService类 ===")
    use_rag_service_directly()
    
    print("\n\n=== 通过API使用服务 ===")
    use_rag_api()