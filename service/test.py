#!/usr/bin/env python3
import os
import requests
import json
from pprint import pprint

# 直接使用RAGService类
def use_rag_service_directly():
    """直接使用RAGService类的示例"""
    from rag_service import RAGService
    
    # 初始化服务
    try:
        service = RAGService(
            vector_db_path="./vector_db",                       # 向量数据库路径
            config_path="config.yaml",                          # 配置文件路径
            openai_model="rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4"  # 使用与llm_sample_code.py相同的模型
        )
    except Exception as e:
        print(f"初始化服务错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
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

# 通过API使用服务 - 非流式
def use_rag_api_non_streaming():
    """通过非流式API使用服务的示例"""
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
    
    print(f"发送请求到: {api_url}")
    print(f"请求数据: {json.dumps(payload)}")
    
    # 发送POST请求
    try:
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # 打印回复
            print("\n====== API回复内容 ======")
            print(result.get("completion", "无回复内容"))
            
            # 打印使用情况
            if "usage" in result:
                print("\n====== API令牌使用情况 ======")
                print(f"提示词令牌: {result['usage'].get('prompt_tokens', 'N/A')}")
                print(f"完成令牌: {result['usage'].get('completion_tokens', 'N/A')}")
                print(f"总令牌: {result['usage'].get('total_tokens', 'N/A')}")
            
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
    except Exception as e:
        print(f"请求错误: {e}")

# 通过API使用服务 - 流式
def use_rag_api_streaming():
    """通过流式API使用服务的示例"""
    # API端点URL
    api_url = "http://localhost:8000"
    
    # 新格式的请求数据
    payload = {
        "prompt": "こんにちは、アロナさん。今日の予定を教えてください。",
        "id": "test-client-123"
    }
    
    print(f"发送请求到: {api_url}/")
    print(f"请求数据: {json.dumps(payload)}")
    
    # 发送POST请求并获取流式响应
    try:
        with requests.post(api_url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                print(response.text)
                return
                
            print("====== API流式响应内容 ======")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    print(chunk, end="", flush=True)
                    full_response += chunk
            print("\n")  # 完成后换行
            
            print("====== 完整响应 ======")
            print(full_response)
    except Exception as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    # 选择一种方式运行
    print("=== 直接使用RAGService类 ===")
    use_rag_service_directly()
    
    print("\n\n=== 通过API使用服务(非流式) ===")
    use_rag_api_non_streaming()
    
    print("\n\n=== 通过API使用服务(流式) ===")
    use_rag_api_streaming()
