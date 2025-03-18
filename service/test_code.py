from openai import OpenAI
import requests
import json

# LLM API 设置
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:60002/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4"

# RAG API 设置
rag_api_url = "http://localhost:8000"  # RAG 服务地址


def generate_with_llm(history):
    """直接使用LLM生成回复（不使用RAG）"""
    result = client.chat.completions.create(
            model=model,
            messages=history,
            # stream = True,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
    return result


def generate_with_rag_api(prompt, client_id="default_client"):
    """使用RAG API生成回复，支持会话历史"""
    response = requests.post(
        f"{rag_api_url}/v1/completions",
        json={
            "query": prompt,
            "id": client_id,          # 客户端ID，用于追踪会话历史
            "return_context": True,   # 返回检索的上下文
            "include_history": True   # 包含会话历史
        }
    )
    return response.json()


def generate_with_rag_stream(prompt, client_id="default_client"):
    """使用RAG API生成流式回复，支持会话历史"""
    response = requests.post(
        f"{rag_api_url}/",
        json={
            "prompt": prompt,
            "id": client_id  # 客户端ID，用于追踪会话历史
        },
        stream=True
    )
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            full_response += chunk
            print(chunk, end="", flush=True)  # 实时显示生成内容
    
    print("\n")  # 换行
    return full_response


def clear_conversation_history(client_id="default_client"):
    """清除指定客户端的会话历史"""
    response = requests.post(
        f"{rag_api_url}/clear_history",
        json={"id": client_id}
    )
    return response.json()


# In test_code.py, enhance error handling:
def get_conversation_history(client_id="default_client"):
    """获取指定客户端的会话历史"""
    try:
        response = requests.post(
            f"{rag_api_url}/get_history",
            json={"id": client_id}
        )
        
        # Check for response errors
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取历史记录错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_msg = e.response.json()
                print(f"服务器返回: {error_msg}")
                return error_msg  # Return the error message from server
            except:
                print(f"服务器返回状态码: {e.response.status_code}")
        return {"history": [], "success": False, "message": str(e)}


def demo_conversation():
    """演示多轮对话功能"""
    client_id = "user123"  # 唯一的客户端ID
    
    # 首先清除这个客户端的历史记录（如果有的话）
    clear_result = clear_conversation_history(client_id)
    print(f"清除历史记录: {clear_result}\n")
    
    # 第一轮对话
    print("=== 第一轮对话 ===")
    print("用户: 你好，我是谁？")
    response1 = generate_with_rag_stream("你好，我是谁？", client_id)
    
    # 查看当前历史记录
    history = get_conversation_history(client_id)
    print("\n当前历史记录:")
    for msg in history["history"]:
        print(f"{msg['role']}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{msg['role']}: {msg['content']}")
    
    # 第二轮对话，引用上下文
    print("\n=== 第二轮对话 ===")
    print("用户: 你能告诉我更多关于自己的信息吗？")
    response2 = generate_with_rag_stream("你能告诉我更多关于自己的信息吗？", client_id)
    
    # 查看更新后的历史记录
    history = get_conversation_history(client_id)
    print("\n更新后的历史记录:")
    for msg in history["history"]:
        print(f"{msg['role']}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{msg['role']}: {msg['content']}")
    
    # 第三轮对话，测试记忆
    print("\n=== 第三轮对话 ===")
    print("用户: 刚才我们聊了什么？")
    response3 = generate_with_rag_stream("刚才我们聊了什么？", client_id)
    
    # 清除历史记录
    clear_result = clear_conversation_history(client_id)
    print(f"\n清除历史记录: {clear_result}")
    
    # 验证历史已清除
    history = get_conversation_history(client_id)
    print("清除后的历史记录:")
    print(history)


def compare_with_without_history():
    """比较有无历史记录的回复差异"""
    client_id = "user456"
    
    # 清除历史
    clear_conversation_history(client_id)
    
    # 第一轮对话
    print("=== 有历史记录 ===")
    print("用户: 我喜欢吃甜食")
    generate_with_rag_stream("我喜欢吃甜食", client_id)
    
    print("\n用户: 你还记得我喜欢吃什么吗？")
    generate_with_rag_stream("你还记得我喜欢吃什么吗？", client_id)
    
    # 清除历史后重新提问
    clear_conversation_history(client_id)
    
    print("\n=== 无历史记录 ===")
    print("用户: 你还记得我喜欢吃什么吗？")
    generate_with_rag_stream("你还记得我喜欢吃什么吗？", client_id)


# 测试 LLM 直接调用
def test_direct_llm():
    print("\n=== 测试直接调用LLM ===")
    history = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]
    response = generate_with_llm(history)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    print("=== 测试RAG API带历史记录的对话 ===")
    demo_conversation()
    
    print("\n\n=== 对比有无历史记录的回复差异 ===")
    compare_with_without_history()
    
    # 测试直接LLM调用
    test_direct_llm()