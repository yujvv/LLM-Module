# LLM-Module
LLM-Module for the digital human

## 设置OpenAI API密钥
export OPENAI_API_KEY="your-openai-api-key"

## 1. 构建向量数据库
python build_vectordb.py --document test.md --persist_dir ./vector_db --chunk_size 256 --chunk_overlap 30

## 2. 查询向量数据库
python query_vectordb.py --query "查询内容" --persist_dir ./vector_db --top_k 3

## 3. 使用更大的块大小重建索引
python build_vectordb.py --document test.md --persist_dir ./vector_db --chunk_size 1024 --chunk_overlap 50 --force_rebuild


## 特点

- 提供类似OpenAI的API接口
- 利用RAG增强回答质量
- 可配置的相似度阈值和检索参数
- 支持自定义系统提示词和模型参数
- 使用开源嵌入模型(BAAI/bge-large-zh)进行文档检索

## 安装与设置

### 前提条件

- Python 3.8+
- 有效的OpenAI API密钥
- 预先准备好的向量数据库（使用提供的`builder_script.py`构建）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

1. 创建`config.yaml`文件或使用默认配置
2. 设置环境变量`OPENAI_API_KEY`或在代码中提供

```bash
export OPENAI_API_KEY=your_api_key_here
```

## 使用方法

### 作为Python模块使用

```python
from rag_service import RAGService

# 初始化服务
service = RAGService(
    vector_db_path="./vector_db",
    config_path="config.yaml",
    openai_model="gpt-3.5-turbo"
)

# 创建完成
result = service.create_completion(
    query="量子计算机的工作原理是什么？",
    similarity_threshold=0.75,
    return_context=True
)

# 使用结果
print(result["completion"])
```

### 通过API使用

1. 启动API服务器：

```bash
python api_server.py
```

2. 发送请求：

```python
import requests

api_url = "http://localhost:8000/v1/completions"

payload = {
    "query": "量子计算机的工作原理是什么？",
    "similarity_threshold": 0.75,
    "return_context": True
}

response = requests.post(api_url, json=payload)
result = response.json()
print(result["completion"])
```

### 使用Docker

```bash
# 构建并启动服务
docker-compose up -d

# 停止服务
docker-compose down
```

## 构建向量数据库

使用`builder_script.py`构建向量数据库：

```bash
python builder_script.py --document your_document.pdf --persist_dir ./vector_db
```

## 配置文件

`config.yaml`支持以下配置：

```yaml
# 系统提示词
system_prompt: "你是一个专业的AI助手。请基于提供的上下文回答用户的问题。"

# RAG相关参数
similarity_threshold: 0.75  # 相似度阈值
top_k: 5                    # 默认检索结果数量

# 语言模型参数
temperature: 0.7            # 温度参数
max_tokens: 1024            # 最大生成token数
```
