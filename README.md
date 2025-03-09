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
