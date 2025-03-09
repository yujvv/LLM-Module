#!/usr/bin/env python3
import os
import argparse
import torch
from typing import List, Dict, Any
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
# from llama_index.embeddings.openai import OpenAIEmbedding
from embedding_service import CustomEmbeddingService
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import NodeWithScore


def query_vector_db(
    query_text: str,
    persist_dir: str,
    embed_model: str = "text-embedding-3-small",
    top_k: int = 3,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    查询向量数据库获取相似文本块
    
    Args:
        query_text: 查询文本
        persist_dir: 数据库存储目录
        embed_model: OpenAI嵌入模型名称
        top_k: 返回结果数量
        verbose: 是否显示详细信息
    
    Returns:
        List[Dict]: 相似文本块列表
    """
    try:
        if not os.path.exists(persist_dir):
            raise ValueError(f"索引目录 {persist_dir} 不存在")
        
        
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise ValueError("未设置 OPENAI_API_KEY 环境变量")
            
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model_instance = CustomEmbeddingService(
            model="BAAI/bge-large-zh-v1.5", 
            device=device
        )
        # embed_model_instance = OpenAIEmbedding(model=embed_model, api_key=api_key)
        Settings.embed_model = embed_model_instance
        
        if verbose:
            print(f"加载索引自 {persist_dir}...")
        
        # 加载持久化索引
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(persist_dir),
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        
        if verbose:
            print(f"查询: '{query_text}'")
        
        # 获取检索器
        retriever = index.as_retriever(similarity_top_k=top_k)
        
        # 获取相似节点
        nodes: List[NodeWithScore] = retriever.retrieve(query_text)
        
        # 提取文本、元数据和分数
        results = []
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node
            score = node_with_score.get_score(raise_error=False)
            
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
            
            if verbose:
                print(f"\n--- 结果 {i+1} (相似度: {score:.4f}) ---")
                print(f"来源: {metadata.get('source', '未知')}")
                print(f"文本块: {text[:300]}..." if len(text) > 300 else f"文本块: {text}")
        
        return results
        
    except Exception as e:
        print(f"查询向量数据库错误: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description="查询向量数据库")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--persist_dir", default="./vector_db", help="数据库存储目录")
    parser.add_argument("--top_k", type=int, default=3, help="返回结果数量")
    args = parser.parse_args()
    
    query_vector_db(
        query_text=args.query,
        persist_dir=args.persist_dir,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()