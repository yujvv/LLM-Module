#!/usr/bin/env python3
import os
import argparse
import torch
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.openai import OpenAIEmbedding
from embedding_service import CustomEmbeddingService
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

def build_vector_db(
    document_path: str,
    persist_dir: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    force_rebuild: bool = False,
    verbose: bool = True
) -> bool:
    """
    构建向量数据库，确保文本切块有效
    
    Args:
        document_path: 输入文档路径
        persist_dir: 数据库存储目录
        embed_model: OpenAI嵌入模型名称
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        force_rebuild: 是否强制重建
        verbose: 是否显示详细信息
    
    Returns:
        bool: 构建是否成功
    """
    try:
        if not force_rebuild and os.path.exists(persist_dir):
            if verbose:
                print(f"索引已存在于 {persist_dir}，使用 --force_rebuild 重建")
            return True
            
        # 设置嵌入模型
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise ValueError("未设置 OPENAI_API_KEY 环境变量")
            
        # embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model_instance = CustomEmbeddingService(
            model="BAAI/bge-large-zh-v1.5", 
            device=device
        )
        
        # 设置全局配置
        Settings.embed_model = embed_model_instance
        
        # 使用SentenceSplitter进行文本切块
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        Settings.node_parser = node_parser
        
        if verbose:
            print(f"配置文本切块: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        
        # 读取文档
        if verbose:
            print(f"读取文档: {document_path}")
            
        reader = SimpleDirectoryReader(
            input_files=[document_path],
            file_metadata=lambda _: {"source": document_path}
        )
        documents = reader.load_data()
        
        if verbose:
            print(f"加载了 {len(documents)} 个文档")
        
        # 显式进行文本切块，并打印信息
        nodes = node_parser.get_nodes_from_documents(documents)
        
        if verbose:
            print(f"文档被切分为 {len(nodes)} 个文本块")
            print(f"前3个文本块示例:")
            for i, node in enumerate(nodes[:3]):
                print(f"块 {i+1}: {node.text[:100]}..." if len(node.text) > 100 else f"块 {i+1}: {node.text}")
        
        # 创建FAISS向量存储
        vector_store = FaissVectorStore(
            faiss_index=faiss.IndexFlatL2(1536)  # OpenAI嵌入维度为1536
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # 构建索引
        if verbose:
            print("构建向量索引...")
            
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        # 持久化到磁盘
        if verbose:
            print(f"持久化索引到 {persist_dir}...")
            
        storage_context.persist(persist_dir=persist_dir)
        
        if verbose:
            print("向量数据库构建成功!")
            
        return True
        
    except Exception as e:
        print(f"构建向量数据库错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="构建向量数据库并验证文本切块")
    parser.add_argument("--document", required=True, help="输入文档路径")
    parser.add_argument("--persist_dir", default="./vector_db", help="数据库存储目录")
    parser.add_argument("--chunk_size", type=int, default=512, help="文本块大小")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="文本块重叠大小")
    parser.add_argument("--force_rebuild", action="store_true", help="强制重建数据库")
    args = parser.parse_args()
    
    build_vector_db(
        document_path=args.document,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_rebuild=args.force_rebuild
    )

if __name__ == "__main__":
    main()