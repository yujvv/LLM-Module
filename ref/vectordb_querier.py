import os
from typing import Optional, List, Dict
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import NodeWithScore

class VectorDBQuerier:
    def __init__(
        self,
        persist_dir: str,
        embed_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize vector database querier
        
        Args:
            persist_dir: Directory containing persisted index
            embed_model: OpenAI embedding model name
            openai_api_key: OpenAI API key (default uses OPENAI_API_KEY env)
        """
        if not os.path.exists(persist_dir):
            raise ValueError(f"Index directory {persist_dir} does not exist")
            
        self.embed_model = OpenAIEmbedding(
            model=embed_model,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        Settings.embed_model = self.embed_model
        
        # Load persisted index
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(persist_dir),
            persist_dir=persist_dir
        )
        self.index = load_index_from_storage(storage_context)

    def query_similar_chunks(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Query similar chunks from vector database
        
        Args:
            query: Query text
            top_k: Number of similar chunks to return
            
        Returns:
            List of dicts containing chunk text and metadata
        """
        # Get retriever (no response synthesis)
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # Get similar nodes
        nodes: List[NodeWithScore] = retriever.retrieve(query)
        
        # Extract text, metadata and score
        results = []
        for node_with_score in nodes:
            node = node_with_score.node
            
            # Handle Document type
            if hasattr(node, 'text'):
                text = node.text
            else:
                text = str(node)
                
            results.append({
                "text": text,
                "metadata": getattr(node, 'metadata', {}),
                "score": node_with_score.get_score(raise_error=False)
            })
            
        return results
    
    # [
    #     {
    #         "text": "文本内容",
    #         "metadata": {"source": "文档路径", ...},
    #         "score": 0.95  # 相似度分数
    #     },
    #     ...
    # ]