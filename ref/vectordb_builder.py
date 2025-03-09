import os
from typing import Optional, List, Dict
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

class VectorDBBuilder:
    def __init__(
        self,
        embed_model: str = "text-embedding-3-small",
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        # openai_api_key: Optional[str] = None
    ):
        """
        Initialize vector database builder
        
        Args:
            embed_model: OpenAI embedding model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            openai_api_key: OpenAI API key (default uses OPENAI_API_KEY env)
        """
        self.embed_model = OpenAIEmbedding(
            model=embed_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure global settings
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def build_and_persist(
        self,
        document_path: str,
        persist_dir: str,
        force_rebuild: bool = False
    ) -> bool:
        """
        Build vector database from document and persist to disk
        
        Args:
            document_path: Path to input document
            persist_dir: Directory for persistent storage
            force_rebuild: Whether to force rebuild existing index
            
        Returns:
            bool: True if build/persist successful
        """
        try:
            if not force_rebuild and os.path.exists(persist_dir):
                print(f"Index already exists at {persist_dir}")
                return True
                
            # Read document
            reader = SimpleDirectoryReader(
                input_files=[document_path],
                file_metadata=lambda _: {"source": document_path}
            )
            documents = reader.load_data()

            # Create FAISS vector store (1536 is OpenAI embedding dimension)
            vector_store = FaissVectorStore(
                faiss_index=faiss.IndexFlatL2(1536)
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            # Build index
            VectorStoreIndex(
                nodes=documents,
                storage_context=storage_context,
                show_progress=True
            )

            # Persist to disk
            storage_context.persist(persist_dir=persist_dir)
            return True
            
        except Exception as e:
            print(f"Error building/persisting index: {e}")
            return False