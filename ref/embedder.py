import torch
from sentence_transformers import SentenceTransformer

class TextEmbeddingService:
    """
    文本嵌入服务类，用于将文本转换为向量表示
    """
    
    def __init__(self, model_path="BAAI/bge-large-zh-v1.5", device=None):
        """
        初始化文本嵌入服务
        
        参数:
            model_path (str): 预训练模型的路径或名称
            device (str, optional): 计算设备，如果为None，将自动检测可用设备
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)
    
    def get_embeddings(self, texts, normalize=True, batch_size=32, return_single=False):
        """
        获取文本的嵌入向量
        
        参数:
            texts (str or list): 单个文本或文本列表
            normalize (bool): 是否对嵌入向量进行归一化
            batch_size (int): 批处理大小，用于处理大量文本
            return_single (bool): 如果为True且输入为单个文本，则返回单个向量而非数组
            
        返回:
            numpy.ndarray: 嵌入向量或嵌入向量列表
        """
        is_single_text = isinstance(texts, str)
        if is_single_text:
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size
        )
        
        # 如果是单个文本且需要返回单个向量（而非数组），则返回第一个元素
        if is_single_text and return_single:
            return embeddings[0]
        
        return embeddings
    
    def get_embedding_dimension(self):
        """
        获取嵌入向量的维度
        
        返回:
            int: 嵌入向量的维度
        """
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, text1, text2):
        """
        计算两个文本之间的余弦相似度
        
        参数:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            
        返回:
            float: 余弦相似度，范围在[-1, 1]之间
        """
        emb1 = self.get_embeddings(text1, return_single=True)
        emb2 = self.get_embeddings(text2, return_single=True)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0), 
            torch.tensor(emb2).unsqueeze(0)
        ).item()
        
        return similarity

# 使用示例:
# 初始化服务
# embedding_service = TextEmbeddingService()
# 
# # 获取单个文本的嵌入（作为数组）
# text = "这是一个测试文本"
# embedding = embedding_service.get_embeddings(text)
# print(f"嵌入维度: {embedding.shape}")
# 
# # 获取单个文本的嵌入（作为单个向量）- 与原始代码行为一致
# query = "查询文本"
# query_embedding = embedding_service.get_embeddings(query, return_single=True)
# print(f"嵌入形状: {query_embedding.shape}")
# 
# # 获取多个文本的嵌入
# texts = ["这是第一个文本", "这是第二个文本", "这是第三个文本"]
# embeddings = embedding_service.get_embeddings(texts)
# print(f"嵌入维度: {embeddings.shape}")
# 
# # 计算两个文本的相似度
# text1 = "我喜欢吃苹果"
# text2 = "苹果是我最喜欢的水果"
# similarity = embedding_service.similarity(text1, text2)
# print(f"相似度: {similarity}")