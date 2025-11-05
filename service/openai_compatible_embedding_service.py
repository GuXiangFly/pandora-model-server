"""
@auther guxiang
@date 2025-09-15

OpenAI兼容的嵌入服务类
提供文本向量化和维度转换功能
"""

from fastapi import FastAPI, HTTPException, Depends
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import torch
from sklearn.decomposition import PCA
import numpy as np
import logging
import base64

from pojo.openapi_pojo import EmbeddingResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingService:
    """OpenAI兼容的嵌入服务类"""

    def __init__(self, model_path: str):
        """初始化服务，加载模型"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.original_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = self._get_model_name(model_path)

        # 打印初始化信息
        print(f"===== 模型加载完成 =====")
        print(f"模型名称: {self.model_name}")
        print(f"原始维度: {self.original_dim}")
        print(f"运行设备: {self.device}")

    def _load_model(self, model_path: str) -> SentenceTransformer:
        """加载嵌入模型"""
        try:
            return SentenceTransformer(model_path, device=self.device)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _get_model_name(self, model_path: str) -> str:
        """从路径中提取模型名称"""
        return model_path.split("/")[-1]

    def _validate_model(self, model_name: Optional[str]) -> None:
        """验证模型是否支持"""
        if model_name is not None and model_name != self.model_name:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {model_name}，当前仅支持 {self.model_name}"
            )

    def _validate_target_dim(self, target_dim: Optional[int]) -> None:
        """验证目标维度是否有效"""
        if target_dim is not None:
            if target_dim <= 0 or target_dim > self.original_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"dimensions必须在1到{self.original_dim}之间（原始维度）",
                    headers={"X-Error-Type": "invalid_request_error"}
                )

    def _validate_encoding_format(self, encoding_format: Optional[str]) -> None:
        """验证编码格式"""
        if encoding_format is not None and encoding_format not in ["float", "base64"]:
            raise HTTPException(
                status_code=400,
                detail="encoding_format必须是'float'或'base64'",
                headers={"X-Error-Type": "invalid_request_error"}
            )

    def _count_tokens(self, texts: List[str]) -> int:
        """估算令牌数量（实际应用中应使用模型的tokenizer）"""
        # 简单估算：每个中文按1token，英文单词按1token
        return sum(len(text.split()) if text.strip() else 0 for text in texts)

    def _process_input(self, input_data: Union[str, List[str]]) -> List[str]:
        """处理输入，统一转换为列表格式"""
        if isinstance(input_data, str):
            return [input_data]
        return input_data

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成原始嵌入向量"""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device
        )

    def _encode_base64(self, vector: np.ndarray) -> str:
        """将向量编码为base64字符串"""
        # 转换为32位浮点数组，然后编码为base64
        return base64.b64encode(vector.astype(np.float32).tobytes()).decode('utf-8')

    def _reduce_dimensions(self, vectors: np.ndarray, target_dim: int) -> np.ndarray:
        """进行维度 reduction - 根据样本数量选择合适的方法"""
        n_samples, n_features = vectors.shape

        if target_dim == n_features:
            return vectors

        logger.info(f"需要进行降维处理 target_dim:{target_dim} n_features:{n_features}")
        # 当样本数较少时，使用随机投影而不是PCA
        if n_samples < target_dim:
            from sklearn.random_projection import GaussianRandomProjection
            rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
            return rp.fit_transform(vectors)
        else:
            # 样本充足时使用PCA
            pca = PCA(n_components=target_dim)
            return pca.fit_transform(vectors)

    def get_vector_example(self, result_response: EmbeddingResponse):
        try:
            vector_example = result_response.data[0].embedding[:3]
        except Exception as e:
            vector_example = []
        return vector_example

    def create_embeddings(self, input_data: Union[str, List[str]],
                          model_name: Optional[str] = None,
                          dimensions: Optional[int] = None,
                          encoding_format: Optional[str] = "float",
                          user: Optional[str] = None) -> dict:
        """创建嵌入向量的主方法"""
        # 验证参数
        self._validate_model(model_name)
        self._validate_target_dim(dimensions)
        self._validate_encoding_format(encoding_format)

        # 处理输入
        texts = self._process_input(input_data)

        # 计算令牌使用量
        prompt_tokens = self._count_tokens(texts)

        # 生成嵌入
        vectors = self._generate_embeddings(texts)

        # 维度转换
        if dimensions is not None:
            vectors = self._reduce_dimensions(vectors, dimensions)

        # 构造响应
        return self._build_response(vectors, texts, prompt_tokens, encoding_format or "float")

    def _build_response(self, vectors: np.ndarray, texts: List[str], prompt_tokens: int, encoding_format: str) -> dict:
        """构建符合OpenAI规范的响应"""
        data = []
        for i, vector in enumerate(vectors):
            if encoding_format == "base64":
                embedding_data = self._encode_base64(vector)
            else:
                embedding_data = vector.tolist()

            data.append({
                "object": "embedding",
                "embedding": embedding_data,
                "index": i
            })

        return {
            "object": "list",
            "data": data,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens
            }
        }
