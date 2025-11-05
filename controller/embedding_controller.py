"""
@auther guxiang
@date 2025-09-15

FastAPI嵌入服务控制器
提供OpenAI兼容的嵌入API端点
"""

from fastapi import FastAPI, HTTPException

from pojo.openapi_pojo import EmbeddingRequest, EmbeddingResponse
from service.openai_compatible_embedding_service import OpenAICompatibleEmbeddingService
import os
import logging


# 初始化嵌入服务
MODEL_PATH = os.environ.get('MODEL_PATH')
embedding_service = OpenAICompatibleEmbeddingService(MODEL_PATH)
# 创建FastAPI应用
app = FastAPI(title="OpenAI Compatible Embedding API")

logger = logging.getLogger(__name__)

# 定义API端点
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest):
    """创建嵌入向量（符合OpenAI API规范）"""
    result = embedding_service.create_embeddings(
        input_data=request.input,
        model_name=request.model,
        dimensions=request.dimensions,
        encoding_format=request.encoding_format,
        user=request.user
    )
    request_json = request.model_dump_json()
    result_response = EmbeddingResponse(**result)
    vector_example = embedding_service.get_vector_example(result_response)
    logger.info(f"embeddings request is:{request_json} vector_example is: {vector_example}")
    return result_response

