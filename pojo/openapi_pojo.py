"""
@auther guxiang
@date 2025-09-15

OpenAPI数据模型定义
定义嵌入服务的请求和响应数据模型
"""

from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import torch
from sklearn.decomposition import PCA
import math

# 请求模型定义
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="要嵌入的输入文本（字符串或字符串列表）")
    model: Optional[str] = Field(None, description=f"使用的模型名称")
    dimensions: Optional[int] = Field(None, description=f"目标维度")
    encoding_format: Optional[str] = Field("float", description="编码格式：float或base64")
    user: Optional[str] = Field(None, description="用户标识符，用于滥用检测")


# 响应模型定义
class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str] = Field(..., description="嵌入向量或base64编码字符串")
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage
