from langchain_openai import ChatOpenAI, OpenAIEmbeddings

## 如果不使用代理，此处可以删掉
import os
os.environ['NO_PROXY'] = '127.0.0.1'



openAIEmbedding_model = OpenAIEmbeddings(
    model='Qwen3-Embedding-0.6B',
    base_url="http://127.0.0.1:8019/v1",
    dimensions=1024,
    api_key="test",
    check_embedding_ctx_length=False,  # 禁用上下文长度检查，避免tokenization
    skip_empty=True,
    chunk_size=1,  # 减小块大小
    embedding_ctx_length=8191,  # 设置最大长度
)



print(openAIEmbedding_model.embed_query("你好啊"))