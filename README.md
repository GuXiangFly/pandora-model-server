## 关于这个项目
 一个使用sentence transformer 的算法启动 QWEN-embedding-0.6 -4B -8B   等模型的方案


## quick start

### 启动该项目
```bash 

1. 安装UV   (已安装的跳过)
pip install  uv  

2. 使用 uv sync下载依赖
uv sync 

3. 在 start.sh 或者 start_uv.sh 中修改模型为自己电脑的模型路径

4. chmod  +x start.sh   或者   chmod +x start_uv.sh

5. ./start.sh   或者  ./start_uv.sh
```



## 使用OpenAI的 api调用如下
并且提供兼容OpenAI规范的 embedding接口。
调用的使用demo如下
```python
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

```


## Http API调用如下
```bash
curl --location --request POST 'http://127.0.0.1:8019/v1/embeddings' \
--header 'Content-Type: application/json' \
--data-raw '{
    "input": ["你好啊"],
    "model": "Qwen3-Embedding-0.6B",
    "dimensions": 1024
}'

```

结果如下
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.009208028204739094,
                0.01649611070752144,
                -0.010724461637437344,
                -0.02824822999536991,
                0.0037427295465022326,
                  .....
            ],
            "index": 0
        }
    ],
    "model": "Qwen3-Embedding-0.6B",
    "usage": {
        "prompt_tokens": 1,
        "total_tokens": 1
    }
}
```