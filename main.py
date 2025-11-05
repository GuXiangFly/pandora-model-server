#!/usr/bin/env python3
"""
@auther guxiang
@date 2025-09-15

Pandora Model Server - FastAPI嵌入服务启动脚本

该脚本用于启动基于FastAPI的OpenAI兼容嵌入服务，
提供文本向量化和维度转换功能。
"""

import uvicorn
import logging
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embedding_service.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """设置运行环境"""
    # 设置CUDA设备（如果可用）
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 设置模型路径
    if os.environ.get('MODEL_PATH') is None:
        os.environ['MODEL_PATH'] = '/Users/guxiang/dev/softwareworkspace/modelhub/Qwen/Qwen3-Embedding-0.6B'

def main(
    host: str = "0.0.0.0",
    port: int = 8019,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info"
):
    """
    启动FastAPI嵌入服务
    
    Args:
        host: 服务器监听地址
        port: 服务器监听端口
        reload: 是否启用自动重载（开发模式）
        workers: 工作进程数量
        log_level: 日志级别
    """
    try:
        # 设置环境
        setup_environment()
        
        logger.info("=" * 50)
        logger.info("启动Pandora Model Server")
        logger.info(f"监听地址: {host}:{port}")
        logger.info(f"工作进程: {workers}")
        logger.info(f"日志级别: {log_level}")
        logger.info(f"自动重载: {reload}")
        logger.info("=" * 50)
        
        # 导入FastAPI应用
        from controller.embedding_controller import app
        
        # 启动服务
        uvicorn.run(
            "controller.embedding_controller:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level
        )
        
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8019, help="监听端口 (默认: 8019)")
    parser.add_argument("--reload", action="store_true", help="启用自动重载（开发模式）")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量 (默认: 1)")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="日志级别 (默认: info)")
    
    args = parser.parse_args()
    
    main(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )