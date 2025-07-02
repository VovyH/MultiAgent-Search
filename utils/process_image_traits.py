"""
process_image_traits.py - 图像特征处理工具
用于提取和处理图像特征信息
"""
import json
import requests
import os

# API 密钥 - 应通过环境变量或配置文件获取
SILICON_FLOW_API_KEY = os.environ.get("SILICON_FLOW_API_KEY")

def extract_traits_from_otherImages(response_json):
    """
    1. 提取特征中的主要内容content
    
    Args:
        response_json: API响应的JSON字符串
        
    Returns:
        str: 提取的内容字符串
    """
    response_dict = json.loads(response_json)

    # 获取 content 的内容
    content = response_dict.get("choices", [])[0].get("message", {}).get("content", "")

    return content


def get_trait_embedding(input_text):
    """
    2. 计算文本的向量表示
    
    Args:
        input_text: 输入文本
        
    Returns:
        list: 文本的向量表示，失败返回None
    """
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": input_text,
        "encoding_format": "float"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {SILICON_FLOW_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    # 确认响应状态码是成功的
    if response.status_code == 200:
        response_data = response.json()
        # 提取embedding数据
        embeddings = response_data.get('data')[0].get('embedding')
        return embeddings
    else:
        print("Error:", response.status_code, response.text)
        return None


# 测试代码
if __name__ == "__main__":
    input_text = "测试向量计算服务"
    embedding_result = get_trait_embedding(input_text)
    if embedding_result:
        print(f"向量计算成功，长度: {len(embedding_result)}")
        print(f"向量示例: {embedding_result[:5]}")
    else:
        print("向量计算失败")
