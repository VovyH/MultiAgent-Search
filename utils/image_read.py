"""
image_read.py - 图像处理工具函数
用于读取和编码图像数据
"""
import base64
import requests

def encode_image(file_path_or_url):
    """
    读取图像并将其转换为Base64编码
    
    Args:
        file_path_or_url: 图像的本地路径或URL
        
    Returns:
        str: Base64编码的图像字符串，失败返回None
    """
    try:
        if file_path_or_url.startswith('http'):
            response = requests.get(file_path_or_url)
            image_data = response.content
        # 处理本地图像
        else:
            with open(file_path_or_url, "rb") as image_file:
                image_data = image_file.read()
                
        # 编码为Base64
        return base64.b64encode(image_data).decode('ascii')
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

