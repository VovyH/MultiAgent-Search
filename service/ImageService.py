from utils.process_image_traits import extract_traits_from_otherImages  # 处理特征数据
from model.modelVL import get_other_image_traits, get_image_traits  # 得到图像的特征

"""
1.得到多图像的特征，并对数据进行处理得到主要部分
"""

def getTraitsFrom_Images(image_paths):
    """
    1. 得到多图像的特征，并对数据进行处理得到主要部分
    
    Args:
        image_paths: 图像URL列表
        
    Returns:
        处理后的图像特征
    """
    traits = get_other_image_traits(image_paths)  # 读取的是url
    processed_traits = extract_traits_from_otherImages(traits)
    return processed_traits


def getTraitsFrom_Image(image_path):
    """
    2. 返回主图像的特征
    
    Args:
        image_path: 图像路径
        
    Returns:
        处理后的图像特征
    """

    traits = get_image_traits(image_path)  # 读取本地上传的图片
    processed_main_traits = extract_traits_from_otherImages(traits)
    return processed_main_traits



# 测试代码
if __name__ == "__main__":
    image_paths = [
        'https://aos-comment.amap.com/B00155LGHO/headerImg/content_media_external_images_media_66314_1659242286949_d6a6a0e8.jpg',
        'https://aos-comment.amap.com/B00155LGHO/headerImg/0b7ebe391f0b82eff50fa6373861f7b3_2048_2048_80.jpg'
    ]
    image_path = "D://Answer//python-learning//Fuction//game//image//sues01.png"
    
    # 测试多图像特征提取
    other_images_traits = getTraitsFrom_Images(image_paths)
    print("\ncompare:", other_images_traits)
    
    # 测试单图像特征提取
    # main_image_traits = getTraitsFrom_Image(image_path)
    # print("\nmain:", main_image_traits)
