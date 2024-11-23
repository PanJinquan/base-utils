import cv2
import numpy as np
from pybaseutils import image_utils,file_utils

def wide_dynamic_range(image, alpha=0.5, beta=1.2):
    """
    实现简单的宽动态范围增强
    
    参数:
        image: 输入图像
        alpha: 局部对比度参数
        beta: 亮度增强参数
    """
    # 转换为浮点型
    img_float = image.astype(np.float32) / 255.0
    
    # 计算图像的亮度通道
    if len(image.shape) == 3:
        luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    else:
        luminance = img_float

    # 使用高斯模糊创建局部亮度图
    blur_radius = max(3, int(min(image.shape[:2]) * 0.02))
    if blur_radius % 2 == 0:
        blur_radius += 1
    local_mean = cv2.GaussianBlur(luminance, (blur_radius, blur_radius), 0)

    # 计算局部对比度增强
    detail_layer = luminance - local_mean
    enhanced_detail = detail_layer * alpha

    # 增强亮度
    enhanced_luminance = local_mean * beta + enhanced_detail
    
    # 确保值在合理范围内
    enhanced_luminance = np.clip(enhanced_luminance, 0, 1)

    # 如果是彩色图像，保持色彩不变，仅调整亮度
    if len(image.shape) == 3:
        # 计算增益
        gain = np.divide(enhanced_luminance, luminance, 
                        out=np.ones_like(luminance), where=luminance!=0)
        
        # 应用增益到每个通道
        enhanced_image = img_float * gain[:,:,np.newaxis]
        enhanced_image = np.clip(enhanced_image, 0, 1)
    else:
        enhanced_image = enhanced_luminance

    # 转换回8位整数格式
    result = (enhanced_image * 255).astype(np.uint8)
    
    return result

# 使用示例
if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/暗光数据"
    image_list = file_utils.get_files_lists(image_dir)
    for file in image_list:
        # 读取图像
        image = image_utils.read_image_ch(file)
        
        # 应用宽动态范围增强
        enhanced = wide_dynamic_range(image)
        
        # 显示结果
        image_utils.cv_show_image("Original", image,delay=10)
        image_utils.cv_show_image("Enhanced", enhanced)
