# -*- coding: utf-8 -*-
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import sys

# 尝试导入 cv2, 但即使开始时缺少也不报错
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("警告: 未找到 OpenCV (cv2) 库。图像转换功能受限。", file=sys.stderr)
    CV2_AVAILABLE = False

def convert_cv_qt(cv_img):
    """
    将图像（假定为 NumPy 数组）转换为 QImage。
    处理 RGB、BGR（如果 OpenCV 可用）、灰度图。
    假定来自摄像头线程的输入在线程内已转换为 RGB。
    """
    if cv_img is None:
        return None

    try:
        if not isinstance(cv_img, np.ndarray):
            print(f"错误: 输入不是 NumPy 数组, 类型为 {type(cv_img)}", file=sys.stderr)
            return None

        if len(cv_img.shape) == 3: # 彩色图像
            height, width, channel = cv_img.shape
            bytes_per_line = channel * width # 处理 3 或 4 通道

            if channel == 3:
                # 假定输入是 RGB (因为线程应该已从 BGR 转换)
                qt_format = QImage.Format_RGB888
                img_data = cv_img.data
                # 如果需要显式处理 BGR 并且 cv2 可用:
                # if CV2_AVAILABLE:
                #     bgr_image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                #     img_data = bgr_image.data
                # else: # 没有 cv2 无法转换 BGR
                #     return None
            elif channel == 4:
                 # 假定是 RGBA
                 qt_format = QImage.Format_RGBA8888
                 img_data = cv_img.data
                 # 如果需要并且 cv2 可用，添加 BGR<->RGB 转换
            else:
                 print(f"错误: 不支持的通道数: {channel}", file=sys.stderr)
                 return None

            qt_image = QImage(img_data, width, height, bytes_per_line, qt_format)

        elif len(cv_img.shape) == 2: # 灰度图像
            height, width = cv_img.shape
            bytes_per_line = width
            qt_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
             print(f"错误: 不支持的图像形状: {cv_img.shape}", file=sys.stderr)
             return None

        # 返回副本以避免多线程中的共享内存问题
        return qt_image.copy()

    except Exception as e:
        print(f"错误：转换图像时出错: {e}", file=sys.stderr)
        print(f"图像形状: {getattr(cv_img, 'shape', 'N/A')}, 类型: {getattr(cv_img, 'dtype', 'N/A')}", file=sys.stderr)
        return None