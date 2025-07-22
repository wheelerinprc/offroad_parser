import os.path

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import random
from torchvision import transforms
from torchvision.datasets import VOCSegmentation  # 示例数据集（含图像和标签）

# --------------------- 配置部分 ---------------------
# 1. 定义标签到颜色的映射字典（示例：VOC数据集类别颜色）
label_colormap = {
        0: [0, 0, 0],
        1: [20, 64, 108],
        2: [0, 102, 0],
        3: [0, 255, 0],
        4: [153, 153, 0],
        5: [255, 128, 0],
        6: [255, 0, 0],
        7: [0, 255, 255],
        8: [127, 0, 255],
        9: [64, 64, 64],
        10: [0, 0, 255],
        11: [0, 0, 102],
        12: [255, 153, 204],
        13: [204, 0, 102],
        14: [204, 153, 255],
        15: [170, 170, 170],
        16: [255, 121, 41],
        17: [239, 255, 134],
        18: [34, 66, 99],
        19: [138, 22, 110],
        20: [255, 255, 255]
    }


# --------------------- 核心函数 ---------------------
def label_to_rgb(label_tensor, colormap):
    """
    将单通道标签Tensor转换为3通道RGB图像
    Args:
        label_tensor: [H, W] 单通道标签（值为类别索引）
        colormap:     字典 {类别索引: [R,G,B]}
    Returns:
        rgb_label:    [H, W, 3] 的RGB图像
    """
    label_np = label_tensor.squeeze().numpy().astype(np.uint8)  # 去掉batch维，转NumPy
    rgb_label = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)

    for class_idx, color in colormap.items():
        rgb_label[label_np == class_idx] = color  # 根据字典填充颜色

    return rgb_label


def visualize_sample(image_tensor, label_tensor, colormap):
    """
    合并图像和标签：上半部分为原图，下半部分为伪彩色标签
    Args:
        image_tensor: [C, H, W] 的RGB图像Tensor
        label_tensor: [1, H, W] 的单通道标签Tensor
    Returns:
        combined:     [2H, W, 3] 的合并图像
    """
    # 处理原图：Tensor -> NumPy -> [H,W,3]
    image_np = image_tensor.permute(1, 2, 0).numpy()  # [H,W,3]
    image_np = (image_np * 255).astype(np.uint8)  # 反归一化（若需要）

    # 处理标签：单通道 -> 3通道伪彩色
    label_rgb = label_to_rgb(label_tensor, colormap)  # [H,W,3]

    # 合并图像
    h, w = image_np.shape[:2]
    combined = np.vstack([image_np, label_rgb])  # 垂直拼接

    return combined

def check_dataset(dataset, workdir):
    # --------------------- 可视化循环 ---------------------

    idx = 0
    count = 1
    while True:
        # 随机选择样本
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]  # image:[3,H,W], label:[1,H,W]

        # 可视化合并
        combined = visualize_sample(image, label, label_colormap)
        image_name = os.path.join(workdir, str(idx) + ".png")
        cv2.imwrite(image_name, combined)

        # 键盘控制
        count += 1
        if count == 10:
            break