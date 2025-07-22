import os
from PIL import Image
import numpy as np

# 映射关系
mapping = {
    (0, 0, 0): 0,
    (108, 64, 20): 1,
    (0, 102, 0): 2,
    (0, 255, 0): 3,
    (0, 153, 153): 4,
    (0, 128, 255): 5,
    (0, 0, 255): 6,
    (255, 255, 0): 7,
    (255, 0, 127): 8,
    (64, 64, 64): 9,
    (255, 0, 0): 10,
    (102, 0, 0): 11,
    (204, 153, 255): 12,
    (102, 0, 204): 13,
    (255, 153, 204): 14,
    (170, 170, 170): 15,
    (41, 121, 255): 16,
    (134, 255, 239): 17,
    (99, 66, 34): 18,
    (110, 22, 138): 19
}


def convert_image(image):
    img_array = np.array(image)
    new_img_array = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel = tuple(img_array[i, j])
            new_img_array[i, j] = mapping.get(pixel, 4)

    return Image.fromarray(new_img_array)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):
            image = Image.open(os.path.join(input_folder, filename)).convert("RGB")
            new_image = convert_image(image)
            new_image.save(os.path.join(output_folder, filename))

if __name__=='__main__':
    # 设置输入输出文件夹
    input_folder = "Rellis_3D_pylon_camera_node_label_color_3channel"
    output_folder = "Rellis_3D_pylon_camera_node_label_color"

    process_images(input_folder, output_folder)
    print("处理完成，结果保存在 output_images 文件夹中。")
