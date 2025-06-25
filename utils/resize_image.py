import os
from PIL import Image
import numpy as np
from utils.preprocess import preprocess

def process_images(input_folder, output_folder, target="image"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):
            if target=="image":
                image = Image.open(os.path.join(input_folder, filename)).convert("RGB")
                new_image = preprocess(image, "image")
            else:
                image = Image.open(os.path.join(input_folder, filename)).convert("L")
                new_image = preprocess(image, "label")
            new_image.save(os.path.join(output_folder, filename))

if __name__=='__main__':
    # 设置输入输出文件夹
    image_input_folder = "../Rellis_3D_pylon_camera_node"
    image_output_folder = "../Rellis_3D_pylon_camera_node_480"

    process_images(image_input_folder, image_output_folder)
    print("处理完成，结果保存在 output_images 文件夹中。")

    label_input_folder = "../Rellis_3D_pylon_camera_node_label_color"
    label_output_folder = "../Rellis_3D_pylon_camera_node_label_color_480"

    process_images(label_input_folder, label_output_folder, "L")
    print("处理完成，结果保存在 output_images 文件夹中。")