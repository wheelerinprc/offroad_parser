import sys
import torch
from utils.preprocess import preprocess
from PIL import Image
import os
from unet import  UNet
import numpy as np
import cv2
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime


def post_process_model_output(output_one_image, original_size=(1200, 1920)):
    """
    Post-process the model output.

    Parameters:
    - output: numpy array of shape [1, 21, 256, 480]
    - original_size: tuple (height, width) of the original image

    Returns:
    - RGB image of shape (1200, 1920, 3) with bottom alignment
    """
    softmax_tensor = F.softmax(output_one_image, dim=0)
    # Step 1: Reduce 21-channel to single-channel by argmax
    class_map = torch.argmax(softmax_tensor, dim=0)  # shape: (256, 480)

    # Step 2: Map class indices to RGB using predefined dictionary
    index_to_rgb = {
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

    # Create RGB image
    rgb_image = np.zeros((256, 480, 3), dtype=np.uint8)
    for class_idx, color in index_to_rgb.items():
        rgb_image[class_map == class_idx] = color

    # Step 3: Resize to original size with bottom alignment
    resized_image = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
    scale = original_size[1] // 480
    new_height = 256 * scale
    resized_rgb = cv2.resize(rgb_image, (original_size[1], new_height), interpolation=cv2.INTER_NEAREST)
    resized_image[-new_height:, :] = resized_rgb  # bottom alignment

    return resized_image

def save_result_image(result_image, image_name, input_image):
    cv2.imwrite(image_name, result_image)
    input_image = np.array(input_image)
    if result_image.shape != input_image.shape:
        print("Warn, result image shape {} != input image shape {}".format(result_image.shape, input_image.shape))
        return
    blended = cv2.addWeighted(result_image, 0.3, input_image, 0.7, 0)
    cv2.imwrite("image_mix.png", blended)


if __name__=='__main__':
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d")

    if os.path.exists(image_path) and image_path.lower().endswith((".jpg", ".png")):
        input_image = Image.open(image_path).convert("RGB")
    else:
        print("Image path wrong.")
        exit(-1)
    image_transform = transforms.ToTensor()
    processed_image = image_transform(preprocess(input_image)).unsqueeze(0)
    model = UNet(in_channel=3, contract_channels=[16, 32, 64], bottle_channel=128, out_channel=21)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("Model path wrong.")
        exit(-1)

    model_time = os.path.getmtime(model_path)
    # 转换为可读的日期时间格式
    model_datetime = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d-%h-%m")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    processed_image = processed_image.to(device)
    with torch.no_grad():
        output = model(processed_image)
    # output_host = output.cpu()
    output_host = Image.open("/home/yyu/Desktop/20250418_UNet/Rellis_3D_pylon_camera_node_label_color_480/train/frame001804-1581623970_750.png").convert("L")
    transforms = transforms.PILToTensor()
    output_host = transforms(output_host)
    output_host = F.one_hot(output_host.squeeze(0).long(), num_classes=21).float()
    output_host = output_host.permute(2, 0, 1)

    if output_host.ndim == 4:
        for i in range(output_host.shape[0]):
            result_image = post_process_model_output(output_host[i])
            output_image_name = "output_" + str(model_datetime) + "_" + date_str + "_" +  str(i) + ".png"
            save_result_image(result_image, output_image_name, input_image)
    elif output_host.ndim == 3:
        result_image = post_process_model_output(output_host)
        save_result_image(result_image,"output.jpg", input_image)
    else:
        print("Output tensor dimension is wrong - {}".format(output_host.dim))



