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
from utils.parse_json import JsonParser
from skimage.metrics import hausdorff_distance
import json


def post_process_model_output(output_one_image):
    softmax_tensor = F.softmax(output_one_image, dim=0)
    # Step 1: Reduce 21-channel to single-channel by argmax
    predict_mask = torch.argmax(softmax_tensor, dim=0)  # shape: (256, 480)
    return predict_mask.numpy()

def save_result_image(predict_mask, work_dir, input_image, original_size=(1200, 1920)):
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
        rgb_image[predict_mask == class_idx] = color

    # Step 3: Resize to original size with bottom alignment
    result_image = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
    scale = original_size[1] // 480
    new_height = 256 * scale
    resized_rgb = cv2.resize(rgb_image, (original_size[1], new_height), interpolation=cv2.INTER_NEAREST)
    result_image[-new_height:, :] = resized_rgb  # bottom alignment

    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d-%h-%m")
    output_image_name = "output_" + date_str + ".png"
    output_image_path = os.path.join(work_dir, output_image_name)
    # result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, result_image)

    input_image = np.array(input_image)
    if result_image.shape != input_image.shape:
        print("Warn, result image shape {} != input image shape {}".format(result_image.shape, input_image.shape))
        return
    blended = cv2.addWeighted(result_image, 0.3, input_image, 0.7, 0)
    mix_image_name = "mix_" + date_str + ".png"
    mix_image_path = os.path.join(work_dir, mix_image_name)
    # blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mix_image_path, blended)


def calculate_metrics(pred_mask, gt_mask, num_classes):
    """
    计算分割任务的多类评估指标
    :param pred_mask: 模型预测mask (H,W) 值为0~num_classes-1
    :param gt_mask: 真实标注mask (H,W) 值为0~num_classes-1
    :param num_classes: 类别数(包含背景)
    :return: 字典包含各类指标
    """
    metrics = {}

    class_dictionary = {
        0: "void",
        1: "dirt",
        2: "grass",
        3: "tree",
        4: "pole",
        5: "water",
        6: "sky",
        7: "vehicle",
        8: "object",
        9: "asphalt",
        10: "building",
        11: "log",
        12: "person",
        13: "fence",
        14: "bush",
        15: "concrete",
        16: "barrier",
        17: "puddle",
        18: "mud",
        19: "rubble",
        20: "undefined"
    }

    # 逐类别计算指标
    for cls in range(num_classes):
        pred = (pred_mask == cls).astype(np.uint8)
        gt = (gt_mask == cls).astype(np.uint8)

        # 计算混淆矩阵
        tp = np.sum(pred * gt)
        fp = np.sum(pred * (1 - gt))
        fn = np.sum((1 - pred) * gt)
        tn = np.sum((1 - pred) * (1 - gt))

        if tn == pred_mask.shape[0] * pred_mask.shape[1]:
            print("Class", class_dictionary[cls], "has no pixel, skip")
            continue

        # 基础指标
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

        # 分割专用指标
        iou = tp / (tp + fp + fn + 1e-10)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

        # 边界指标(仅对前景类计算)
        if cls > 0:
            contours_pred = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours_gt = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            hd = hausdorff_distance(pred, gt) if (contours_pred and contours_gt) else 0

        metrics[class_dictionary[cls]] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'accuracy': round(accuracy, 4),
            'iou': round(iou, 4),
            'dice': round(dice, 4),
            'hausdorff_distance': round(hd, 4) if cls > 0 else None
        }

    # 计算平均指标(排除背景类)
    foreground_metrics = [v for k, v in metrics.items() if k != 'sky' or k != 'void']
    for metric in ['iou', 'dice']:
        avg_val = np.mean([m[metric] for m in foreground_metrics])
        metrics[f'mean_{metric}'] = round(avg_val, 4)

    return metrics

if __name__=='__main__':
    work_dir = sys.argv[1]
    json_parse = JsonParser(work_dir)
    image_name = sys.argv[2]
    if os.path.exists(image_name) and image_name.lower().endswith((".jpg", ".png")):
        input_image = Image.open(image_name).convert("RGB")
    else:
        print("Image path wrong.")
        exit(-1)

    image_transform = transforms.ToTensor()
    processed_image = image_transform(preprocess(input_image)).unsqueeze(0)
    model = UNet(in_channel=json_parse.input_channel,
                 contract_channels=json_parse.internal_channel,
                 bottle_channel=json_parse.bottle_channel,
                 out_channel=json_parse.class_num)
    model_path = os.path.join(work_dir, json_parse.model_name)
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
    output_host = output.cpu()

    label_image = cv2.imread("/home/yyu/Desktop/20250418_UNet/Rellis_3D_pylon_camera_node_label_color_480/train/frame001804-1581623970_750.png",
                         cv2.IMREAD_GRAYSCALE)
    # transforms = transforms.PILToTensor()
    # output_host = transforms(output_host)
    # output_host = F.one_hot(output_host.squeeze(0).long(), num_classes=json_parse.class_num).float()
    # output_host = output_host.permute(2, 0, 1)

    if output_host.ndim == 4:
        output_host = output_host[0]
    predict_image = post_process_model_output(output_host)

    indication = calculate_metrics(predict_image, label_image, json_parse.class_num)
    yaml_path = os.path.join(work_dir, "indication.json")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        json.dump(indication, f, indent=4, ensure_ascii=False)

    save_result_image(predict_image, work_dir, input_image)





