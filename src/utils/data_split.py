import os
import random
import shutil

def get_file_without_extension(file_name):
    return file_name.split("/")[-1].split(".")[0]

def compare(path_str1, path_str2):
    name1 = get_file_without_extension(path_str1)
    name2 = get_file_without_extension(path_str2)
    if name1 == name2:
        return 0
    elif name1 > name2:
        return 1
    else:
        return -1

def split_images(source_folder, train_folder, val_folder, train_ratio=0.9):
    """
    将源文件夹中的图片按比例随机分配到训练集和验证集文件夹中

    参数:
        source_folder: 包含原始图片的文件夹路径
        train_folder: 训练集文件夹路径
        val_folder: 验证集文件夹路径
        train_ratio: 训练集所占比例(默认0.9即9:1)
    """
    # 创建目标文件夹(如果不存在)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(source_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算分割点
    split_point = int(len(image_files) * train_ratio)

    # 分割文件
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]

    # 复制文件到训练集文件夹
    for file in train_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(train_folder, file)
        shutil.copy2(src, dst)

    # 复制文件到验证集文件夹
    for file in val_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(val_folder, file)
        shutil.copy2(src, dst)

    print(f"完成分割! 总图片数: {len(image_files)}")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")

def find_target_image(label_folder, image_folder):
    folder_name = label_folder.split("/")[-1]
    image_folder_new = str(os.path.join(image_folder,folder_name))
    os.makedirs(image_folder_new, exist_ok=True)
    label_files = [f for f in os.listdir(label_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    label_files = set([get_file_without_extension(f) for f in label_files])
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for file in image_files:
        file_no_extension = get_file_without_extension(file)
        if file_no_extension in label_files:
            src = os.path.join(image_folder, file)
            dst = os.path.join(image_folder_new, file)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    # 设置路径(请根据实际情况修改)
    label_dir = "../../Rellis_3D_pylon_camera_node_label_color_480"  # 原始图片文件夹
    image_dir = "../../Rellis_3D_pylon_camera_node_480"  # 原始图片文件夹
    train_dir = "../../Rellis_3D_pylon_camera_node_label_color_480/train"  # 训练集文件夹
    val_dir = "../../Rellis_3D_pylon_camera_node_label_color_480/val"  # 验证集文件夹

    # 调用函数进行分割
    split_images(label_dir, train_dir, val_dir)
    find_target_image(train_dir, image_dir)
    find_target_image(val_dir, image_dir)