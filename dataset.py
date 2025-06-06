from PIL import Image
from torchvision import transforms
import torch
import os
from enum import Enum


class Color(Enum):
    SKY = 1
    GREEN = 2
    BLUE = 3

def get_specific_files(folder_path, extensions=('.jpg', '.png', '.jpeg')):
    """获取特定扩展名的文件"""
    if not os.path.isdir(folder_path):
        raise ValueError("提供的路径不是一个有效的文件夹")
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                file_list.append(os.path.join(root, file))
                # file_list.append(file)
    return sorted(file_list)

def compare(path_str1, path_str2):
    name1 = path_str1.split("/")[-1].split(".")[0]
    name2 = path_str2.split("/")[-1].split(".")[0]
    if name1 == name2:
        return 0
    elif name1 > name2:
        return 1
    else:
        return -1

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory, label_directory, transform=None):
        super(MyDataset, self).__init__()
        image_names_raw = get_specific_files(image_directory)
        label_names_raw = get_specific_files(label_directory)

        self.image_names = []
        self.label_names = []
        image_num = len(image_names_raw)
        label_num = len(label_names_raw)
        i = 0
        j = 0
        while i<image_num and j < label_num:
            if compare(image_names_raw[i], label_names_raw[j]) == 0:
                self.image_names.append(image_names_raw[i])
                self.label_names.append(label_names_raw[j])
                i+=1
                j+=1
            elif compare(image_names_raw[i], label_names_raw[j]) > 0:
                j+=1
            else:
                i +=1
        self.transform = transform

    def __getitem__(self, item):
        img_name = self.image_names[item]
        label_name = self.label_names[item]
        img = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.image_names)

if __name__=='__main__':
    root_img = "./Rellis_3D_pylon_camera_node"
    root_label = "./Rellis_3D_pylon_camera_node_label_color"
    data_tf = transforms.Compose([transforms.ToTensor()])
    train_data = MyDataset(root_img, root_label, transform=data_tf)

