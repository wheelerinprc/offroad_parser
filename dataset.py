from PIL import Image
from torchvision import transforms
import torch
import os
from enum import Enum
from utils.timecost import get_time
import numpy as np
from torch.utils.data import DataLoader

def channel_expansion(image):
    image_array = np.array(image)
    # Get image dimensions
    height, width = image_array.shape
    # Initialize a 21-channel array with zeros
    one_hot = np.zeros((height, width, 21), dtype=np.uint8)

    # Set the corresponding channel to 1 for each pixel
    for i in range(21):
        one_hot[:, :, i] = (image_array == i).astype(np.uint8)

    return one_hot


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory, label_directory):
        super(MyDataset, self).__init__()
        image_names = [f for f in os.listdir(image_directory)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        label_names = [f for f in os.listdir(label_directory)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        image_names = sorted(image_names)
        label_names = sorted(label_names)
        images_PIL = [Image.open(os.path.join(image_directory, img_name)).convert('RGB') for img_name in image_names]
        labels_PIL = [Image.open(os.path.join(label_directory, label_name)).convert('L') for label_name in label_names]

        image_transform = transforms.ToTensor()
        label_transform = transforms.PILToTensor()
        self.images = [image_transform(image) for image in images_PIL]
        self.labels = [label_transform(label) for label in labels_PIL]
        # max_item = [label.max().item() for label in self.labels]
        # min_item = [label.min().item() for label in self.labels]
        # result = [(index, value) for index, value in enumerate(max_item) if value > 20]

    # @get_time
    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return len(self.images)

if __name__=='__main__':
    root_img = "./Rellis_3D_pylon_camera_node_480/train"
    root_label = "./Rellis_3D_pylon_camera_node_label_color_480/train"
    data_tf = transforms.Compose([
        # transforms.Lambda(lambda img: transforms.Resize(
        #     (img.height // 2, img.width // 2))(img)),
        transforms.ToTensor()]
    )
    train_data = MyDataset(root_img, root_label)
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    image_test, label_test = MyDataset[1]
    print("Test")
