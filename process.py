import dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import unet
from torch.utils.tensorboard import SummaryWriter
import train

from torchvision import transforms

if __name__=='__main__':
    if torch.cuda.is_available():
        print("Use cuda")

    train_images = "./Rellis_3D_pylon_camera_node_480/train"
    train_labels = "./Rellis_3D_pylon_camera_node_label_color_480/train"

    val_images = "./Rellis_3D_pylon_camera_node_480/val"
    val_labels = "./Rellis_3D_pylon_camera_node_label_color_480/val"

    print("Dataset reading ...")
    train_data = dataset.MyDataset(train_images, train_labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = dataset.MyDataset(val_images, val_labels)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    print("Dataset read done.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = unet.UNet(in_channel=3, contract_channels=[64,128,256,512], bottle_channel=1024,  out_channel=3)
    model = unet.UNet(in_channel=3, contract_channels=[16, 32, 64], bottle_channel=128, out_channel=21)
    model.to(device)
    logger = SummaryWriter("./log")

    train.train_model(model, logger, train_loader, val_loader, device)

    logger.close()

