import os.path
import sys
from utils.parse_json import JsonParser
from fontTools.misc.cython import returns
import dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import unet
from torch.utils.tensorboard import SummaryWriter
import train
from utils.dataset_check import check_dataset

if __name__=='__main__':
    train_images = "./Rellis_3D_pylon_camera_node_480/train"
    train_labels = "./Rellis_3D_pylon_camera_node_label_color_480/train"
    val_images = "./Rellis_3D_pylon_camera_node_480/val"
    val_labels = "./Rellis_3D_pylon_camera_node_label_color_480/val"

    if len(sys.argv) < 2:
        print("Error - Working directory miss")
        exit(-1)
    working_dir = sys.argv[1]
    print("Working directory: ", working_dir)
    if not os.path.exists(working_dir):
        print("Error - Working directory does not exist")
        exit(-1)
    json_parse = JsonParser(working_dir)

    print("Dataset reading ...")
    train_data = dataset.MyDataset(train_images, train_labels)
    # check_dataset(train_data, working_dir)

    train_loader = DataLoader(train_data, batch_size=json_parse.train_batch, shuffle=True)
    val_data = dataset.MyDataset(val_images, val_labels)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    print("Dataset read done.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Use cuda")

    log_dir = os.path.join(working_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    model = unet.UNet(in_channel=json_parse.input_channel,
                      contract_channels=json_parse.internal_channel,
                      bottle_channel=json_parse.bottle_channel,
                      out_channel=json_parse.class_num)
    model.to(device)

    train.train_model(model, logger, train_loader, val_loader, device, working_dir, json_parse, num_epochs=100)
    logger.close()

