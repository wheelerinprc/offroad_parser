import dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import unet
from torch.utils.tensorboard import SummaryWriter
import train

if __name__=='__main__':
    if torch.cuda.is_available():
        print("Use cuda")

    root_img = "./Rellis_3D_pylon_camera_node"
    root_label = "./Rellis_3D_pylon_camera_node_label_color"
    data_tf = transforms.Compose([transforms.ToTensor()])
    train_data = dataset.MyDataset(root_img, root_label, transform=data_tf)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = unet.UNet(in_channel=3, contract_channels=[64,128,256,512], bottle_channel=1024,  out_channel=3)
    model = unet.UNet(in_channel=3, contract_channels=[8, 16, 32, 64], bottle_channel=128, out_channel=3)
    model.to(device)
    logger = SummaryWriter("./log")

    train.train_model(model, logger, train_loader, device)

    logger.close()

