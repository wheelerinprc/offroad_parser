import torch.nn as nn
import torch
def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(kernel_size=(3,3), in_channels=out_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ExpansionBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=in_channels // 2,
                                     kernel_size=(3, 3),
                                     stride=2,
                                     padding=1,
                                     output_padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, e, d):
        d = self.up(d)
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - d.size()[3]
        e = e[:, :, diffY // 2: e.size()[2] - diffY // 2, diffX // 2: e.size()[3] - diffX // 2]
        cat = torch.cat([e, d], dim = 1)
        out = self.block(cat)
        return out

def final_block(in_channel, out_channel):
    block = nn.Conv2d(kernel_size=(1, 1), in_channels=in_channel, out_channels=out_channel)
    return block


class UNet(nn.Module):
    def __init__(self, in_channel, contract_channels, bottle_channel, out_channel):
        super(UNet, self).__init__()
        self.conv_encode_list = nn.ModuleList()
        self.conv_decode_list = nn.ModuleList()
        expansion_channels = contract_channels[::-1]
        contract_channels.insert(0, in_channel)
        expansion_channels.insert(0, bottle_channel)
        for i in range(len(contract_channels)-1):
            self.conv_encode_list.append(contracting_block(in_channels=contract_channels[i], out_channels=contract_channels[i+1]))
            self.conv_decode_list.append(ExpansionBlock(expansion_channels[i], expansion_channels[i+1], expansion_channels[i+1]))
        self.conv_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=contract_channels[-1], out_channels=bottle_channel, padding=1),
            nn.BatchNorm2d(bottle_channel),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=bottle_channel, out_channels=bottle_channel, padding=1),
            nn.BatchNorm2d(bottle_channel),
            nn.ReLU()
        )
        self.final_layer = final_block(expansion_channels[-1], out_channel)

    def forward(self, x):
        encode_inter_result = []
        for encode_block in self.conv_encode_list:
            # print(f"memory allocated - 1: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            x = encode_block(x)
            encode_inter_result.append(x)
            x = self.conv_pool(x)

        x = self.bottleneck(x)
        assert(len(encode_inter_result)==len(self.conv_decode_list))
        for decode_block in self.conv_decode_list:
            # print(f"memory allocated - 1: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            x = decode_block(encode_inter_result[-1], x)
            encode_inter_result.pop()

        x = self.final_layer(x)
        return x
    