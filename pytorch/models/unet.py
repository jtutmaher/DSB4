import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Build U-Net model
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.ReflectionPad2d(kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReflectionPad2d(kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        c = self.block(x)
        return c

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.conv = DownBlock(2*out_channels, out_channels)
    
    def forward(self, x1, x2):
        x = torch.cat([self.up(x1), x2], dim=1)
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.dblock1 = DownBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock2 = DownBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock3 = DownBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock4 = DownBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock5 = DownBlock(512, 1024)
        self.ublock5 = UpBlock(1024, 512)
        self.ublock4 = UpBlock(512, 256)
        self.ublock3 = UpBlock(256, 128)
        self.ublock2 = UpBlock(128, 64)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1, groups=1, stride=1)
        
    def forward(self, x):
        c1 = self.dblock1(x)
        p1 = self.pool1(c1)
        c2 = self.dblock2(p1)
        p2 = self.pool1(c2)
        c3 = self.dblock3(p2)
        p3 = self.pool1(c3)
        c4 = self.dblock4(p3)
        p4 = self.pool1(c4)
        c5 = self.dblock5(p4)
        u5 = self.ublock5(c5, c4)
        u4 = self.ublock4(u5, c3)
        u3 = self.ublock3(u4, c2)
        u2 = self.ublock2(u3, c1)

        return self.output(u2)