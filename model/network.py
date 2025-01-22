import torch
import numpy as np 
import torch.nn as nn

from model.backbone.fpn import FPN101, FPN50, FPN18, ResNext50_FPN
from model.backbone.mobilenet import MobileNet_FPN
from model.backbone.vgg_fpn import VGG_FPN
from model.backbone.res2net import res2net50_FPN

from model.dht import DHT_Layer

class Net(nn.Module):
    def __init__(self, numAngle, numRho, backbone):
        super(Net, self).__init__()
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)
            output_stride = 32
        if backbone == 'resnet50':
            self.backbone = FPN50(pretrained=True, output_stride=16)
            output_stride = 16
        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)
            output_stride = 16
        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)
            output_stride = 16
        if backbone == 'vgg16':
            self.backbone = VGG_FPN()
            output_stride = 16
        if backbone == 'mobilenetv2':
            self.backbone = MobileNet_FPN()
            output_stride = 32
        if backbone == 'res2net50':
            self.backbone = res2net50_FPN()
            output_stride = 32
        
        if backbone == 'mobilenetv2':
            self.dht_detector1 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            
            self.last_conv = nn.Sequential(
                nn.Conv2d(128, 1, 1)
            )
        else:
            self.dht_detector1 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            
            # self.last_conv = nn.Sequential(
            #     nn.Conv2d(512, 1, 1)
            # )

        self.numAngle = numAngle
        self.numRho = numRho

        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Reduce to a smaller fixed size
            # [btach_size, 4, 1, 1]
            nn.LeakyReLU(inplace=True),
        )

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(64 * 10 * 10, 64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(64, 4),
        #     nn.LeakyReLU(inplace=True),
        # )

    def upsample_cat(self, p1, p2, p3, p4):
        p1 = nn.functional.interpolate(p1, size=(self.numAngle, self.numRho), mode='bilinear')
        p2 = nn.functional.interpolate(p2, size=(self.numAngle, self.numRho), mode='bilinear')
        p3 = nn.functional.interpolate(p3, size=(self.numAngle, self.numRho), mode='bilinear')
        p4 = nn.functional.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear')
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x):
        # x = [batch_size, channel, height, width]
        p1, p2, p3, p4 = self.backbone(x)
        # [batch_size, num_conv, height, width]
        # p1 100, 100 height, width
        # p2 50, 50
        # p3 25, 25
        # p4 25, 25
      
        p1 = self.dht_detector1(p1)
        p2 = self.dht_detector2(p2)
        p3 = self.dht_detector3(p3)
        p4 = self.dht_detector4(p4)

        out = self.upsample_cat(p1, p2, p3, p4)
        out = self.conv_layers(out)
        # out = self.fc_layers(out)
        return out.view(out.size(0), -1) # [batch_size, 4] x1, y1, x2, y2
