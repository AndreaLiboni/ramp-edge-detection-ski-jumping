import torch
import numpy as np 
import torch.nn as nn

from model.backbone.fpn import FPN101, FPN50, FPN18, ResNext50_FPN
from model.backbone.mobilenet import MobileNet_FPN
from model.backbone.vgg_fpn import VGG_FPN
from model.backbone.res2net import res2net50_FPN

from model.dht import DHT_Layer

class Net(nn.Module):
    def __init__(self, backbone, dh_dimention, num_conv_layer, num_pool_layer, num_fc_layer=None):
        super(Net, self).__init__()
        numAngle, numRho = dh_dimention if type(dh_dimention) != int else (dh_dimention, dh_dimention)
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

        self.numAngle = numAngle
        self.numRho = numRho

        #conv_layers
        layers = []
        in_channels = 512
        out_channels = in_channels // 2 if num_conv_layer > 1 else 4
        for i in range(num_conv_layer):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels if i < num_conv_layer - 1 else 4,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            layers.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
            out_channels = max(out_channels // 2, 4)
        
        self.conv_layers = nn.Sequential(*layers)

        self.pooling_layers = None
        self.fc_layers = None
        if num_fc_layer is None:
            # pooling_layers
            layers = []
            start_dim = self.numAngle
            for i in range(num_pool_layer):
                start_dim = start_dim // 2
                layers.append(nn.AdaptiveAvgPool2d(
                    output_size=start_dim if i < num_pool_layer - 1 else 1
                ))
                layers.append(nn.LeakyReLU(inplace=True))
            
            self.pooling_layers = nn.Sequential(*layers)
        else:
            # fc_layers
            layers = [
                nn.AdaptiveAvgPool2d(25),
                nn.Flatten()
            ]
            in_channels = 25 * 25 * 4
            out_channels = 512 if num_fc_layer > 1 else 4
            for i in range(num_fc_layer):
                out_channels = out_channels if i < num_fc_layer - 1 else 4
                layers.append(nn.Linear(
                    in_features=in_channels,
                    out_features=out_channels
                ))
                layers.append(nn.LeakyReLU(inplace=True))
                if out_channels != 4:
                    layers.append(nn.Dropout(p=0.5))
                in_channels = out_channels
                out_channels = max(out_channels // 2, 4)
            
            self.fc_layers = nn.Sequential(*layers)


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
        if self.pooling_layers is not None:
            out = self.pooling_layers(out)
            out = out.view(out.size(0), -1)
        else:
            out = self.fc_layers(out)
        return out # [batch_size, 4] x1, y1, x2, y2
