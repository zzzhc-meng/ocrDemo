import torch
import torch.nn as nn
import torchvision.models as models

class DBNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DBNet, self).__init__()
        # 使用ResNet50作为骨干特征提取网络
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的fc层和avgpool
        self.conv1x1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.binarize = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)   # [B, 2048, H/32, W/32]
        features = self.conv1x1(features)  # [B, 256, H/32, W/32]
        prob_map = self.binarize(features) # [B, 1, H/32, W/32]
        return prob_map
