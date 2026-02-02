import torch.nn as nn
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Type

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #   residual block的兩個conv layer
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        self.blockprelu1 = nn.PReLU()
        if stride != 1 or inchannel != outchannel:
            #shortcut，為了與Conv layer 後的結果相同
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        #經過2個conv layer後的結果與x相加
        out = out + self.shortcut(x)
        # out = F.relu(out)
        out = self.blockprelu1(out)
        
        
        return out 
    
class ResNet(nn.Module):
    def __init__(self, ResBlock=ResBlock, num_classes=20):
        super(ResNet, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Conv2d(1,32 , kernel_size=7, stride=2, padding=3, bias=False)   #original f36t7
        # self.conv1 = nn.Conv2d(1,32 , kernel_size=5, stride=2, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(1,32 , kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU() 
        self.layer1 = self.make_layer(ResBlock, 32, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 64, 2, stride=1)
        # self.layer3 = self.make_layer(ResBlock, 64, 2, stride=1)        
        # self.layer4 = self.make_layer(ResBlock, 64, 2, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1) #original
        # self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1,padding=3)   #try
        # self.fc1 = nn.Linear(12032,512 )    #f108t7
        # self.fc1 = nn.Linear(16128,512)   #kernel3 f36t28
        # self.fc1 = nn.Linear(12032,512)   #kernel7 f94t7  
        # self.fc1 = nn.Linear(11648,2048)   #kernel7 f26t28        
        self.fc1 = nn.Linear(3456,512 )    #original
        # self.fc1 = nn.Linear(3328,512 )    #f26t7
        # self.fc2 = nn.Linear(1024,512 )
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(64, 20)

    #重複產生同個residual block    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # print("Input shape:", x.shape)
        out = self.conv1(x)        
        # out=F.relu(out)
        out=self.prelu1(out)
        # print("Input shape:", out.shape)
        out = self.layer1(out)
        # print("Input shape:", out.shape)
        out = self.layer2(out)
        # print("Input shape:", out.shape)
        # out = self.layer3(out)
        # # print("Input shape:", out.shape)
        # out = self.layer4(out)
        # # print("Input shape:", out.shape)
        out = self.avgpool(out)
        # print("Input shape:", out.shape)
        out = torch.flatten(out, 1)
        # print("Input shape:", out.shape)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = F.normalize(out,p=2,dim=1)        #L2 normalize        
        # out = self.fc3(out)
        # out = F.softmax(out, dim = 1)
        # print("Input shape:", out.shape)
        return out
    
    # def forward(self, anchor, positive, negative):
    #     # 分别计算anchor、positive和negative样本的嵌入向量
    #     out_anchor = self.forward_one(anchor)
    #     out_positive = self.forward_one(positive)
    #     out_negative = self.forward_one(negative)
    #     return out_anchor, out_positive, out_negative

"""
2222222222222222222222222222222222222222222222
"""
# if __name__ == '__main__':
#     tensor = torch.rand([1, 3, 224, 224])
#     model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=1000)
#     print(model)
    
#     # Total parameters and trainable parameters.
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.")
#     output = model(tensor)

# class ResNet(nn.Module):
#     def __init__(self,img_channels = 1, num_layers = 18,block=BasicBlock,num_classes = 20) -> None:
#         super(ResNet, self).__init__()
#         if num_layers == 18:
#             # The following `layers` list defines the number of `BasicBlock`
#             # to use to build the network and how many basic blocks to stack
#             # together.
#             layers = [2, 2, 2, 2]
#             self.expansion = 1
#         self.in_channels = 64
#         # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
#         # three layers. Here, kernel size is 7.
#         self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
#         # self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512*self.expansion, num_classes)
        
#     def _make_layer(self,block: Type[BasicBlock],out_channels: int, blocks: int,stride: int = 1) -> nn.Sequential:
#         downsample = None
#         if stride != 1:
#             """
#             This should pass from `layer2` to `layer4` or 
#             when building ResNets50 and above. Section 3.3 of the paper
#             Deep Residual Learning for Image Recognition
#             (https://arxiv.org/pdf/1512.03385v1.pdf).
#             """
#             downsample = nn.Sequential(
    #             nn.Conv2d(self.in_channels,out_channels*self.expansion,kernel_size=1,stride=stride,bias=False),
    #             nn.BatchNorm2d(out_channels * self.expansion),
    #         )
    #     layers = [] # for storing the layers
    #     layers.append(
    #         block(
    #             self.in_channels, out_channels, stride, self.expansion, downsample
    #         )
    #     )
    #     self.in_channels = out_channels * self.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(
    #             self.in_channels,
    #             out_channels,
    #             expansion=self.expansion
    #         ))
    #     return nn.Sequential(*layers)
    # def forward(self, x: Tensor) -> Tensor:
    #     x = self.conv1(x)
    #     # x = self.bn1(x)
    #     x = self.relu(x)
    #     # x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     # The spatial dimension of the final layer's feature
    #     # map should be (7, 7) for all ResNets.
    #     # print('Dimensions of the last convolutional feature map: ', x.shape)
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #     x = torch.softmax(x, dim = 1)
    #     return x