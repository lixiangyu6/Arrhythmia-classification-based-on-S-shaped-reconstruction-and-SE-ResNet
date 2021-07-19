import torch.nn.functional as F
import torch.nn as nn
import torch
from torchsummary import summary

class se_resnet_block_1d(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(se_resnet_block_1d, self).__init__()

        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.avg_pool=nn.AdaptiveAvgPool1d(1)

        #通道注意力权重
        self.calculateweight=nn.Sequential(
            nn.Linear(outchannel, outchannel // 2, bias=False),
            nn.Linear(outchannel // 2, outchannel, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        #计算通道关注度
        b, c, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        weight = self.calculateweight(y).view(b, c, 1)
        #关注度乘以原数据
        out = out*weight.expand_as(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class se_resnet_19_1d(nn.Module):
    def __init__(self, Block, num_classes=5):
        super(se_resnet_19_1d, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Block, 64,  2, stride=1)
        self.layer2 = self.make_layer(Block, 128, 2, stride=2)
        self.layer3 = self.make_layer(Block, 256, 2, stride=2)
        self.layer4 = self.make_layer(Block, 512, 2, stride=2)
        self.avg_pool1d = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool1d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class se_resnet_block_2d(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(se_resnet_block_2d, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        #通道注意力权重
        self.calculateweight=nn.Sequential(
            nn.Linear(outchannel, outchannel // 2, bias=False),
            nn.Linear(outchannel // 2, outchannel, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        #计算通道关注度
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        weight = self.calculateweight(y).view(b, c, 1, 1)
        #关注度乘以原数据
        out = out*weight.expand_as(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class se_resnet_19_2d(nn.Module):
    def __init__(self, Block, num_classes=5):
        super(se_resnet_19_2d, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Block, 64,  2, stride=1)
        self.layer2 = self.make_layer(Block, 128, 2, stride=2)
        self.layer3 = self.make_layer(Block, 256, 2, stride=2)
        self.layer4 = self.make_layer(Block, 512, 2, stride=2)
        self.avg_pool2d = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

def Get_se_resnet_19_2d(num_classes):
    return se_resnet_19_2d(se_resnet_block_2d,num_classes=num_classes)

def Get_se_resnet_19_1d(num_classes):
    return se_resnet_19_1d(se_resnet_block_1d,num_classes=num_classes)

if __name__=='__main__':
    model = Get_se_resnet_19_2d(num_classes=5).cuda()
    summary(model, input_size=(1,16,16))