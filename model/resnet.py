import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dropblock import DropBlock



class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        return nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
    


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.drop_size = block_size
        self.DropBlock = DropBlock(block_size=self.drop_size)
        self.num_batches_tracked = 0
        self.downsample = downsample
        self.stride = stride


        self.conv_bn_relu = nn.Sequential(
            convblock(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            convblock(planes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            convblock(planes, planes),
            nn.BatchNorm2d(planes),
            
        )
        self.maxpool = nn.MaxPool2d(stride)



    def forward(self, x):
        self.num_batches_tracked += 1
        residual = self.downsample(x) if self.downsample else x
        out = self.conv_bn_relu(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size(2)
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        super().__init__()
        self.inplanes = 3
        self.keep_avg_pool = avg_pool
        self.drop_rate = drop_rate
        self.keep_prob = keep_prob

        self.layer1 = self._make_layer(block, 64, 2, drop_rate)
        self.layer2 = self._make_layer(block, 160, 2, drop_rate)
        self.layer3 = self._make_layer(block, 320, 2, drop_rate, True, dropblock_size)
        self.layer4 = self._make_layer(block, 640, 2, drop_rate, True, dropblock_size)

        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.dropout = nn.Dropout(p=1 - keep_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride, drop_rate, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)
        self.inplanes = planes * block.expansion
        return nn.Sequential(layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    return ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)

        
