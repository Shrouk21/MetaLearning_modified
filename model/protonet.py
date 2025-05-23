import torch
import torch.nn as nn  # Import nn for layers
import torch.nn.init as init  # Import init for weight initialization
import math

# Build conv layer for feature extraction
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if retain_activation:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


# Build the embedding network (mapping feature space to fixed vector)
class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(x_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation)
        )

        self.__init__weights()  # Explicitly initialize the weights

    def __init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu') 
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1) 
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)