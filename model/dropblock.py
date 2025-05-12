#DropBlock drops contiguous regions of feature map for regularization


import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        x = x.cuda()
        if not self.training or gamma == 0:
            return x  # No regularization during evaluation

        B, C, H, W = x.shape
        mask_shape = (B, C, H, W)
        sampling_mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))

        # Apply max pooling to create block mask
        block_mask = F.max_pool2d(sampling_mask, self.block_size, stride=1, padding=self.block_size // 2)
        block_mask = block_mask.to(x.device)
        block_mask = 1 - block_mask  # To keep the unmasked region

        # Ensure block_mask matches x's shape
        if block_mask.shape != x.shape:
            block_mask = F.interpolate(block_mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

        norm_factor = block_mask.numel() / block_mask.sum()
        return x * block_mask * norm_factor
