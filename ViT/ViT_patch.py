import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

# 假设一个 32x32 的 RGB 图像（生成一个随机图像）
img = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32

# patch 的大小
patch_height, patch_width = 8, 8

# 计算每个 patch 的维度
patch_dim = 3 * patch_height * patch_width  # 3（通道数）* 8 * 8 = 192

# 使用 einops 的 rearrange 来切分图像为 patches
patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)

# patches.shape: (batch_size, num_patches, patch_dim) => (1, 16, 192)
print("Shape of patches:", patches.shape)


# 现在可视化这些 patches
def visualize_patches(image, patch_height, patch_width):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # 创建一个 4x4 的网格来显示 16 个 patch
    axes = axes.flatten()

    # 将图像切分为 patches
    patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)

    # 显示每个 patch
    for i in range(patches.shape[1]):
        # 获取第 i 个 patch，形状为 (C, H, W)，然后 permute 为 (H, W, C)
        patch = patches[0, i].reshape(3, patch_height, patch_width).permute(1, 2, 0).detach().numpy()
        axes[i].imshow(patch)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 可视化切分后的 8x8 patches
visualize_patches(img, patch_height, patch_width)
