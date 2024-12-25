import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重现性
torch.manual_seed(42)
np.random.seed(42)

# 假设输入图像的尺寸是 32x32
image_size = 32
patch_size = 8  # 每个patch的尺寸为8x8

# 生成一个随机的32x32的图像（假设图像有3个通道，例如RGB）
image = torch.randn(3, image_size, image_size)

# 计算每个patch的数量
num_patches = (image_size // patch_size) ** 2  # (32 // 8) ** 2 = 16 patches


# 1. 切分图像成patches
def image_to_patches(image, patch_size):
    # 切分图像成patches并展平
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, num_patches, patch_size * patch_size)  # 展平每个patch
    patches = patches.permute(1, 0, 2)  # 转置以使patch维度为(num_patches, 3, patch_size*patch_size)
    return patches


patches = image_to_patches(image, patch_size)
print(f"Number of patches: {patches.shape[0]}")  # 16 patches
print(f"Shape of one patch: {patches.shape[1:]}")
print(f"Shape of all patches: {patches.shape}")


# 2. 进行掩蔽（Masking）
def mask_patches(patches, mask_ratio=0.5):
    # mask_ratio 控制要掩蔽掉的patch比例
    num_masked = int(mask_ratio * patches.shape[0])  # 掩蔽patch的数量
    mask_indices = np.random.choice(patches.shape[0], num_masked, replace=False)  # 随机选择要掩蔽的patch
    patches[mask_indices] = 0  # 将这些patch的值置为0表示被掩蔽
    return patches, mask_indices


masked_patches, masked_indices = mask_patches(patches.clone())  # 保持原始patch不变
print(f"Masked patch indices: {masked_indices}")


# 可视化掩蔽后的图像
def visualize_patches(patches, masked_indices, patch_size, image_size):
    # 将patch恢复为图像的形式（还原为32x32的图像）
    unmasked_patches = patches.clone()
    for idx in masked_indices:
        unmasked_patches[idx] = 0  # 将被掩蔽的patch设置为0

    # 重建图像（注意这里是简单的合并patches并没有做反卷积等处理）
    patch_image = torch.zeros(3, image_size, image_size)
    patch_idx = 0
    for i in range(0, image_size, patch_size):
        for j in range(0, image_size, patch_size):
            patch_image[:, i:i + patch_size, j:j + patch_size] = unmasked_patches[patch_idx].view(3, patch_size,
                                                                                                  patch_size)
            patch_idx += 1

    plt.imshow(patch_image.permute(1, 2, 0).detach().numpy())  # 将通道维度移到最后以便显示
    plt.title("Masked Image Visualization")
    plt.axis('off')
    plt.show()


# 可视化掩蔽后的图像
visualize_patches(masked_patches, masked_indices, patch_size, image_size)
