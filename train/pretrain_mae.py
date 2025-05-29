import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.dataset import UnlabeledImageDataset
from models.mae import MAE

# --- Masking function ---
def random_masking(x, mask_ratio=0.75, patch_size=16):
    B, C, H, W = x.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_keep = int((1 - mask_ratio) * num_patches)

    # Flatten patches
    x_patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x_patches = x_patches.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, N, p, p]
    x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [B, N, C, p, p]

    # Mask
    idx = torch.rand(x_patches.shape[1], device=x.device).argsort()[:num_keep]
    masked_patches = x_patches[:, idx]

    # Reconstruct back to image shape
    mask = torch.ones_like(x_patches)
    mask[:, idx] = 0
    masked = x.clone()
    masked_patches_flat = mask.permute(0, 2, 3, 4, 1).reshape(B, C * patch_size * patch_size, -1)
    masked = masked.view(B, C, -1)
    masked = masked * (1 - masked_patches_flat).view(B, C, H, W)

    return masked

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Dataset ---
dataset = UnlabeledImageDataset([
    'data/train.X1', 'data/train.X2', 'data/train.X3', 'data/train.X4'
], transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# --- Model ---
model = MAE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# --- Training ---
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images in tqdm(loader):
        images = images.cuda()
        masked_images = random_masking(images, mask_ratio=0.75)

        recon = model(masked_images)
        target = images.view(images.size(0), -1)  # Flatten for MSE
        loss = loss_fn(recon, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
