import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.encoder = vit_b_16(pretrained=False)
        self.decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768)
        )
        self.reconstruction = nn.Linear(768, 3 * 16 * 16)  # Assume patch size 16

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.reconstruction(decoded)
        return output
