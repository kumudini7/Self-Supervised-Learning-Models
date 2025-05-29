import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLR, self).__init__()
        base_model = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return h, z
