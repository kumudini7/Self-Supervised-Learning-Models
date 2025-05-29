import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import LabeledImageDataset  # Ensure this path is valid
from models.simclr import SimCLR               # Ensure this path is valid
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset and Dataloader
train_data = LabeledImageDataset("data/train.X1", "data/Labels.json", transform)
val_data = LabeledImageDataset("data/val.X", "data/Labels.json", transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Model setup
backbone = SimCLR().encoder
backbone.eval()
backbone.to(device)
for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(2048, 100).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
print(f"[INFO] Starting training on device: {device}")
for epoch in range(2):
    print(f"[INFO] Epoch {epoch+1} started")

    classifier.train()
    for batch_idx, (img, label) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"[DEBUG] First training batch: img shape {img.shape}, label type: {type(label)}")

        img = img.to(device)
        if isinstance(label, tuple):
            label = label[0]
        label = label.to(device)

        with torch.no_grad():
            feat = backbone(img).squeeze()
        out = classifier(feat)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[INFO] Epoch {epoch+1} training complete, starting evaluation...")


    # Evaluation loop
    classifier.eval()
    preds, targets = [], []
    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(device)
            if isinstance(label, tuple):
                label = label[0]
            label = label.to(device)
            feat = backbone(img).squeeze()
            out = classifier(feat)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(label.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    print(f"Epoch {epoch+1}: Accuracy={acc:.4f}, F1={f1:.4f}")
