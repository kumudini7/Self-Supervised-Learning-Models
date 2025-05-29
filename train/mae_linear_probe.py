import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data.dataset import LabeledImageDataset
from models.mae import MAE

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Dataset ---
train_dataset = LabeledImageDataset('data/train.X1', 'data/Labels.json', transform=transform)
val_dataset = LabeledImageDataset('data/val.X', 'data/Labels.json', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# --- Load pretrained MAE encoder ---
mae = MAE()
checkpoint = torch.load('mae_pretrained.pth')
mae.load_state_dict(checkpoint)
encoder = mae.encoder  # Extract only the encoder
encoder.eval()
encoder.cuda()
for param in encoder.parameters():
    param.requires_grad = False

# --- Linear Classifier ---
classifier = nn.Linear(encoder.output_dim, 100).cuda()  # 100 classes in ImageNet-100
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(20):
    classifier.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            features = encoder(images)
        logits = classifier(features)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

# --- Evaluation ---
classifier.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.cuda()
        features = encoder(images)
        logits = classifier(features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
