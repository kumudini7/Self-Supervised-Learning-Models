import sys
import os
sys.path.append(os.path.abspath("/content/ssl_dataset/ssl_dataset"))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.dataset import UnlabeledImageDataset
from models.simclr import SimCLR
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    batch_size = z1.size(0)
    mask = torch.eye(batch_size * 2, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    return F.cross_entropy(sim, labels)

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB
])

if __name__ == "__main__":
    paths = [ '/content/ssl_dataset/ssl_dataset/train.X1']
        #'/content/data/ssl_dataset/ssl_dataset/train.X2',
        #'/content/data/ssl_dataset/ssl_dataset/train.X3',
        #'/content/data/ssl_dataset/ssl_dataset/train.X4']
    for path in paths:
        print(f"{path} exists? {os.path.exists(path)}")

    dataset = UnlabeledImageDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR().to(device)
    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Quick sanity check on single batch
    print("\nRunning quick single-batch test...")
    try:
        img_sample = next(iter(loader)).to(device)
        model.eval()
        with torch.no_grad():
            _, out_proj1 = model(img_sample)
            _, out_proj2 = model(img_sample.clone())
        print(f"Model outputs shapes: {out_proj1.shape}, {out_proj2.shape}")
        loss = nt_xent_loss(out_proj1, out_proj2)
        print(f"NT-Xent loss on single batch: {loss.item()}\n")
    except Exception as e:
        print(f"Exception during single-batch test: {e}")
        exit(1)

    for epoch in range(5):
        model.train()
        total_loss = 0
        print(f"Starting epoch {epoch+1}")
        for i, img in enumerate(tqdm(loader)):
            try:
                img = img.to(device)
                _, z1 = model(img)
                _, z2 = model(img.clone())
                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # Print every 50 batches
                if (i + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {i+1} loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Exception in epoch {epoch+1} batch {i+1}: {e}")
                break
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/simclr_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
