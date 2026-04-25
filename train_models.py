import torch, torchvision, os
from torchvision import datasets, transforms, models
import torch.nn as nn, torch.optim as optim
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# MUST match alphabetical folder names: mask(0), no_mask(1)
CLASS_NAMES = ["Mask", "No Mask"]

def train_one_epoch(model, loader, criterion, optimizer, phase):
    model.train() if phase == "train" else model.eval()
    running_loss, running_corrects = 0.0, 0
    
    with torch.set_grad_enabled(phase == "train"):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if phase == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc

def main():
    print(f"🚀 Training on {DEVICE}")
        # Data transforms (Updated to handle PNG transparency/palette warnings)
    data_transforms = {
        "train": transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),  # ✅ FIX: Convert all to RGB
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),  # ✅ FIX: Convert all to RGB
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {
        "train": datasets.ImageFolder(DATA_DIR/"train", data_transforms["train"]),
        "val": datasets.ImageFolder(DATA_DIR/"test", data_transforms["val"])
    }
    # Use num_workers=0 for maximum cross-platform compatibility
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) 
                   for x in ["train", "val"]}
    
    model_configs = {
        "mobilenet_v2": models.mobilenet_v2(weights="IMAGENET1K_V1"),
        "efficientnet_b0": models.efficientnet_b0(weights="IMAGENET1K_V1"),
        "custom_cnn": None
    }
    
    for name, base in model_configs.items():
        print(f"\n🔥 Training {name}...")
        if name == "mobilenet_v2":
            base.classifier[1] = nn.Linear(base.last_channel, 2)
        elif name == "efficientnet_b0":
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, 2)
        elif name == "custom_cnn":
            base = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
            )
            
        model = base.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        epochs = 10
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, "train")
            val_loss, val_acc = train_one_epoch(model, dataloaders["val"], criterion, optimizer, "val")
            print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            scheduler.step()
            
        torch.save(model.state_dict(), MODEL_DIR / f"{name}_mask.pth")
        print(f"✅ Saved models/{name}_mask.pth")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()