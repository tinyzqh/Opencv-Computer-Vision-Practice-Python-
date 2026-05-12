"""Train a binary classifier (empty vs occupied) on parking-spot crops.

Transfer-learning from a pretrained VGG16 backbone (the first few conv blocks
are frozen). Saves the trained weights to ``car1.pth`` for inference by
``park_test.py``.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2
LR = 1e-4
MOMENTUM = 0.9
WEIGHTS_PATH = "car1.pth"
TRAIN_DIR = "train_data/train"
TEST_DIR = "train_data/test"


def build_model(num_classes: int) -> nn.Module:
    """VGG16 backbone with a fresh classifier head."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # Freeze the first ten conv layers (matches the original Keras setup)
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    for layer in conv_layers[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    # Replace classifier head with a single Linear (matches the Keras "Flatten + Dense")
    # VGG16 features output is (B, 512, IMG_SIZE/32, IMG_SIZE/32); at 48x48 input this is (B, 512, 1, 1)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, num_classes),
    )
    return model


def build_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)
    print(f"train={len(train_ds)}  test={len(test_ds)}  classes={train_ds.classes}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        train_ds.classes,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader, classes = build_loaders()
    model = build_model(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM,
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, WEIGHTS_PATH)
            print(f"  ↳ saved {WEIGHTS_PATH} (best={best_acc:.4f})")

    print(f"done. best val_acc={best_acc:.4f}  -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
