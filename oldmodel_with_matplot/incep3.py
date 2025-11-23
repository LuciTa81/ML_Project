import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED)

TRAIN_DIR = '/workspace/Images'
IMG_SIZE = 299 # Changed from 150 to 299 for InceptionV3
BATCH_SIZE = 8
NUM_CLASSES = 120

# ✅ InceptionV3 권장 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ✅ 전체 데이터셋 로드
full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)

total_count = len(full_dataset)
train_count = int(total_count * 0.7)
valid_count = int(total_count * 0.1)
test_count  = total_count - train_count - valid_count

train_ds, valid_ds, test_ds = random_split(
    full_dataset,
    [train_count, valid_count, test_count],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print("Train:", len(train_ds))
print("Valid:", len(valid_ds))
print("Test :", len(test_ds))

# ✅ InceptionV3 불러오기
model = models.inception_v3(weights="IMAGENET1K_V1")
model.aux_logits = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# ✅ 모든 레이어 freeze (너 코드 유지)
for param in model.parameters():
    param.requires_grad = False

# ✅ classifier만 학습
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

EPOCHS = 10
# 학습
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        _, pred_labels = torch.max(preds, 1)
        train_correct += (pred_labels == y).sum().item()
        train_total += y.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Validation 계산
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            valid_loss += loss.item() * x.size(0)
            _, pred_labels = torch.max(preds, 1)
            valid_correct += (pred_labels == y).sum().item()
            valid_total += y.size(0)

    avg_valid_loss = valid_loss / valid_total
    valid_acc = valid_correct / valid_total

    #history
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_valid_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(valid_acc)

    # 로그 출력
    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.3f}, Train Acc: {train_acc:.4f} | "
          f"Valid Loss: {avg_valid_loss:.3f}, Valid Acc: {valid_acc:.4f}")

epochs = range(1, len(history["train_loss"]) + 1)
plt.rcParams["figure.figsize"] = [18, 8]

# Accuracy
plt.subplot(1, 2, 1)
plt.title("Accuracy")
plt.plot(epochs, history["train_acc"], label="Train Acc")
plt.plot(epochs, history["val_acc"], label="Valid Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.title("Loss")
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("incep3_history.png", dpi=300)
plt.show()

# ✅ 평가 함수 (model.evaluate equivalent)
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            _, pred_labels = torch.max(preds, 1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total

print()
print("Valid accuracy:", evaluate(valid_loader))
print("Test accuracy :", evaluate(test_loader))

