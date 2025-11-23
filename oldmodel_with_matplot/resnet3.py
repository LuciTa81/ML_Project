# ============================================
# Dog Breed Classification with ResNet50
# - ImageFolder + random_split (train/valid/test)
# - tqdm progress bar per epoch
# ============================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 진행상황 표시용

# ===========================
# 1. 설정 & 랜덤 시드 고정
# ===========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 데이터 경로 및 하이퍼파라미터
DATA_DIR = '/workspace/Images'
IMG_SIZE = 299           # ResNet50 권장 입력 크기
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 10              # 필요에 따라 조정
NUM_WORKERS = 2          # 환경에 따라 0/2/4 등

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===========================
# 2. 데이터셋 & DataLoader
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ImageFolder: DATA_DIR / class_name / image.jpg
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
total = len(dataset)
print("총 이미지 개수:", total)
print("클래스 목록:", dataset.classes)

# 클래스 수
NUM_CLASSES = len(dataset.classes)
print("NUM_CLASSES:", NUM_CLASSES)

# 7 : 1 : 2 비율로 train / valid / test 분할
train_cnt = int(total * 0.7)
valid_cnt = int(total * 0.1)
test_cnt  = total - train_cnt - valid_cnt

train_set, valid_set, test_set = random_split(
    dataset,
    [train_cnt, valid_cnt, test_cnt],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

use_cuda = torch.cuda.is_available()

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda
)

valid_loader = DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda
)

# ===========================
# 3. ResNet50 모델 정의
# ===========================
print("\n[INFO] Building ResNet50 model...")

try:
    # 최신 torchvision 스타일
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
except Exception:
    # 구버전 호환
    model = models.resnet50(pretrained=True)

# 마지막 FC를 우리가 원하는 클래스 수로 교체
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)
print(model.fc)

# ===========================
# 4. Loss, Optimizer, Scheduler
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# val accuracy 기준으로 learning rate 감소
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,   # 필요하면 np.sqrt(0.1) 등으로 조정 가능
    patience=3,
    min_lr=1e-7
)

# 간단 EarlyStopping 비슷하게
best_val_acc = 0.0
epochs_no_improve = 0
earlystop_patience = 10

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

# ===========================
# 5. 학습 루프 (tqdm 포함)
# ===========================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ----- Train -----
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    train_pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False)

    for x, y in train_pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size_curr = x.size(0)
        total_loss += loss.item() * batch_size_curr
        total_correct += (pred.argmax(1) == y).sum().item()
        total_samples += batch_size_curr

        curr_loss = total_loss / total_samples
        curr_acc = total_correct / total_samples
        train_pbar.set_postfix({"loss": f"{curr_loss:.4f}", "acc": f"{curr_acc:.4f}"})

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # ----- Validation -----
    model.eval()
    vloss, vcorrect, vsamples = 0.0, 0, 0

    val_pbar = tqdm(valid_loader, desc=f"Valid {epoch+1}/{EPOCHS}", leave=False)

    with torch.no_grad():
        for x, y in val_pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            batch_size_curr = x.size(0)
            vloss += loss.item() * batch_size_curr
            vcorrect += (pred.argmax(1) == y).sum().item()
            vsamples += batch_size_curr

            curr_vloss = vloss / vsamples
            curr_vacc = vcorrect / vsamples
            val_pbar.set_postfix({"loss": f"{curr_vloss:.4f}", "acc": f"{curr_vacc:.4f}"})

    val_loss = vloss / vsamples
    val_acc = vcorrect / vsamples

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

    # scheduler: val_acc 기준으로 lr 감소
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr:.8f}")

    # Early stopping 체크
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_resnet50_breeds.pth")
        print("  -> Best model saved.")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= earlystop_patience:
        print("Early stopping triggered.")
        break

print("\nResNet50 학습 완료!")

# ===========================
# 6. Test set 평가
# ===========================
print("\n[INFO] Evaluating on test set with best model...")

# 동일 구조의 새 모델에 best weight 로드
try:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    best_model = models.resnet50(weights=None)
except Exception:
    best_model = models.resnet50(pretrained=False)

in_features = best_model.fc.in_features
best_model.fc = nn.Linear(in_features, NUM_CLASSES)
best_model.load_state_dict(torch.load("best_resnet50_breeds.pth", map_location=device))
best_model = best_model.to(device)
best_model.eval()

test_loss, test_correct, test_samples = 0.0, 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Test", leave=False):
        x, y = x.to(device), y.to(device)
        pred = best_model(x)
        loss = criterion(pred, y)
        test_loss += loss.item() * x.size(0)
        test_correct += (pred.argmax(1) == y).sum().item()
        test_samples += y.size(0)

test_loss /= test_samples
test_acc = test_correct / test_samples

print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ===========================
# 7. 학습 곡선 시각화
# ===========================
plt.rcParams["figure.figsize"] = [18, 8]
epochs_range = range(1, len(history["train_loss"]) + 1)

plt.subplot(1, 2, 1)
plt.title("Accuracy")
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Valid Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Loss")
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("res50_train_valid_curve.png", dpi=300)
plt.show()
