import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ==================== 0) RANDOM SEED ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==================== 1) 데이터 경로 및 설정 ====================
DATA_DIR = '/workspace/Images20'  # ✅ 20종만 들어있는 폴더
IMG_SIZE = 299
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 10

# ==================== 2) 데이터 변환 ====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==================== 3) 데이터셋 로드 ====================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES = len(dataset.classes)
print("클래스 수:", NUM_CLASSES)
print("클래스 목록:", dataset.classes)

# ==================== 4) Train / Valid / Test 분할 ====================
total = len(dataset)
train_cnt = int(total*0.7)
valid_cnt = int(total*0.1)
test_cnt  = total - train_cnt - valid_cnt

train_set, valid_set, test_set = random_split(
    dataset, [train_cnt, valid_cnt, test_cnt],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 5) 학습 시각화 함수 ====================
def plot_history(train_losses, val_losses, train_accs, val_accs, name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses,label='Train Loss')
    plt.plot(val_losses,label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs,label='Train Acc')
    plt.plot(val_accs,label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()

# ==================== 6) 학습 함수 (Loss 출력 수정) ====================
def train(model, name="model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (pred.argmax(1)==y).sum().item()
            
        train_acc = total_correct / len(train_set)
        train_loss_avg = total_loss / len(train_loader) # 평균 Train Loss
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)

        model.eval()
        vloss, vcorrect = 0,0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred,y)
                vloss += loss.item()
                vcorrect += (pred.argmax(1)==y).sum().item()
                
        valid_acc = vcorrect/len(valid_set)
        valid_loss_avg = vloss/len(valid_loader) # 평균 Valid Loss
        val_losses.append(valid_loss_avg)
        val_accs.append(valid_acc)
        
        # ✅ 수정된 출력문
        print(f"[{epoch+1}/{EPOCHS}] Train Loss:{train_loss_avg:.4f}, Train Acc:{train_acc:.4f} | Valid Loss:{valid_loss_avg:.4f}, Valid Acc:{valid_acc:.4f}")

    plot_history(train_losses, val_losses, train_accs, val_accs, name)
    print(f"✅ {name} 학습 완료!")

# ==================== 7) VGG16 모델 정의 ====================
def build_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad=False
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

# ==================== 8) 학습 실행 ====================
model = build_vgg16()
train(model, "VGG16")
