
import os, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import timm  # EfficientNetB7

# 랜덤 시드 고정
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# 데이터 경로 및 하이퍼파라미터
DATA_DIR = '/workspace/Images'
IMG_SIZE = 299; BATCH_SIZE = 8; LR = 0.0001; EPOCHS = 10; NUM_CLASSES = 120

# 데이터셋 및 로더
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
total = len(dataset)
train_cnt = int(total*0.7); valid_cnt = int(total*0.1); test_cnt  = total - train_cnt - valid_cnt
train_set, valid_set, test_set = random_split(dataset, [train_cnt, valid_cnt, test_cnt],
                                              generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

# 모델 정의
model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=NUM_CLASSES)

# 학습 루프
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("EfficientNetB7 학습 시작...")

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0,0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (pred.argmax(1)==y).sum().item()
        
    train_acc = total_correct / len(train_set)
    train_loss_avg = total_loss / len(train_loader) # ✅ 평균 Train Loss

    model.eval()
    vloss, vcorrect = 0,0
    with torch.no_grad():
        for x,y in valid_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            vloss += loss.item()
            vcorrect += (pred.argmax(1)==y).sum().item()
            
    valid_acc = vcorrect / len(valid_set)
    valid_loss_avg = vloss / len(valid_loader) # ✅ 평균 Valid Loss
    
    # ✅ 수정된 출력문
    print(f"[{epoch+1}/{EPOCHS}] Train Loss:{train_loss_avg:.4f}, Train Acc:{train_acc:.4f} | Valid Loss:{valid_loss_avg:.4f}, Valid Acc:{valid_acc:.4f}")

print("✅ EfficientNetB7 학습 완료!")
