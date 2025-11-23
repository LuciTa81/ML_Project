import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, Sequential, CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import glob
from PIL import Image

# ===============================
# 1. 기본 설정
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/workspace/Images" 
IMG_SIZE = 224 # [최적화] VGG16은 224가 기본이며 속도가 더 빠릅니다 (299도 가능)
LR = 0.0001
EPOCHS = 10
BATCH_SIZE = 8
PATIENCE = 5
NUM_WORKERS = 2 # [최적화] 데이터 로딩 병렬 처리 (속도 향상 핵심)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. 데이터셋 및 로더
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_class = len(dataset.classes)
print("클래스 수:", num_class)
print("클래스 목록:", dataset.classes)

train_num = int(len(dataset)*0.7)
valid_num = int(len(dataset)*0.1)
test_num  = len(dataset) - train_num - valid_num

train_set, valid_set, test_set = random_split(
    dataset, [train_num, valid_num, test_num],
    generator=torch.Generator().manual_seed(SEED)
)

# num_workers 추가됨
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 전역 변수 history (학습 함수에서 채워질 예정)
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

# ===============================
# 3. Mixup 및 모델 정의
# ===============================
def mixup(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def build_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Fine-tuning (Conv5 block부터 학습)
    for name, param in model.features.named_parameters():
        layer_idx = int(name.split(".")[0])
        if layer_idx >= 24:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.classifier = Sequential(
        Linear(25088, 4096),
        ReLU(True),
        Dropout(0.3),
        Linear(4096, 2048),
        ReLU(True),
        Dropout(0.5),
        Linear(2048, num_class)
    )
    return model

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False

    def check(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ===============================
# 4. 학습 함수
# ===============================
def train(model, name="model"):
    model.to(device)
    crossentropyloss = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early = EarlyStopping(patience=PATIENCE)

    print(f"[{name}] 학습 시작...")

    for epoch in range(EPOCHS):
        model.train()
        sum_loss, sum_correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Mixup
            x, y_a, y_b, lam = mixup(x, y, alpha=1.0)

            optimizer.zero_grad()
            pred = model(x)
            loss = mixup_loss(crossentropyloss, pred, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_correct += (pred.argmax(1) == y).sum().item()

        train_loss = sum_loss / len(train_loader)
        train_acc  = sum_correct / len(train_set)

        # VALIDATION
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = crossentropyloss(pred, y)
                val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).sum().item()

        val_loss /= len(valid_loader)
        val_acc = val_correct / len(valid_set)
        
        # history에 기록
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step()

        early.check(val_acc)
        if early.early_stop:
            print(f"Early stopping triggered! (Best Val Acc: {early.best_acc:.4f})")
            break

    print(f"최종 Best Val Acc: {early.best_acc:.4f}")

# ===============================
# 5. 그래프 그리기 함수 (순서 수정됨)
# ===============================
def draw_chart():
    plt.rcParams["figure.figsize"] = [18, 8]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    plt.subplot(1, 2, 1)
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
    plt.savefig("vgg16_result.png", dpi=300)
    print("그래프 저장 완료: vgg16_result.png")

# ===============================
# 6. 예측 함수
# ===============================
def predict(model, folder_path):
    model.eval()
    image_files = glob.glob(os.path.join(folder_path, "*"))
    
    if not image_files:
        print(f"경로에 이미지가 없습니다: {folder_path}")
        return

    print("\n[예측 결과]")
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
            
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)
            class_index = pred.argmax(1).item()

        predict_breed = dataset.classes[class_index]
        print(f"파일: {os.path.basename(img_path)} -> 예측: {predict_breed}")

# ===============================
# 7. 실행부 (여기가 중요합니다!)
# ===============================
# 1) 모델 생성 및 학습
model = build_vgg16()
train(model, "VGG16")  # <--- 학습을 먼저 해야 history가 채워집니다.

# 2) 그래프 그리기 (학습 후 호출)
draw_chart()

# 3) 예측 하기
# 경로 수정: colab 경로(/content/...)가 아니라 현재 작업공간(/workspace/predict)으로 가정
predict_path = "/workspace/predict" 
if not os.path.exists(predict_path):
    os.makedirs(predict_path, exist_ok=True) # 폴더 없으면 생성

predict(model, predict_path)
