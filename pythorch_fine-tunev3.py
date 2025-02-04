# PyTorch version of the provided TensorFlow/Keras code
# -------------------------------------------------------
# 본 코드는 TensorFlow/Keras로 구성된 코드를 PyTorch로 변환한 예시입니다.
# 코드 구조가 꽤 크고, 다양한 하이퍼파라미터 반복(tainableLen, nodecnt 등)이 있기 때문에
# 핵심 아이디어를 중심으로 간략화/재구성했습니다.
#
# * 주의사항 *
# 1. TensorFlow ImageDataGenerator & flow_from_directory -> PyTorch Dataset & DataLoader
# 2. ReduceLROnPlateau, EarlyStopping, ModelCheckpoint -> PyTorch의 Scheduler, 수동 EarlyStopping 로직, 모델 저장 로직
# 3. F1Score(tf.addons) -> scikit-learn의 f1_score 또는 Custom metric
# 4. Model summary, callbacks -> PyTorch는 콜백 개념이 없으므로, 수동으로 처리
#
# 아래 코드는 예시이며, 실제 환경에 맞게 수정해야 합니다.
#

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score
import shutil
import argparse

# -------------------------------------------------------
# 하이퍼파라미터
# -------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
STARTLAYER = 124  # 예시
LAYERS_TRAINABLE = 1
NODE_COUNTS = [2048, 1024]

# val_acc, val_loss, val_f1 (중 하나)
MONITOR = 'val_acc'
if MONITOR == 'val_acc':
    op_mode = 'max'
elif MONITOR == 'val_loss':
    op_mode = 'min'
elif MONITOR == 'val_f1':
    op_mode = 'max'
else:
    op_mode = 'max'

OBJECT_TYPES = ['ch']


# -------------------------------------------------------
# PyTorch용 모델 정의 함수 (EfficientNet B4 사용 예시)
# -------------------------------------------------------

def get_FineTuneModel(num_classes, tainableLen=124, node_count=1024, dropout_rate=0.33):
    # torchvision.models.efficientnet_b4 (pretrained)
    model_ft = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    # 특정 레이어 이하 freeze
    # 실제로는 PyTorch에서 레이어 번호로 접근하기 번거로움. 필요하다면 세부 레이어를 지정해서 freeze
    # 여기서는 간단히 모든 feature extractor 부분을 freeze하고 classifier만 수정
    for param in model_ft.features.parameters():
        param.requires_grad = False

    # classifier 수정
    in_features = model_ft.classifier[1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, node_count),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(node_count, num_classes)
    )

    return model_ft


# -------------------------------------------------------
# EarlyStopping 구현 (PyTorch 콜백 없음 -> 수동 구현)
# -------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == 'max':
                if current_score < self.best_score + self.min_delta:
                    self.counter += 1
                else:
                    self.best_score = current_score
                    self.counter = 0
            elif self.mode == 'min':
                if current_score > self.best_score - self.min_delta:
                    self.counter += 1
                else:
                    self.best_score = current_score
                    self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True

# -------------------------------------------------------
# train/validation 루프
# -------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # F1 score 계산 (macro)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1


# -------------------------------------------------------
# main 함수 형태로 구현 예시
# -------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 경로 (Keras 코드와 유사하게)
    base_dir = os.path.join(os.getcwd(), 'datasets', 'out1')
    # 예:  base_dir/character/train, base_dir/character/validation 등

    # Transform (Keras ImageDataGenerator와 비슷하게)
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 예시: 하나의 OBJECT_TYPE, 하나의 tainableLen, 하나의 nodecnt 에 대해서 동작.
    for DEFAULT_OBJ_TYPE in OBJECT_TYPES:
        for tainableLen in range(STARTLAYER, STARTLAYER + LAYERS_TRAINABLE):
            for nodecnt in NODE_COUNTS:

                class_str = 'character'  # 예시 "ch"
                data_dir_train = os.path.join(base_dir, class_str, 'train')
                data_dir_val = os.path.join(base_dir, class_str, 'validation')

                # ImageFolder로 Dataset 생성
                train_dataset = datasets.ImageFolder(data_dir_train, transform=train_transform)
                val_dataset = datasets.ImageFolder(data_dir_val, transform=val_transform)

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

                # 클래스 정보
                num_classes = len(train_dataset.classes)
                print(f"num_classes: {num_classes}, categories: {train_dataset.classes}")

                # 모델 생성
                model = get_FineTuneModel(num_classes, tainableLen, nodecnt, dropout_rate=0.33)
                model = model.to(device)

                # 손실함수, 옵티마이저, 스케줄러
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                scheduler = ReduceLROnPlateau(optimizer, mode=op_mode, factor=0.1, patience=5, verbose=True, min_lr=1e-6)

                # EarlyStopping, 수동으로 구현
                early_stopping = EarlyStopping(patience=PATIENCE, mode=op_mode)

                best_val_metric = -np.inf if op_mode == 'max' else np.inf
                best_model_path = None

                # 로그 기록용
                train_acc_list, val_acc_list = [], []
                train_loss_list, val_loss_list = [], []
                train_f1_list, val_f1_list = [], []

                # 학습 루프
                for epoch in range(EPOCHS):
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                    train_f1_list.append(0.0)  # train f1 별도 계산하려면 별도 로직 필요
                    val_f1_list.append(val_f1)

                    # ReduceLROnPlateau 갱신
                    if MONITOR == 'val_acc':
                        scheduler.step(val_acc)
                    elif MONITOR == 'val_loss':
                        scheduler.step(val_loss)
                    elif MONITOR == 'val_f1':
                        scheduler.step(val_f1)
                    else:
                        scheduler.step(val_acc)

                    # 모니터링 지표 계산
                    if MONITOR == 'val_acc':
                        current_metric = val_acc
                    elif MONITOR == 'val_loss':
                        current_metric = val_loss
                    elif MONITOR == 'val_f1':
                        current_metric = val_f1
                    else:
                        current_metric = val_acc

                    if op_mode == 'max':
                        is_best = current_metric > best_val_metric
                    else:  # 'min'
                        is_best = current_metric < best_val_metric

                    if is_best:
                        best_val_metric = current_metric
                        # 모델 가중치 저장
                        weight_filename = (
                            f"model_epoch_{epoch+1}_val_acc_{val_acc:.4f}_val_loss_{val_loss:.4f}_val_f1_{val_f1:.4f}.pth"
                        )
                        torch.save(model.state_dict(), weight_filename)
                        best_model_path = weight_filename

                    print(f"Epoch [{epoch+1}/{EPOCHS}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

                    # EarlyStopping 체크
                    early_stopping(current_metric)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break

                # 마지막 혹은 best 모델 저장
                if best_model_path is not None:
                    print(f"Best model saved at {best_model_path}")
                else:
                    # 아무 개선이 없었을 경우(이론적으론 드뭄)
                    model_save_filename = f"final_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
                    torch.save(model.state_dict(), model_save_filename)

                # 그래프 출력
                epochs_range = range(1, len(train_acc_list) + 1)
                plt.figure()
                plt.plot(epochs_range, train_acc_list, 'bo-', label='Train Acc')
                plt.plot(epochs_range, val_acc_list, 'ro-', label='Val Acc')
                plt.title('Training and Validation Accuracy')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(epochs_range, train_loss_list, 'bo-', label='Train Loss')
                plt.plot(epochs_range, val_loss_list, 'ro-', label='Val Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(epochs_range, val_f1_list, 'go-', label='Val F1')
                plt.title('Validation F1 Score')
                plt.legend()
                plt.show()


if __name__ == "__main__":
    main()

# -------------------------------------------------------
# 위 코드는 TensorFlow/Keras 기반 코드를 PyTorch로 포팅한 예시입니다.
# 1) DataLoader 부분에서 train/validation 디렉토리 구조가 ImageFolder 형식에 맞아야 합니다.
# 2) get_FineTuneModel 함수의 layer freeze 로직은 실제로는 세밀하게 설정해야 합니다.
# 3) EarlyStopping, ModelCheckpoint는 PyTorch에서 별도 클래스로 작성하거나 직접 로직으로 구현합니다.
# 4) "imgaug"와 같은 외부 augmentation 라이브러리는 PyTorch transforms와 별도로 처리해야 합니다.
# 5) MONITOR 지표(val_acc, val_loss, val_f1)에 따라 체크포인트와 스케줄러를 다르게 제어.
# -------------------------------------------------------
