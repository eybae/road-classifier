import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import models
from sklearn.metrics import classification_report
from torch.optim import lr_scheduler
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 데이터 증강 및 전처리
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로딩
train_dataset = torchvision.datasets.ImageFolder(root='augmented_dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation 데이터셋 준비 (Train 데이터의 10%로 분할)
train_data, val_data = train_test_split(train_dataset.imgs, test_size=0.1, random_state=42)
train_dataset.imgs = train_data
val_dataset = torchvision.datasets.ImageFolder(root='augmented_dataset/val', transform=transform)
val_dataset.imgs = val_data

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 설정: ResNet34
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3개 클래스로 설정

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Early Stopping을 위한 변수 설정
patience = 5  # 성능 향상이 없는 에폭 수
best_val_loss = float('inf')
epochs_without_improvement = 0

# 학습을 위한 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    # Early Stopping을 위한 변수 설정
    best_val_loss = float('inf')  # 초기화
    epochs_without_improvement = 0

    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct / total
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        
        # 저장
        all_train_losses.append(epoch_train_loss)
        all_train_accuracies.append(epoch_train_accuracy)
        all_val_losses.append(epoch_val_loss)
        all_val_accuracies.append(epoch_val_accuracy)
        
        # Validation 성능 개선 여부 체크
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'models/best_model_91.67.pth')  # 모델 저장
        else:
            epochs_without_improvement += 1
        
        # Early Stopping
        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break
        
        scheduler.step()

        # 출력
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Accuracy: {epoch_train_accuracy:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_accuracy:.4f}")
    
    return all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies
# 모델 학습 실행
train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30)

# 결과 시각화 (학습 손실 및 정확도)
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 학습 손실
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')  # 저장
    plt.show()

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# 모델 로딩
model.load_state_dict(torch.load('best_model.pth'))

# Confusion Matrix
def plot_confusion_matrix(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')  # 저장
    plt.show()

plot_confusion_matrix(model, val_loader)

# 분류 보고서 저장
def save_classification_report(model, data_loader, filename='classification_report.txt'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"Classification report saved to {filename}")

save_classification_report(model, val_loader)

