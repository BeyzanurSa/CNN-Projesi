import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışma cihazı: {device}")

# VGG16 modelini yükle (pretrained=True ile ImageNet üzerinde eğitilmiş ağırlıkları kullan)
model = torchvision.models.vgg16(pretrained=True)

# MNIST için son sınıflandırıcı katmanını değiştir (1000 sınıf -> 10 sınıf)
model.classifier[6] = nn.Linear(4096, 10)

# Giriş katmanını 1 kanallı (grayscale) görüntüler için uyarla
# VGG16'nın ilk katmanı 3 kanallı RGB görüntüler için tasarlanmıştır
# Grayscale görüntüleri 3 kanallı olarak kopyalayarak çözüm:
original_conv_layer = model.features[0]
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
# Ağırlıkları ortalamasını al, 1 kanallı görüntü için adapte et
model.features[0].weight.data = original_conv_layer.weight.data.mean(dim=1, keepdim=True)

# Modeli seçilen cihaza taşı
model = model.to(device)

# MNIST için dönüşümler
# VGG16 için giriş boyutu 224x224 olmalı
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Veri setlerini yükle
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
# Sadece sınıflandırıcı katmanını eğit (feature extraction yaklaşımı)
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Eğitim fonksiyonu
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}, Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

# Test fonksiyonu
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return accuracy

# Eğitim
num_epochs = 5  # VGG16 büyük bir model olduğu için epoch sayısını düşük tutuyoruz
train_accuracies = []
test_accuracies = []

for epoch in range(1, num_epochs + 1):
    train_acc = train(model, train_loader, optimizer, criterion, epoch)
    test_acc = test(model, test_loader, criterion)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Sonuçları görselleştir
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Transfer Learning Model Accuracy')
plt.legend()
plt.savefig('transfer_model_accuracy.png')
plt.show()

# Modeli kaydet
torch.save(model.state_dict(), 'vgg16_transfer_model.pth')
print("Eğitim tamamlandı ve model kaydedildi.")