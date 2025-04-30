import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from LeNet5 import LeNet5

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışma cihazı: {device}")

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((32, 32)),
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)  # shuffle=False özellik çıkarımı için
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Eğitilmiş LeNet5 modelini yükle (eğer yoksa eğit)
model_path = 'lenet5_model.pth'
try:
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model başarıyla yüklendi: {model_path}")
except:
    print(f"Model bulunamadı: {model_path}. Yeni model eğitiliyor...")
    from train_lenet_models import train_and_evaluate_model
    model = LeNet5().to(device)
    _, _ = train_and_evaluate_model(model, "lenet5", num_epochs=5)

# Özellik çıkarıcı fonksiyon - fc2 katmanına kadar olan özellikleri çıkarır
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # conv1 + pool + conv2 + pool + flatten + fc1
        self.features = nn.Sequential(
            model.conv1,
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            model.conv2,
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = model.fc1

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        return x

# Özellik çıkarıcıyı oluştur
feature_extractor = FeatureExtractor(model).to(device)

# Tam modeli oluştur (özellik çıkarıcı + sınıflandırıcı)
class FullModelWithExtractor(nn.Module):
    def __init__(self, feature_extractor):
        super(FullModelWithExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(120, 10)  # FC2 ve FC3 yerine tek bir lineer katman

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# Eğitim veri seti için özellik çıkarımı
print("Eğitim veri seti için özellik çıkarımı yapılıyor...")
start_time = time.time()
train_features = []
train_labels = []

with torch.no_grad():
    for i, (data, targets) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)} işleniyor...")
        data = data.to(device)
        features = feature_extractor(data).cpu().numpy()
        train_features.append(features)
        train_labels.append(targets.numpy())

train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels)
print(f"Eğitim özellik çıkarımı tamamlandı: {time.time() - start_time:.2f} saniye")

# Test veri seti için özellik çıkarımı
print("Test veri seti için özellik çıkarımı yapılıyor...")
start_time = time.time()
test_features = []
test_labels = []

with torch.no_grad():
    for i, (data, targets) in enumerate(test_loader):
        if i % 20 == 0:
            print(f"Batch {i}/{len(test_loader)} işleniyor...")
        data = data.to(device)
        features = feature_extractor(data).cpu().numpy()
        test_features.append(features)
        test_labels.append(targets.numpy())

test_features = np.vstack(test_features)
test_labels = np.concatenate(test_labels)
print(f"Test özellik çıkarımı tamamlandı: {time.time() - start_time:.2f} saniye")

# Özellikleri kaydet
np.save("train_features.npy", train_features)
np.save("train_labels.npy", train_labels)
np.save("test_features.npy", test_features)
np.save("test_labels.npy", test_labels)

print(f"Özellik çıkarımı tamamlandı ve dosyalar kaydedildi.")
print(f"Eğitim özellikleri: {train_features.shape}")
print(f"Test özellikleri: {test_features.shape}")

# 5. Model: Tam CNN Modeli
# Karşılaştırma için aynı özellik çıkarıcıyı kullanan tam bir CNN modeli eğitelim
full_model = FullModelWithExtractor(feature_extractor).to(device)

# Eğitim parametreleri
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001)
num_epochs = 5

# Eğitim ve test doğruluk değerlerini sakla
train_accuracies = []
test_accuracies = []

# Eğitim döngüsü
print("\nTam CNN modeli eğitiliyor...")
for epoch in range(1, num_epochs + 1):
    # Eğitim modu
    full_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = full_model(data)
        loss = criterion(outputs, targets)
        
        # Backward ve optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if i % 100 == 99:
            print(f'Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%')
            running_loss = 0.0
    
    train_acc = 100. * correct / total
    train_accuracies.append(train_acc)
    
    # Test modu
    full_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = full_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_acc = 100. * correct / total
    test_accuracies.append(test_acc)
    print(f'Epoch {epoch}, Test Accuracy: {test_acc:.2f}%')

# Modeli kaydet
torch.save(full_model.state_dict(), 'full_model_with_extractor.pth')

# Sonuçları görselleştir
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='Test Accuracy')
plt.title('Full CNN with Feature Extractor')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('full_model_accuracy.png')
plt.show()

print("İşlem tamamlandı! Özellikler ve model kaydedildi.")