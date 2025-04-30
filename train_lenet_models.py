import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from LeNet5 import LeNet5 
from ImLeNet5 import ImprovedLeNet5 

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışma cihazı: {device}")

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Eğitim ve test veri setlerini indir
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Veri seti başarıyla yüklendi ve ön işlemeler tamamlandı.")

def train(model, device, train_loader, optimizer, criterion, epoch, model_name):
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
        
        # Doğruluk hesapla
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 99:  # Her 100 batch'te bir yazdır
            print(f'Epoch {epoch} [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}, Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    train_accuracy = 100. * correct / total
    return train_accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

# Her iki modeli eğit ve karşılaştır
def train_and_evaluate_model(model, model_name, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- {model_name} Epoch {epoch}/{num_epochs} ---")
        train_acc = train(model, device, train_loader, optimizer, criterion, epoch, model_name)
        test_acc = test(model, device, test_loader, criterion)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # Modeli kaydet
    torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_model.pth')
    
    return train_accuracies, test_accuracies

# 1. LeNet5 Modeli
lenet5 = LeNet5().to(device)
print("\n=== LeNet5 Modeli Eğitimi ===")
lenet5_train_acc, lenet5_test_acc = train_and_evaluate_model(lenet5, "lenet5", num_epochs=10)

# 2. İyileştirilmiş LeNet5 Modeli
improved_lenet5 = ImprovedLeNet5().to(device)
print("\n=== İyileştirilmiş LeNet5 Modeli Eğitimi ===")
improved_lenet5_train_acc, improved_lenet5_test_acc = train_and_evaluate_model(improved_lenet5, "improved_lenet5", num_epochs=10)

# Sonuçları Görselleştir
plt.figure(figsize=(12, 5))

# Eğitim Doğruluğu
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), lenet5_train_acc, 'b-', label='LeNet5')
plt.plot(range(1, 11), improved_lenet5_train_acc, 'g-', label='Improved LeNet5')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Test Doğruluğu
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), lenet5_test_acc, 'b-', label='LeNet5')
plt.plot(range(1, 11), improved_lenet5_test_acc, 'g-', label='Improved LeNet5')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lenet_model_comparison.png')
plt.show()

print("Model eğitimi ve karşılaştırma tamamlandı.")
print(f"LeNet5 Son Test Doğruluğu: {lenet5_test_acc[-1]:.2f}%")
print(f"İyileştirilmiş LeNet5 Son Test Doğruluğu: {improved_lenet5_test_acc[-1]:.2f}%")

# İyileştirme farkı
improvement = improved_lenet5_test_acc[-1] - lenet5_test_acc[-1]
print(f"İyileştirme: {improvement:.2f}%")
