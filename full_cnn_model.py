import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hibrit modellerle karşılaştırmak için tam bir CNN modeli
class FullCNN(nn.Module):
    def __init__(self):
        super(FullCNN, self).__init__()
        
        # Konvolüsyon katmanları
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # İlk blok
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # İkinci blok
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Üçüncü blok
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Düzleştirme
        x = x.view(-1, 128 * 4 * 4)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Eğitim fonksiyonu
def train_full_cnn():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Çalışma cihazı: {device}")
    
    # Veri dönüşümleri
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
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model, kayıp fonksiyonu ve optimizer
    model = FullCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Eğitim ve test doğruluk listeleri
    train_accuracies = []
    test_accuracies = []
    
    # Epoch sayısı
    num_epochs = 10
    
    # Eğitim döngüsü
    for epoch in range(1, num_epochs + 1):
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
            
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch} [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}, Accuracy: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        train_acc = 100. * correct / total
        train_accuracies.append(train_acc)
        
        # Test et
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
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # Modeli kaydet
    torch.save(model.state_dict(), 'full_cnn_model.pth')
    
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Full CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('full_cnn_accuracy.png')
    plt.show()
    
    print(f"Full CNN model eğitimi tamamlandı. Son test doğruluğu: {test_accuracies[-1]:.2f}%")
    return model

if __name__ == "__main__":
    train_full_cnn()
