import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from LeNet5 import LeNet5 
from ImLeNet5 import ImprovedLeNet5 

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dönüşümler:
# - Resize(32x32): LeNet-5 mimarisi 32x32 giriş alır.
# - ToTensor(): Görüntüyü tensöre dönüştürür.
# - Normalize(): Verileri [-1, 1] aralığına çeker.
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST: grayscale olduğu için tek kanal
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


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        print(f"Target shape: {target.shape}")  # Hedeflerin boyutunu kontrol edin
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 100 == 99:  # Her 100 batch'te bir yazdır
            print(f'Epoch {epoch} [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Modeli seç (LeNet5 veya ImprovedLeNet5)
model = LeNet5().to(device)
# model = ImprovedLeNet5().to(device)  # Bunu kullanmak istersen üsttekini yorum satırı yap

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Epoch sayısı
num_epochs = 10

# Eğitim ve test döngüsü
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

