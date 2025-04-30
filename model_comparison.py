import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

from LeNet5 import LeNet5
from ImLeNet5 import ImprovedLeNet5

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST için dönüşümler (LeNet modelleri için)
lenet_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test veri seti
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=lenet_transform
)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 1. LeNet5 modeli
lenet5 = LeNet5().to(device)
lenet5.load_state_dict(torch.load('lenet5_model.pth'))
lenet5.eval()

# 2. Geliştirilmiş LeNet5 modeli
improved_lenet5 = ImprovedLeNet5().to(device)
improved_lenet5.load_state_dict(torch.load('improved_lenet5_model.pth'))
improved_lenet5.eval()

# 5. CNN modeli (Eğer hibrit model karşılaştırması için kullanılıyorsa)
# full_cnn = CNN5().to(device)
# full_cnn.load_state_dict(torch.load('full_cnn_model.pth'))
# full_cnn.eval()

# Test fonksiyonu
def evaluate_model(model, test_loader, model_name):
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'{model_name} Accuracy: {accuracy:.2f}%')
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Classification Report
    report = classification_report(all_targets, all_preds, digits=3)
    print(f"{model_name} Classification Report:")
    print(report)
    
    return accuracy, cm, all_preds, all_targets

# Modelleri değerlendir
lenet_acc, lenet_cm, lenet_preds, lenet_targets = evaluate_model(lenet5, test_loader, "LeNet5")
improved_lenet_acc, improved_lenet_cm, improved_lenet_preds, improved_lenet_targets = evaluate_model(improved_lenet5, test_loader, "Improved LeNet5")

# VGG16 modeli için değerlendirmeyi burada yapacağız, ancak farklı giriş boyutu gerektirdiğinden
# ayrı bir transform kullanılacak
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

vgg_test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=vgg_transform
)

vgg_test_loader = DataLoader(vgg_test_dataset, batch_size=32, shuffle=False)

# 3. VGG16 Transfer Learning modeli
vgg_model = torchvision.models.vgg16(pretrained=False)
vgg_model.classifier[6] = nn.Linear(4096, 10)
vgg_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
vgg_model = vgg_model.to(device)
vgg_model.load_state_dict(torch.load('vgg16_transfer_model.pth'))
vgg_model.eval()

vgg_acc, vgg_cm, vgg_preds, vgg_targets = evaluate_model(vgg_model, vgg_test_loader, "VGG16 Transfer")

# 4. Hibrit Model (SVM) sonuçlarını yükle
# Bu kısım svm_classifier.py çalıştırıldıktan sonra güncellenebilir
# sklearn'in accuracy_score ve classification_report fonksiyonları kullanılarak sonuçlar alınabilir
hybrid_acc = 0  # SVM sonucunu buraya koyun

# Sonuçları görselleştir
models = ['LeNet5', 'Improved LeNet5', 'VGG16 Transfer', 'Hybrid SVM']
accuracies = [lenet_acc, improved_lenet_acc, vgg_acc, hybrid_acc]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)

# Bar'ların üzerinde değerleri göster
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom')

plt.savefig('model_comparison.png')
plt.show()

# Confusion Matrix visualizations
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(lenet_cm, annot=True, fmt='d', cmap='Blues')
plt.title('LeNet5 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 2)
sns.heatmap(improved_lenet_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Improved LeNet5 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 3)
sns.heatmap(vgg_cm, annot=True, fmt='d', cmap='Reds')
plt.title('VGG16 Transfer Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

print("Model karşılaştırması tamamlandı.")
