import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import os
import time

from LeNet5 import LeNet5
from ImLeNet5 import ImprovedLeNet5

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışma cihazı: {device}")

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

# Zamanlama fonksiyonu
def time_model_evaluation(model, test_loader, model_name):
    start_time = time.time()
    
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
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    print(f'{model_name} Accuracy: {accuracy:.2f}%')
    print(f'{model_name} Evaluation Time: {evaluation_time:.2f} seconds')
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Classification Report
    report = classification_report(all_targets, all_preds, digits=3)
    print(f"{model_name} Classification Report:")
    print(report)
    
    return accuracy, cm, all_preds, all_targets, evaluation_time

# 1. LeNet5 modeli
print("\nEvaluating LeNet5 model...")
lenet5 = LeNet5().to(device)
lenet5.load_state_dict(torch.load('lenet5_model.pth'))
lenet5.eval()
lenet_acc, lenet_cm, lenet_preds, lenet_targets, lenet_time = time_model_evaluation(lenet5, test_loader, "LeNet5")

# 2. Geliştirilmiş LeNet5 modeli
print("\nEvaluating Improved LeNet5 model...")
improved_lenet5 = ImprovedLeNet5().to(device)
improved_lenet5.load_state_dict(torch.load('improved_lenet5_model.pth'))
improved_lenet5.eval()
improved_lenet_acc, improved_lenet_cm, improved_lenet_preds, improved_lenet_targets, improved_lenet_time = time_model_evaluation(improved_lenet5, test_loader, "Improved LeNet5")

# VGG16 modeli için değerlendirme
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
print("\nEvaluating VGG16 Transfer Learning model...")
vgg_model = torchvision.models.vgg16(pretrained=False)
vgg_model.classifier[6] = nn.Linear(4096, 10)
vgg_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
vgg_model = vgg_model.to(device)
vgg_model.load_state_dict(torch.load('vgg16_transfer_model.pth'))
vgg_model.eval()
vgg_acc, vgg_cm, vgg_preds, vgg_targets, vgg_time = time_model_evaluation(vgg_model, vgg_test_loader, "VGG16 Transfer")

# 4. Hibrit Model (SVM) sonuçlarını değerlendir
print("\nEvaluating Hybrid Model (CNN+SVM)...")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# SVM için özellik dosyaları var mı kontrol et
if os.path.exists("test_features.npy") and os.path.exists("test_labels.npy"):
    # Test verilerini yükle
    X_test = np.load("test_features.npy")
    y_test = np.load("test_labels.npy")
    
    # SVM modelini oluştur ve eğit (daha önce eğitilmediyse)
    if not os.path.exists("svm_model.pkl"):
        from sklearn.externals import joblib
        print("Training SVM model...")
        X_train = np.load("train_features.npy")
        y_train = np.load("train_labels.npy")
        start_time = time.time()
        svm_model = SVC(kernel='rbf', C=10, gamma='scale')
        svm_model.fit(X_train, y_train)
        # Modeli kaydet
        joblib.dump(svm_model, "svm_model.pkl")
        svm_train_time = time.time() - start_time
        print(f"SVM training completed in {svm_train_time:.2f} seconds")
    else:
        from sklearn.externals import joblib
        svm_model = joblib.load("svm_model.pkl")
    
    # SVM modelini değerlendir
    start_time = time.time()
    svm_preds = svm_model.predict(X_test)
    svm_eval_time = time.time() - start_time
    
    svm_acc = accuracy_score(y_test, svm_preds) * 100
    svm_cm = confusion_matrix(y_test, svm_preds)
    svm_report = classification_report(y_test, svm_preds, digits=3)
    
    print(f"Hybrid Model (CNN+SVM) Accuracy: {svm_acc:.2f}%")
    print(f"Hybrid Model (CNN+SVM) Evaluation Time: {svm_eval_time:.2f} seconds")
    print("Hybrid Model (CNN+SVM) Classification Report:")
    print(svm_report)
else:
    print("SVM feature files not found. Please run feature_extraction.py first.")
    svm_acc = 0
    svm_cm = np.zeros((10, 10))
    svm_preds = []
    svm_eval_time = 0

# 5. Full CNN model (eğer hibrit model karşılaştırması için kullanılıyorsa)
try:
    from full_cnn_model import FullCNN
    print("\nEvaluating Full CNN model...")
    full_cnn = FullCNN().to(device)
    full_cnn.load_state_dict(torch.load('full_cnn_model.pth'))
    full_cnn.eval()
    full_cnn_acc, full_cnn_cm, full_cnn_preds, full_cnn_targets, full_cnn_time = time_model_evaluation(full_cnn, test_loader, "Full CNN")
except:
    print("Full CNN model file not found or error loading model.")
    full_cnn_acc = 0
    full_cnn_cm = np.zeros((10, 10))
    full_cnn_time = 0

# Sonuçları görselleştir
models = ['LeNet5', 'Improved\nLeNet5', 'VGG16\nTransfer', 'Hybrid\nCNN+SVM', 'Full CNN']
accuracies = [lenet_acc, improved_lenet_acc, vgg_acc, svm_acc, full_cnn_acc]
eval_times = [lenet_time, improved_lenet_time, vgg_time, svm_eval_time, full_cnn_time]

# Accuracy karşılaştırma grafiği
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
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

plt.savefig('model_accuracy_comparison.png')
plt.close()

# Evaluation time karşılaştırma grafiği
plt.figure(figsize=(12, 6))
bars = plt.bar(models, eval_times, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Model')
plt.ylabel('Evaluation Time (seconds)')
plt.title('Model Evaluation Time Comparison')

# Bar'ların üzerinde değerleri göster
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom')

plt.savefig('model_time_comparison.png')
plt.close()

# Confusion Matrix visualizations
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
sns.heatmap(lenet_cm, annot=True, fmt='d', cmap='Blues')
plt.title('LeNet5 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 4, 2)
sns.heatmap(improved_lenet_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Improved LeNet5 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 4, 3)
sns.heatmap(vgg_cm, annot=True, fmt='d', cmap='Reds')
plt.title('VGG16 Transfer Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 4, 4)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Purples')
plt.title('Hybrid Model (CNN+SVM) Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Sonuçları bir CSV dosyasına kaydet
import csv
with open('model_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Accuracy (%)', 'Evaluation Time (s)'])
    writer.writerow(['LeNet5', f'{lenet_acc:.2f}', f'{lenet_time:.2f}'])
    writer.writerow(['Improved LeNet5', f'{improved_lenet_acc:.2f}', f'{improved_lenet_time:.2f}'])
    writer.writerow(['VGG16 Transfer', f'{vgg_acc:.2f}', f'{vgg_time:.2f}'])
    writer.writerow(['Hybrid CNN+SVM', f'{svm_acc:.2f}', f'{svm_eval_time:.2f}'])
    writer.writerow(['Full CNN', f'{full_cnn_acc:.2f}', f'{full_cnn_time:.2f}'])

print("\nModel karşılaştırması tamamlandı. Sonuçlar CSV dosyasına ve görsellere kaydedildi.")

# Sonuçları bir sözlük olarak döndür (README güncellemesi için)
results = {
    'LeNet5': {'accuracy': lenet_acc, 'time': lenet_time},
    'Improved LeNet5': {'accuracy': improved_lenet_acc, 'time': improved_lenet_time},
    'VGG16 Transfer': {'accuracy': vgg_acc, 'time': vgg_time},
    'Hybrid CNN+SVM': {'accuracy': svm_acc, 'time': svm_eval_time},
    'Full CNN': {'accuracy': full_cnn_acc, 'time': full_cnn_time}
}

# Sonuçları ekrana yazdır
print("\nFinal Results Summary:")
print("="*50)
print(f"{'Model':<20} {'Accuracy (%)':<15} {'Time (s)':<10}")
print("-"*50)
for model, data in results.items():
    print(f"{model:<20} {data['accuracy']:<15.2f} {data['time']:<10.2f}")
print("="*50)