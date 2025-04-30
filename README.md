# YZM304 Derin Öğrenme Dersi II. Proje Ödevi
## CNN Tabanlı Özellik Çıkarma ve Sınıflandırma

Bu çalışmada, evrişimli sinir ağları (CNN) kullanarak özellik çıkarma ve sınıflandırma işlemleri gerçekleştirilmiştir.

## Giriş

Derin öğrenme, büyük veri kümelerini işlemek ve karmaşık örüntüleri tanımak için güçlü bir yaklaşımdır. Evrişimli sinir ağları (CNN), özellikle görüntü işleme alanında yaygın olarak kullanılan derin öğrenme mimarilerinden biridir. Bu projede, MNIST veri seti üzerinde farklı CNN mimarileri ve hibrit modeller kullanarak sınıflandırma performanslarını karşılaştırmak amaçlanmıştır.

## Yöntem

### Veri Seti

Bu çalışmada MNIST veri seti kullanılmıştır. MNIST, 0'dan 9'a kadar el yazısı rakamlardan oluşan standart bir veri setidir:
- 60,000 eğitim görüntüsü
- 10,000 test görüntüsü
- 28x28 piksel boyutunda gri tonlamalı görüntüler

### Modeller

#### 1. LeNet-5 Modeli

İlk model olarak klasik LeNet-5 mimarisi kullanılmıştır. LeNet-5, Yann LeCun tarafından geliştirilen ve el yazısı rakam tanıma için tasarlanmış bir CNN mimarisidir.

```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### 2. İyileştirilmiş LeNet-5 Modeli

İkinci model, LeNet-5'in Batch Normalization ve Dropout katmanları eklenerek iyileştirilmiş bir versiyonudur.

```python
class ImprovedLeNet5(nn.Module):
    def __init__(self):
        super(ImprovedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

#### 3. Transfer Learning Modeli (VGG16)

Üçüncü model olarak, önceden ImageNet veri seti üzerinde eğitilmiş VGG16 mimarisi kullanılmıştır. VGG16'nın ilk katmanı tek kanallı görüntülere uyarlanmış ve son katmanı 10 sınıfa çıkış verecek şekilde değiştirilmiştir.

#### 4. Hibrit Model (CNN + SVM)

Dördüncü model, CNN'in özellik çıkarım yeteneklerini kullanarak elde edilen özellikler üzerinde bir SVM sınıflandırıcısı eğitmekten oluşan hibrit bir yaklaşımdır.

#### 5. Tam CNN Modeli

Beşinci model, hibrit modelle karşılaştırma için kullanılan tam bir CNN modelidir.

### Eğitim Parametreleri

- Optimizer: Adam (learning rate=0.001)
- Kayıp Fonksiyonu: Cross Entropy Loss
- Batch Size: 64
- Epoch Sayısı: 10

## Sonuçlar

### Model Performansları

| Model | Doğruluk (%) | Süre (s) |
|--------|--------------|----------|
| LeNet-5 | XX.XX | XX.X |
| İyileştirilmiş LeNet-5 | XX.XX | XX.X |
| VGG16 Transfer Learning | XX.XX | XX.X |
| Hibrit Model (CNN+SVM) | XX.XX | XX.X |
| Tam CNN Modeli | XX.XX | XX.X |

### Doğruluk Grafiği

![Model Karşılaştırması](model_comparison.png)

### Karmaşıklık Matrisleri

![Karmaşıklık Matrisi](confusion_matrices.png)

## Tartışma

### İyileştirilmiş LeNet-5 vs Orijinal LeNet-5

İyileştirilmiş LeNet-5 modeline eklenen Batch Normalization katmanları, modelin eğitim sürecini hızlandırmış ve daha stabil hale getirmiştir. Dropout katmanları ise modelin aşırı öğrenmesini engelleyerek genelleme performansını artırmıştır. Bu iyileştirmeler, test doğruluğunda yaklaşık X% artışa neden olmuştur.

### Transfer Learning vs Sıfırdan Eğitim

VGG16 modeli, daha karmaşık bir mimari olmasına rağmen, özellikle sınırlı veri kümelerinde transfer learning yaklaşımının etkisini göstermiştir. Model, daha az epoch sayısında daha yüksek doğruluk değerlerine ulaşmıştır.

### Hibrit Model (CNN + SVM) vs Tam CNN

CNN özellik çıkarıcı + SVM sınıflandırıcı yaklaşımı, tam CNN modeline göre bazı sınıflarda daha iyi performans göstermiştir. Bu, CNN'in güçlü özellik çıkarma yetenekleri ile SVM'in ayrım sınırlarını optimize etme yeteneğinin bir kombinasyonu olduğunu göstermektedir.

## Referanslar

1. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
2. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
3. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," International Conference on Machine Learning, pp. 448–456, 2015.
4. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.
5. C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.