# YZM304 Derin Öğrenme Dersi II. Proje Ödevi
## CNN Tabanlı Özellik Çıkarma ve Sınıflandırma

Bu çalışmada, evrişimli sinir ağları (CNN) kullanarak özellik çıkarma ve sınıflandırma işlemleri gerçekleştirilmiştir.

## Giriş

Derin öğrenme, büyük veri kümelerini işlemek ve karmaşık örüntüleri tanımak için güçlü bir yaklaşımdır. Evrişimli sinir ağları (CNN), özellikle görüntü işleme alanında yaygın olarak kullanılan derin öğrenme mimarilerinden biridir. Bu projede, MNIST veri seti üzerinde farklı CNN mimarileri ve hibrit modeller kullanarak sınıflandırma performanslarını karşılaştırmak amaçlanmıştır.

Proje kapsamında beş farklı model oluşturulmuş ve değerlendirilmiştir:
1. Klasik LeNet-5 modeli
2. Batch Normalization ve Dropout katmanları eklenmiş İyileştirilmiş LeNet-5 modeli
3. Transfer Learning yaklaşımı ile önceden eğitilmiş VGG16 modeli
4. CNN özellik çıkarıcı + SVM sınıflandırıcı hibrit modeli
5. Tam CNN modeli (hibrit modelle karşılaştırma için)

## Yöntem

### Veri Seti

Bu çalışmada MNIST veri seti kullanılmıştır. MNIST, 0'dan 9'a kadar el yazısı rakamlardan oluşan standart bir veri setidir:
- 60,000 eğitim görüntüsü
- 10,000 test görüntüsü
- 28x28 piksel boyutunda gri tonlamalı görüntüler

Veri önişleme adımları:
- Görüntüler 32x32 piksel boyutuna yeniden boyutlandırılmıştır (LeNet-5 modelleri için)
- Veri normalizasyonu yapılmıştır (piksel değerleri [-1, 1] aralığına dönüştürülmüştür)
- VGG16 modeli için görüntüler 224x224 piksel boyutuna yeniden boyutlandırılmıştır

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

```python
model = torchvision.models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 10)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
```

#### 4. Hibrit Model (CNN + SVM)

Dördüncü model, CNN'in özellik çıkarım yeteneklerini kullanarak elde edilen özellikler üzerinde bir SVM sınıflandırıcısı eğitmekten oluşan hibrit bir yaklaşımdır. Özellik çıkarıcı olarak eğitilmiş LeNet-5 modelinin konvolüsyon katmanları ve ilk tam bağlantılı katmanı (fc1) kullanılmıştır.

```python
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
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
```

Çıkarılan özellikler üzerinde SVM eğitimi:

```python
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
```

#### 5. Tam CNN Modeli

Beşinci model, hibrit modelle karşılaştırma için kullanılan tam bir CNN modelidir. Bu model, daha fazla konvolüsyon katmanı ve batch normalization içermektedir.

```python
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
```

### Eğitim Parametreleri

- Optimizer: Adam (learning rate=0.001) ve SGD (VGG16 için)
- Kayıp Fonksiyonu: Cross Entropy Loss
- Batch Size: 64 (VGG16 için 32)
- Epoch Sayısı: 10 (VGG16 için 5)

## Sonuçlar

### Model Performansları

| Model | Doğruluk (%) | Eğitim Süresi (s) |
|--------|--------------|----------|
| LeNet-5 | 98.42 | 67.3 |
| İyileştirilmiş LeNet-5 | 99.17 | 83.6 |
| VGG16 Transfer Learning | 98.95 | 312.8 |
| Hibrit Model (CNN+SVM) | 97.78 | 41.5 |
| Tam CNN Modeli | 99.24 | 95.1 |


## Tartışma

### İyileştirilmiş LeNet-5 vs Orijinal LeNet-5

İyileştirilmiş LeNet-5 modeline eklenen Batch Normalization katmanları, modelin eğitim sürecini hızlandırmış ve daha stabil hale getirmiştir. Batch Normalization, her katmandaki girişleri normalize ederek iç değişken değişimini azaltır ve daha hızlı ve stabil öğrenme sağlar. Dropout katmanları ise modelin aşırı öğrenmesini engelleyerek genelleme performansını artırmıştır.

Bu iyileştirmeler, test doğruluğunda yaklaşık %0.75'lik bir artışa neden olmuştur. Eğitim süresi biraz daha uzun olmasına rağmen, model performansındaki artış bu uzamayı dengelemektedir. İyileştirilmiş LeNet-5, özellikle rakam "8" ve "9" gibi karmaşık şekillerin sınıflandırılmasında daha başarılı olmuştur.

### Transfer Learning vs Sıfırdan Eğitim

VGG16 modeli, daha karmaşık bir mimari olmasına rağmen, özellikle sınırlı veri kümelerinde transfer learning yaklaşımının etkisini göstermiştir. Önceden ImageNet üzerinde eğitilmiş ağırlıkların kullanılması, modelin daha az epoch sayısında yüksek doğruluk değerlerine ulaşmasını sağlamıştır. VGG16, yaklaşık %98.95 doğruluk oranı elde etmiştir.

Ancak, VGG16'nın eğitim süresi diğer modellere göre çok daha uzundur (312.8 saniye). Bu, modelin daha karmaşık yapısından ve daha fazla parametre içermesinden kaynaklanmaktadır. MNIST gibi göreceli olarak basit bir veri seti için, VGG16'nın sağladığı ek doğruluk artışı, hesaplama maliyetini karşılamayabilir.

### Hibrit Model (CNN + SVM) vs Tam CNN

CNN özellik çıkarıcı + SVM sınıflandırıcı yaklaşımı (%97.78), tam CNN modeline (%99.24) göre daha düşük doğruluk oranı göstermiştir. Bu sonuç, derin öğrenme modellerinin uçtan uca eğitiminin, özellik çıkarımı ve sınıflandırmanın ayrı ayrı optimize edilmesine göre daha etkili olabileceğini göstermektedir.

Ancak, hibrit modelin eğitim süresi (41.5 saniye) tam CNN modeline göre (95.1 saniye) çok daha kısadır. Bu durum, hesaplama kaynakları sınırlı olduğunda veya hızlı bir şekilde sonuç alınması gerektiğinde hibrit modellerin tercih edilebileceğini göstermektedir. Ayrıca, hibrit model daha az veri ile daha iyi genelleme yapabilir.

### Genel Değerlendirme

Sonuçlar, modern CNN tekniklerinin (Batch Normalization, Dropout) klasik modellere göre önemli iyileştirmeler sağladığını göstermektedir. En iyi performansı tam CNN modeli göstermiştir (%99.24), ancak iyileştirilmiş LeNet-5 modeli de neredeyse aynı doğruluk oranına (%99.17) ulaşarak daha basit bir mimari ile yüksek performans elde edilebileceğini kanıtlamıştır.

Transfer learning, özellikle sınırlı veri setlerinde etkili bir yaklaşım olsa da, MNIST gibi basit veri setlerinde sıfırdan eğitilen özel modeller kadar veya daha iyi performans gösterebilmektedir. Hibrit modeller ise daha hızlı eğitim süresi sunmakla birlikte, tam CNN modelleri kadar yüksek doğruluk sağlayamamıştır.

## Referanslar

1. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
2. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
3. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," International Conference on Machine Learning, pp. 448–456, 2015.
4. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.
5. C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.