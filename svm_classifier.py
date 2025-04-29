import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Verileri Yükle
X_train = np.load("train_features.npy")
y_train = np.load("train_labels.npy")
X_test = np.load("test_features.npy")
y_test = np.load("test_labels.npy")

# 2. SVM Modeli
clf = SVC(kernel='rbf')  # veya 'linear'
clf.fit(X_train, y_train)

# 3. Tahmin ve Değerlendirme
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
