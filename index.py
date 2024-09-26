# NOT: BU KODLAR, ERCAN IŞIK TARAFINDAN CHAT GPT'YE YAPTIRILMIŞTIR. 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Örnek veri seti oluşturuluyor
data = {
    'age': [25, 30, 35, 40, 45],
    'weight': [70, 80, 75, 85, 90],
    'height': [175, 180, 165, 170, 160],
    'steps': [10000, 12000, 8000, 11000, 9000]
}

# Veri seti bir DataFrame'e dönüştürülüyor
df = pd.DataFrame(data)

# Özellikler (X) ve hedef değişken (y) ayrılıyor
X = df[['age', 'weight', 'height']]
y = df['steps']

# Verilerin Eğitim ve Test Setlerine Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verilerin Ölçeklendirilmesi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Eğitim verileri ölçeklendiriliyor
X_test = scaler.transform(X_test)  # Test verileri aynı ölçekle dönüştürülüyor

# Yapay Sinir Ağı Modelinin Oluşturulması
model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))  # İlk katman, 32 nöronlu ve ReLU aktivasyon fonksiyonlu
model.add(Dense(16, activation='relu'))  # İkinci katman, 16 nöronlu ve ReLU aktivasyon fonksiyonlu
model.add(Dense(1, activation='linear'))  # Çıkış katmanı, 1 nöronlu ve lineer aktivasyon fonksiyonlu

# Modelin derlenmesi
model.compile(optimizer='adam', loss='mean_squared_error')  # Model, Adam optimizasyon algoritması ve MSE kayıp fonksiyonu ile derleniyor

# Modelin Eğitilmesi
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)  # Model, 50 epoch boyunca ve 10'luk mini-batch'ler ile eğitiliyor

# Modelin Test Edilmesi
predictions = model.predict(X_test)  # Test verileri ile tahminler yapılıyor
print("Gerçek Adım Sayıları:", y_test.values)  # Gerçek adım sayıları yazdırılıyor
print("Tahmin Edilen Adım Sayıları:", predictions.flatten())  # Tahmin edilen adım sayıları yazdırılıyor
