import numpy as np
import os
import random

from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from random_forest import RandomForest
from sklearn.metrics import classification_report
from mlp import MLP
from knn import KNN

# Đường dẫn đến tập dữ liệu hình ảnh
data_dir = "path_to_dataset\\test"
image_size = 64
categories = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]
X = []
y = []

for i, category in enumerate(categories):
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = image.load_img(img_path, target_size=(image_size, image_size))
        img_array = image.img_to_array(img_array)
        X.append(img_array)
        y.append(i)

X = np.array(X)
y = np.array(y)

# Chuẩn hóa hình ảnh (normalize pixel values to range 0-1)
X = X / 255.0

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten dữ liệu hình ảnh để sử dụng với các mô hình không phải CNN
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encoding cho nhãn
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Khởi tạo mô hình Random Forest
rf_model = RandomForest(n_trees=50, max_depth=20)
rf_model.fit(X_train, y_train)

# Dự đoán
rf_predictions = rf_model.predict(X_test)
print("Random Forest predictions:", rf_predictions)

# Khởi tạo mô hình MLP
mlp_model = MLP(input_size=X_train.shape[1], hidden_size=64, output_size=4)
y_train_mlp = np.argmax(y_train, axis=1)
y_test_mlp = np.argmax(y_test, axis=1)
mlp_model.fit(X_train, y_train_mlp, epochs=100)

# Dự đoán với mô hình MLP
mlp_predictions = np.argmax(mlp_model.predict(X_test), axis=1)
print("MLP predictions:", mlp_predictions)

# Khởi tạo mô hình KNN
knn_model = KNN(k=15)
knn_model.fit(X_train, y_train_mlp)

# Dự đoán với mô hình KNN
knn_predictions = knn_model.predict(X_test)
print("KNN predictions:", knn_predictions)

# Đánh giá mô hình (Random Forest, MLP và KNN)
print("Random Forest Classification Report:")
print(classification_report(y_test_mlp, rf_predictions))

print("MLP Classification Report:")
print(classification_report(y_test_mlp, mlp_predictions))

print("KNN Classification Report:")
print(classification_report(y_test_mlp, knn_predictions))
