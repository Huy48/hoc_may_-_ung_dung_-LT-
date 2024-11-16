import numpy as np
import os

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Class KNN
class KNN:
	def __init__(self, k=5, weighted=False):
		self.k = k
		self.weighted = weighted

	def fit(self, X, y):
		self.scaler = StandardScaler()
		self.X_train = self.scaler.fit_transform(X)
		self.y_train = y

	def predict(self, X):
		X = self.scaler.transform(X)
		y_pred = [self._predict(x) for x in X]
		return np.array(y_pred)

	def _predict(self, x):
		distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
		
		# Lấy chỉ số của k điểm gần nhất
		k_indices = np.argsort(distances)[:self.k]
		
		# Lấy nhãn của các điểm gần nhất
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		
		k_nearest_labels = [tuple(label) if isinstance(label, np.ndarray) else label for label in k_nearest_labels]
		
		if self.weighted:
			weights = [1 / (distances[i] + 1e-5) for i in k_indices]  # Tránh chia cho 0
			weighted_counts = Counter()
			for label, weight in zip(k_nearest_labels, weights):
				weighted_counts[label] += weight
			return weighted_counts.most_common(1)[0][0]
		
		most_common = Counter(k_nearest_labels).most_common(1)
		return most_common[0][0]

# Đường dẫn đến dữ liệu
train_data_dir = "path_to_dataset\\train"
test_data_dir = "path_to_dataset\\test"

image_size = 64
categories = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]

def load_data(data_dir):
	X, y = [], []
	for i, category in enumerate(categories):
		path = os.path.join(data_dir, category)
		for img in os.listdir(path):
			img_path = os.path.join(path, img)
			img_array = image.load_img(img_path, target_size=(image_size, image_size))
			img_array = image.img_to_array(img_array)
			X.append(img_array)
			y.append(i)
	return np.array(X), np.array(y)

# Load train và test data
X_train, y_train = load_data(train_data_dir)
X_test, y_test = load_data(test_data_dir)

# Chuẩn hóa pixel
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten dữ liệu
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Khởi tạo mô hình KNN
knn_model = KNN(k=5, weighted=False)

# Huấn luyện
knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
knn_predictions = knn_model.predict(X_test)

# Đánh giá mô hình
print("Đánh giá thuật toán KNN:")
print(classification_report(y_test, knn_predictions, target_names=categories))

# Dự đoán ảnh bên ngoài
external_image_path = "c_test.png"

def preprocess_image(image_path, image_size):
	img = image.load_img(image_path, target_size=(image_size, image_size))  # Tải ảnh và resize
	img_array = image.img_to_array(img)  # Chuyển đổi thành mảng
	img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel
	img_array = img_array.flatten()  # Chuyển thành vector 1D
	return np.array([img_array])  # Trả về mảng 2D (1, số feature)

# Chuẩn bị ảnh bên ngoài
external_image = preprocess_image(external_image_path, image_size)

# Sử dụng mô hình KNN để dự đoán
predicted_class_index = knn_model.predict(external_image)[0]
predicted_class = categories[predicted_class_index]

# In kết quả dự đoán
print(f"Ảnh bên ngoài được dự đoán là thuộc lớp: {predicted_class}")
