import numpy as np
import os

from collections import Counter
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClass
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Class Random Forest
class RandomForest:
	def __init__(self, n_trees=100, max_depth=10):
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.trees = []

	def fit(self, X, y):
		for _ in range(self.n_trees):
			tree = DecisionTreeClass(max_depth=self.max_depth)
			idxs = np.random.choice(len(y), len(y), replace=True)
			tree.fit(X[idxs], y[idxs])
			self.trees.append(tree)

	def predict(self, X):
		tree_preds = np.array([tree.predict(X) for tree in self.trees])
		final_preds = np.array([Counter(tree_preds[:, i].astype(int)).most_common(1)[0][0]for i in range(tree_preds.shape[1])])
		return final_preds

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

# Khởi tạo mô hình Random Forest
rf_model = RandomForest(n_trees=15, max_depth=6)


# Huấn luyện
rf_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
rf_predictions = rf_model.predict(X_test)

# Đánh giá mô hình
print("Đánh giá thuật toán Random Forest:")
print(classification_report(y_test, rf_predictions, target_names=categories))

# Dự đoán ảnh bên ngoài
external_image_path = "c_test.png"

def preprocess_image(image_path, image_size):
	img = image.load_img(image_path, target_size=(image_size, image_size))  # Tải ảnh và resize
	img_array = image.img_to_array(img)
	img_array = img_array / 255.0
	img_array = img_array.flatten()
	return np.array([img_array])

# Chuẩn bị ảnh bên ngoài
external_image = preprocess_image(external_image_path, image_size)

# Sử dụng mô hình Random Forest để dự đoán
predicted_class_index = rf_model.predict(external_image)[0]
predicted_class = categories[predicted_class_index]

# In kết quả dự đoán
print(f"Ảnh dự đoán là: {predicted_class}")
