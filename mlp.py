import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Lớp MLP
class MLP:
	def __init__(self, input_size, hidden_size, output_size, lr=0.001):
		self.lr = lr
		self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
		self.weights2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
		self.weights3 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
		self.bias1 = np.zeros((1, hidden_size))
		self.bias2 = np.zeros((1, hidden_size))
		self.bias3 = np.zeros((1, output_size))

	def leaky_relu(self, x, alpha=0.01):
		return np.where(x > 0, x, alpha * x)

	def leaky_relu_deriv(self, x, alpha=0.01):
		return np.where(x > 0, 1, alpha)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		return x * (1 - x)

	def batch_normalization(self, x):
		mean = np.mean(x, axis=0)
		std = np.std(x, axis=0)
		return (x - mean) / (std + 1e-7)

	def dropout(self, layer, dropout_rate=0.2):
		mask = np.random.binomial(1, 1 - dropout_rate, size=layer.shape)
		return layer * mask / (1 - dropout_rate)

	def fit(self, X, y, epochs=200):
		y_one_hot = np.zeros((X.shape[0], self.weights3.shape[1]))
		y_one_hot[np.arange(X.shape[0]), y] = 1

		for epoch in range(epochs):
			# Forward pass
			hidden1 = self.leaky_relu(np.dot(X, self.weights1) + self.bias1)
			hidden1 = self.batch_normalization(hidden1)
			hidden2 = self.leaky_relu(np.dot(hidden1, self.weights2) + self.bias2)
			hidden2 = self.batch_normalization(hidden2)
			hidden2 = self.dropout(hidden2, dropout_rate=0.2)
			output = self.sigmoid(np.dot(hidden2, self.weights3) + self.bias3)

			# Backpropagation
			output_error = y_one_hot - output
			d_weights3 = self.lr * np.dot(hidden2.T, output_error * self.sigmoid_deriv(output))
			d_bias3 = self.lr * np.sum(output_error * self.sigmoid_deriv(output), axis=0, keepdims=True)

			hidden2_error = np.dot(output_error, self.weights3.T)
			d_weights2 = self.lr * np.dot(hidden1.T, hidden2_error * self.leaky_relu_deriv(hidden2))
			d_bias2 = self.lr * np.sum(hidden2_error * self.leaky_relu_deriv(hidden2), axis=0, keepdims=True)

			hidden1_error = np.dot(hidden2_error, self.weights2.T)
			d_weights1 = self.lr * np.dot(X.T, hidden1_error * self.leaky_relu_deriv(hidden1))
			d_bias1 = self.lr * np.sum(hidden1_error * self.leaky_relu_deriv(hidden1), axis=0, keepdims=True)

			# Update weights and biases
			self.weights1 += d_weights1
			self.weights2 += d_weights2
			self.weights3 += d_weights3
			self.bias1 += d_bias1
			self.bias2 += d_bias2
			self.bias3 += d_bias3

			if epoch % 10 == 0:
				loss = np.mean(np.abs(output_error))
				print(f'Epoch {epoch}/{epochs}, Loss: {loss}')

	def predict(self, X):
		hidden1 = self.leaky_relu(np.dot(X, self.weights1) + self.bias1)
		hidden1 = self.batch_normalization(hidden1)
		hidden2 = self.leaky_relu(np.dot(hidden1, self.weights2) + self.bias2)
		hidden2 = self.batch_normalization(hidden2)
		output = self.sigmoid(np.dot(hidden2, self.weights3) + self.bias3)
		return np.argmax(output, axis=1)

# Hàm tải và tiền xử lý dữ liệu
def load_data(data_dir, categories, image_size):
	data = []
	labels = []
	for label, category in enumerate(categories):
		category_path = os.path.join(data_dir, category)
		for img_name in os.listdir(category_path):
			img_path = os.path.join(category_path, img_name)
			try:
				img = load_img(img_path, target_size=(image_size, image_size))
				img_array = img_to_array(img) / 255.0
				data.append(img_array.flatten())
				labels.append(label)
			except Exception as e:
				print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
	return np.array(data), np.array(labels)

# Hàm xử lý ảnh bên ngoài
def preprocess_image(image_path, image_size):
	img = load_img(image_path, target_size=(image_size, image_size))
	img_array = img_to_array(img) / 255.0
	return np.array([img_array.flatten()])

# Đường dẫn tập dữ liệu
train_data_dir = "path_to_dataset\\train"
test_data_dir = "path_to_dataset\\test"
categories = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]
image_size = 64

# Tải dữ liệu
train_data, train_labels = load_data(train_data_dir, categories, image_size)
test_data, test_labels = load_data(test_data_dir, categories, image_size)

# Chia dữ liệu train/validation
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
def z_score_normalization(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	return (data - mean) / (std + 1e-7)

X_train = z_score_normalization(X_train)
X_val = z_score_normalization(X_val)
test_data = z_score_normalization(test_data)

# Khởi tạo và huấn luyện MLP
input_size = image_size * image_size * 3
hidden_size = 128
output_size = len(categories)

mlp = MLP(input_size, hidden_size, output_size)
mlp.fit(X_train, y_train, epochs=100)

# Đánh giá trên tập validation
val_predictions = mlp.predict(X_val)
val_accuracy = np.mean(val_predictions == y_val)
print(f"Độ chính xác trên tập validation: {val_accuracy * 100:.2f}%")

# Đánh giá trên tập test
test_predictions = mlp.predict(test_data)
test_accuracy = np.mean(test_predictions == test_labels)
print(f"Độ chính xác trên tập test: {test_accuracy * 100:.2f}%")

# Dự đoán ảnh bên ngoài
external_image_path = "c_test.png"
external_image = preprocess_image(external_image_path, image_size)

predicted_class_index = mlp.predict(external_image)[0]
predicted_class = categories[predicted_class_index]
print(f"Ảnh được dự đoán là: {predicted_class}")

