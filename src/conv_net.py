import os
from datetime import datetime

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import scipy.io as sio
import skimage

start_time = datetime.now()
print("Start time: ", start_time)
data_path = input("What is the data path? ")
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
IMAGE_FILE_FORMAT = "img_{:04}.jpg"
MAT_FILE_FORMAT = "img_{:04}_ann.mat"


def generate_marks_map(marks, shape):
	marks = marks.round()
	marks = marks.astype(dtype=np.uint16)
	ys, xs = marks.T
	mapping = np.zeros(shape, dtype=np.uint8)
	mapping[xs, ys] = 255
	return mapping


class CCDataset(Dataset):
	def __init__(self, root, train=True):
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

		self.full_path = os.path.join(root, "train" if train else "test")
		dir_info = os.listdir(self.full_path)
		self.data_length = len(dir_info) // 2

	def __getitem__(self, index):
		image_path = os.path.join(self.full_path, IMAGE_FILE_FORMAT.format(index + 1))
		image = cv2.imread(image_path)
		image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		marks_path = os.path.join(self.full_path, MAT_FILE_FORMAT.format(index + 1))
		marks = sio.loadmat(marks_path)['annPoints']
		count = len(marks)
		# TODO: Subdivide, for now just resizing image
		# Related: marks_map = generate_marks_map(marks, image_grey.shape)
		return self.transform(np.atleast_3d(image_grey)), torch.from_numpy(np.asarray(count, dtype=np.float32).reshape(1, 1))

	def __len__(self):
		return self.data_length


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 12, kernel_size=7, stride=2, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.fc1 = nn.Linear(6 * 6 * 36, 6 * 6 * 36)
		self.fc2 = nn.Linear(6 * 6 * 36, 6 * 6 * 36)
		self.o = nn.Linear(6 * 6 * 36, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.reshape(x.size(0), -1)
		x = self.fc1(x)
		x = self.fc2(x)
		output = self.o(x)
		return output


cnn = CNN()
cnn.to(device)
print(cnn)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

train_data = CCDataset(data_path, train=True)
test_data = CCDataset(data_path, train=False)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

num_steps = len(train_loader)
for epoch in range(EPOCHS):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		outputs = cnn(images)
		loss = loss_function(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
		      .format(epoch + 1, EPOCHS, i + 1, num_steps, loss.item()))

cnn.eval()
with torch.no_grad():
	outputs = np.array(0, dtype=np.float32)
	correct_vals = np.array(0, dtype=np.float32)
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = np.append(outputs, cnn(images))
		correct_vals = np.append(correct_vals, labels)

	outputs = torch.from_numpy(outputs)
	correct_vals = torch.from_numpy(correct_vals)
	print("Final MSE loss: ", loss_function(outputs, correct_vals))
	print(outputs)
	print(correct_vals)


torch.save(cnn.state_dict(), "model.ckpt")

end_time = datetime.now()
print("Start time: ", start_time)
print("End time: ", end_time)
print("Duration: ", end_time - start_time)
