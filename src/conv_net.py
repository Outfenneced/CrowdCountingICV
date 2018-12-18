import os
import re
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


LEARNING_RATE = 0.001


class CCDataset(Dataset):
    def __init__(self, root, type=""):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.full_path = os.path.join(root, type)
        dir_info = os.listdir(self.full_path)
        self.data_length = len(dir_info)

    def __getitem__(self, index):
        file_path = os.path.join(self.full_path, "{}.npz".format(index))
        file = np.load(file_path)
        image_grey = file["image"]
        count = file["label"].astype(dtype=np.float32)
        return self.transform(np.atleast_3d(image_grey)), torch.from_numpy(count[0])

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


CLASSIFIER_FORMAT = "classifier{epoch}.ckpt"


def train(data_path, classifier_out, gpu=0, epochs=1, batch_size=100, load_threading=8):
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    start_time = datetime.now()
    print("Start time: ", start_time)

    cnn = CNN()
    cnn.to(device)
    print(cnn)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train_data = CCDataset(data_path, type="Train")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=load_threading, shuffle=True)

    num_steps = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch, epochs, i, num_steps, loss.item()))
        if epoch % 10 == 0:
            cls_file = os.path.join(classifier_out, CLASSIFIER_FORMAT.format(epoch=epoch))
            print("Saving epoch {epoch} to {filename}".format(epoch=epoch, filename=cls_file))
            torch.save(cnn.state_dict(), cls_file)
    end_time = datetime.now()
    print("Start time: ", start_time)
    print("End time: ", end_time)
    print("Duration: ", end_time - start_time)
    cls_file = os.path.join(classifier_out, CLASSIFIER_FORMAT.format(epoch="DONE"))
    torch.save(cnn.state_dict(), cls_file)


def test(data_dir, cnn_dir, test_type="Test", gpu=5, batch_size=100, load_threading=12):
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    cnn_loss = list()
    cnn_names = os.listdir(cnn_dir)
    for cnn_name in cnn_names:
        cnn_number = cnn_name.replace("classifier", "").replace(".ckpt", "")

        test_data = CCDataset(data_dir, type=test_type)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=load_threading, shuffle=False)

        cnn_path = os.path.join(cnn_dir, cnn_name)
        cnn = CNN()
        cnn.to(device)
        cnn.load_state_dict(torch.load(cnn_path))
        cnn.eval()

        loss_function = nn.MSELoss()

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
            loss = loss_function(outputs, correct_vals).item()
            print(outputs)
            print(correct_vals)
            print(cnn_number, " MSE loss: ", loss)
        cnn_loss.append((cnn_number, loss))
    return cnn_loss
