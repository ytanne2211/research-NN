import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import random
from PIL import Image
from matplotlib import pyplot as plt
from d2l import torch as d2l
import torch
import numpy as np
from multiprocessing.spawn import freeze_support
from sklearn.model_selection import train_test_split

df = pd.read_csv("./labels.csv")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

label_to_index = {
    "happy": 0,
    "surprise": 1,
    "anger": 2,
    "disgust": 3,
    "fear": 4,
    "sad": 5,
    "neutral": 6,
    "contempt": 7,
}


class ImageDataset(Dataset):
    def __init__(self, file_path_label_pairs, transform=None):
        self.file_path_label_pairs = file_path_label_pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.file_path_label_pairs)
    
    def __getitem__(self, idx):
        img_path, label = self.file_path_label_pairs[idx]
        with Image.open(img_path).convert("RGB") as img:  # Use 'with' statement to open and close the image file
            if self.transform:
                img = self.transform(img)
        label_index = label_to_index[label]  # Convert the label string to an integer index
        label_one_hot = np.zeros(8)  # Create an array of zeros with size 8 (number of classes)
        label_one_hot[label_index] = 1  # Set the position corresponding to the label index to 1
        label_tensor = torch.tensor(label_one_hot, dtype=torch.float)  # Convert the one-hot vector to a tensor
        return img, label_tensor




# Create a list of all image file paths and labels
file_paths = df['pth'].tolist()
labels = df['label'].tolist()

# Combine the file paths and labels into a list of tuples
file_path_label_pairs = list(zip(file_paths, labels))

# Shuffle the list of file path and label pairs
random.shuffle(file_path_label_pairs)

def split_data(test_size=0.2):
    train_data, val_data = train_test_split(file_path_label_pairs, test_size=test_size, random_state=42)

    train_dataset = ImageDataset(train_data, transform=transform)
    val_dataset = ImageDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    return train_loader, val_loader

def train_and_evaluate(trainer, model):
    train_loader, val_loader = split_data(test_size=0.2)
    trainer.fit(model, train_loader, val_loader)
    plt.plot(trainer.train_losses, label="Training Loss")
    plt.plot(trainer.val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        # If the input number of channels or the stride is different from the output number of channels,
        # then we need to perform a 1x1 convolution to change the number of channels or downsample the input
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # First convolutional layer
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolutional layer
        out = self.bn2(self.conv2(out))

        # Shortcut connection
        out += self.shortcut(x)

        # ReLU activation
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=8):  # Change the default value of num_classes to 8
        super(ResNet18, self).__init__()

        self.in_planes = 64

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Final fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        # Create a list of stride values for each BasicBlock in this layer
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            # Create a BasicBlock with the specified number of input and output channels,
            # and the specified stride value
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion

        # Return a Sequential container that contains all the BasicBlocks in this layer
        return nn.Sequential(*layers)

    def forward(self, x):
    # First convolutional layer
        out = F.relu(self.bn1(self.conv1(x)))

    # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Average pooling
        out = F.avg_pool2d(out, 4)

        # Flatten
        out = out.view(out.size(0), -1)

        # Fully connected layer
        out = self.linear(out)

        return out  



class CustomTrainer(d2l.Trainer):
    def __init__(self, optimizer, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.loss = loss_fn
        self.train_losses = []
        self.val_losses = []
    
    def evaluate_loss(self, val_loader):
        model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for X, y in val_loader:
                y_hat = model(X)
                l = self.loss(y_hat, y)
                total_loss += l.item()
                num_batches += 1
        return total_loss / num_batches

    def fit(self, model, train_loader, val_loader):
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for X, y in train_loader:
                #X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = model(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                train_loss += l.item()
                num_batches += 1
                print(f"Batch {num_batches}, Training loss: {l.item():.4f}")
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            val_loss = self.evaluate_loss(val_loader)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}, Average Training loss: {avg_train_loss:.4f}, Validation loss: {val_loss:.4f}")


if __name__ == '__main__':
    freeze_support()
    model = ResNet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    trainer = CustomTrainer(optimizer=optimizer, loss_fn=loss_fn, max_epochs=5)
    train_and_evaluate(trainer, model)
