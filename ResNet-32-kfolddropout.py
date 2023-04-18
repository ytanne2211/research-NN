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
from random import sample




df = pd.read_csv("./labels.csv")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.Resize((96, 96)),
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

# def split_data(test_size=0.2):
#     data_size = len(file_path_label_pairs)
#     test_size = int(test_size * data_size)
#     test_indices = sample(range(data_size), test_size)
#     train_indices = [i for i in range(data_size) if i not in test_indices]

#     train_data = [file_path_label_pairs[i] for i in train_indices]
#     val_data = [file_path_label_pairs[i] for i in test_indices]

#     train_dataset = ImageDataset(train_data, transform=transform)
#     val_dataset = ImageDataset(val_data, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

#     return train_loader, val_loader

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_k_fold_indices(data_size, k):
    fold_size = data_size // k
    indices = list(range(data_size))
    np.random.shuffle(indices)
    return [indices[i * fold_size: (i + 1) * fold_size] for i in range(k)]

def k_fold_cross_validation(k, model):
    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []
    data_size = len(file_path_label_pairs)
    fold_indices = get_k_fold_indices(data_size, k)

    for i in range(k):
        print(f"Fold {i + 1}:")

        val_indices = fold_indices[i]
        train_indices = [j for j in range(data_size) if j not in val_indices]

        train_data = [file_path_label_pairs[j] for j in train_indices]
        val_data = [file_path_label_pairs[j] for j in val_indices]

        train_dataset = ImageDataset(train_data, transform=transform)
        val_dataset = ImageDataset(val_data, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

        
        model.apply(init_weights_he)

        # Initialize a new optimizer and set the trainer's optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        trainer.optimizer = optimizer

        train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(model, train_loader, val_loader)

        all_train_losses.append(train_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)

        print("\n")
    return all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies


# def train_and_evaluate(trainer, model):
#     train_loader, val_loader = split_data(test_size=0.2)
#     trainer.fit(model, train_loader, val_loader)

#     # Plot Training and Validation Loss
#     plt.figure(figsize=(10, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(trainer.train_losses, label="Training Loss")
#     plt.plot(trainer.val_losses, label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()

#     # Plot Training and Validation Accuracy
#     plt.subplot(2, 1, 2)
#     plt.plot(trainer.train_accuracies, label="Training Accuracy")
#     plt.plot(trainer.val_accuracies, label="Validation Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.savefig()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.2):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        # Add a dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # If the input number of channels or the stride is different from the output number of channels,
        # then we need to perform a 1x1 convolution to change the number of channels or downsample the input
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # First convolutional layer
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))

        # Second convolutional layer
        out = self.bn2(self.conv2(out))

        # Shortcut connection
        out += self.shortcut(x)

        # ReLU activation
        out = F.relu(out)

        return out

class ResNet32(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.2):
        super(ResNet32, self).__init__()

        self.in_planes = 64

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks with dropout
        self.layer1 = self._make_layer(64, 5, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(128, 5, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(256, 5, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(512, 5, stride=2, dropout_rate=dropout_rate)

        # Final fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):
        # Create a list of stride values for each BasicBlock in this layer
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            # Create a BasicBlock with the specified number of input and output channels,
            # the specified stride value, and the specified dropout rate
            layers.append(BasicBlock(self.in_planes, planes, stride, dropout_rate=dropout_rate))
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
        out = F.adaptive_avg_pool2d(out, (1, 1))

        # Flatten
        out = out.view(out.size(0), -1)

        # Fully connected layer
        out = self.linear(out)

        return out



class CustomTrainer(d2l.Trainer):
    def __init__(self, optimizer, loss_fn, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.loss = loss_fn
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def evaluate_loss_and_accuracy(self, loader):
        model.eval()
        total_loss = 0
        total_correct = 0
        num_batches = 0
        num_samples = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = model(X)
                l = self.loss(y_hat, y)
                total_loss += l.item()
                
                # Calculate accuracy
                pred_indices = torch.argmax(y_hat, dim=1)
                true_indices = torch.argmax(y, dim=1)
                total_correct += (pred_indices == true_indices).sum().item()
                num_samples += y.size(0)
                
                num_batches += 1
        return total_loss / num_batches, total_correct / num_samples

    def fit(self, model, train_loader, val_loader):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            num_batches = 0
            num_samples = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = model(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                train_loss += l.item()

                # Calculate accuracy
                pred_indices = torch.argmax(y_hat, dim=1)
                true_indices = torch.argmax(y, dim=1)
                train_correct += (pred_indices == true_indices).sum().item()
                num_samples += y.size(0)

                num_batches += 1
                print(f"Batch {num_batches}, Training loss: {l.item():.4f}")
            avg_train_loss = train_loss / num_batches
            train_accuracy = train_correct / num_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            val_loss, val_accuracy = self.evaluate_loss_and_accuracy(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch + 1}, Average Training loss: {avg_train_loss:.4f}, Training accuracy: {train_accuracy:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

        return train_losses, train_accuracies, val_losses, val_accuracies


    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.train_accuracies, label="Training Accuracy")
        plt.plot(self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("metrics_plot.png")
        plt.show()


if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet32(num_classes=8, dropout_rate=0.2)
    model = model.to(device)

    # Save the initial model's state_dict for later use
    torch.save(model.state_dict(), 'resnet32_initial.pth')

    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = CustomTrainer(optimizer=None, loss_fn=loss_fn, device=device, max_epochs=100)

    train_losses, train_accuracies, val_losses, val_accuracies = k_fold_cross_validation(3, model)

    # Update trainer's attributes with the average values from k-fold cross-validation
    trainer.train_losses = [np.mean(epoch_losses) for epoch_losses in zip(*train_losses)]
    trainer.train_accuracies = [np.mean(epoch_accuracies) for epoch_accuracies in zip(*train_accuracies)]
    trainer.val_losses = [np.mean(epoch_losses) for epoch_losses in zip(*val_losses)]
    trainer.val_accuracies = [np.mean(epoch_accuracies) for epoch_accuracies in zip(*val_accuracies)]

    trainer.plot_metrics()


    
