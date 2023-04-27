import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from d2l import torch as d2l
import torch
import numpy as np
from multiprocessing.spawn import freeze_support
from random import sample



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CIFAR10OneHot(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform,
                                                    target_transform=target_transform, download=download)
        self.one_hot = np.eye(10)  # Create a 10x10 identity matrix for one-hot encoding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        one_hot_target = self.one_hot[target]
        return img, one_hot_target



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

    # Load the CIFAR-10 dataset
    full_dataset = CIFAR10OneHot(root='./data', train=True, download=True, transform=transform)
    data_size = len(full_dataset)
    fold_indices = get_k_fold_indices(data_size, k)

    for i in range(k):
        print(f"Fold {i + 1}:")

        val_indices = fold_indices[i]
        train_indices = [j for j in range(data_size) if j not in val_indices]

        train_data = torch.utils.data.Subset(full_dataset, train_indices)
        val_data = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=True, num_workers=4)

        model.apply(init_weights_he)

        # Initialize a new optimizer and set the trainer's optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
        trainer.optimizer = optimizer

        train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(model, train_loader, val_loader)

        all_train_losses.append(train_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)

        print("\n")
    return all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(dropout_rate)

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
        out = self.dropout(out)

        # Second convolutional layer
        out = self.bn2(self.conv2(out))

        # Shortcut connection
        out += self.shortcut(x)

        # ReLU activation
        out = F.relu(out)

        return out



class ResNet18(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0):  # Add dropout_rate parameter
        super(ResNet18, self).__init__()

        self.in_planes = 64

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)


        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout_rate=dropout_rate)

        # Final fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):  # Add dropout_rate parameter
        # Create a list of stride values for each BasicBlock in this layer
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            # Create a BasicBlock with the specified number of input and output channels,
            # and the specified stride value
            layers.append(BasicBlock(self.in_planes, planes, stride, dropout_rate=dropout_rate))  # Pass dropout_rate to BasicBlock
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
            for X, y_one_hot in loader:
                X, y_one_hot = X.to(self.device), y_one_hot.to(self.device)
                y_hat = model(X)
                y = torch.argmax(y_one_hot, dim=1)  # Convert one-hot labels back to class indices
                l = self.loss(y_hat, y)
                total_loss += l.item()

                # Calculate accuracy
                pred_indices = torch.argmax(y_hat, dim=1)
                true_indices = y
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
            for X, y_one_hot in train_loader:
                y = torch.argmax(y_one_hot, dim=1)  # Convert one-hot labels back to class indices
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = model(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                train_loss += l.item()

                # Calculate accuracy
                pred_indices = torch.argmax(y_hat, dim=1)
                true_indices = y
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

        plt.savefig("metrics_plot_18_cifar.png")
        plt.show()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(num_classes=10, dropout_rate=0)
    model = model.to(device)

    # Save the initial model's state_dict for later use
    torch.save(model.state_dict(), 'resnet18_ex4_initial.pth')

    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = CustomTrainer(optimizer=None, loss_fn=loss_fn, device=device, max_epochs=100)

    train_losses, train_accuracies, val_losses, val_accuracies = k_fold_cross_validation(2, model)

    # Update trainer's attributes with the average values from k-fold cross-validation
    trainer.train_losses = [np.mean(epoch_losses) for epoch_losses in zip(*train_losses)]
    trainer.train_accuracies = [np.mean(epoch_accuracies) for epoch_accuracies in zip(*train_accuracies)]
    trainer.val_losses = [np.mean(epoch_losses) for epoch_losses in zip(*val_losses)]
    trainer.val_accuracies = [np.mean(epoch_accuracies) for epoch_accuracies in zip(*val_accuracies)]

    trainer.plot_metrics()


    

