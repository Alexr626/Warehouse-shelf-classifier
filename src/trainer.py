"""
This code is mostly inspired from the PyTorch Lightning API, and is included 
only to give you an idea of the main steps you need to implement.
Feel free to modify the code as you see fit. 
"""

import numpy as np
import torch
import torch.nn as nn
from src.models.ResNetBlock import ResNetBlock
from src.models.layers.InputLayer import InputLayer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import constants
import time
import psutil

class Trainer:
    def __init__(self, model, data_dir, labels_file, batch_size = 32, learning_rate=0.001):
        self.model = model
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_file)
        self.labels_file = labels_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader = self.setup_dataloader()

    def setup_dataloader(self):
        dataset = CustomDataset(self.data_dir, self.labels_file)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)


    def training_step(self, input_batch, labels_batch, *args, **kwargs):
        # TODO: Write the train_mode step for one single batch through your model
        # Your code might contain a forward pass, a backward pass and anything
        # else that is required.
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(input_batch)
        loss = self.criterion(outputs, labels_batch)

        # Backward pass and parameter update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, input_batch, labels_batch, *args, **kwargs):
        # TODO: Write the validation step for one single batch through your model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_batch)
            loss = self.criterion(outputs, labels_batch)

            _, preds = torch.max(outputs, 1)
            accuracy = (preds == labels_batch).float().mean().item()

        return loss.item(), accuracy

    def train(self, num_epochs: int=1, batch_size: int = 32, *args, **kwargs):
        # TODO: Train a single epoch
        # TODO: Print out the mean train_mode loss and train_mode accuracy at
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            for batch_idx, (input_batch, labels_batch) in enumerate(self.train_loader):
                loss = self.training_step(input_batch, labels_batch)
                epoch_loss += loss

                if batch_idx * 10 == 0:
                    print(f"Batch {batch_idx + 1} - Loss: {loss:.4f}")
                    self.log_resource_usage()

            mean_loss = epoch_loss / len(self.data)
            print(f"Epoch {epoch}/{num_epochs}, Mean Training Loss: {mean_loss:.4f}")

            # TODO: Validate the model at each epoch
            # TODO: Print out the mean validation accuracy at the end of each epoch
            val_loss, val_accuracy = self.validate(batch_size)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    def validate(self, batch_size: int = 32, *args, **kwargs):
        # TODO: Run inference over the entire validation set
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                input_batch = self.data[i:i+batch_size]
                labels_batch = self.labels[i:i+batch_size]

                # Single validation step
                loss, accuracy = self.validation_step(input_batch, labels_batch)
                val_loss += loss
                val_accuracy += accuracy

        mean_val_loss = val_loss / (len(self.data) // batch_size)
        mean_val_accuracy = val_accuracy / (len(self.data) // batch_size)
        return mean_val_loss, mean_val_accuracy

    def log_resource_usage(self):
        # Log CPU, RAM, and GPU usage (if available)
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"RAM Usage: {psutil.virtual_memory().percent}%")
        if torch.cuda.is_available():
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        self.label_mapping = {"empty": 0, "filled": 1}

        # Gather paths of all .npy files in the directory
        self.file_paths = []
        self.file_labels = []

        # Populate file paths and labels based on file matching
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

                    # Extract the base name to get corresponding label
                    base_name = os.path.splitext(file)[0].split('_')[0]
                    label = self.labels_df[self.labels_df['image_path'].str.contains(base_name)]['class'].values[0]
                    self.file_labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the numpy array file
        image = np.load(self.file_paths[idx])
        label_str = self.file_labels[idx]
        label = self.label_mapping[label_str]

        if self.transform:
            image = self.transform(image)

        # Convert image to tensor and adjust shape for PyTorch
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def main():
    # Directory and file paths
    data_dir = os.path.join(constants.ROOT_DIR, "training_data", "preprocessed_arrays")
    labels_file = os.path.join(constants.ROOT_DIR, "training_data", "labels.csv")

    # Model parameters
    in_channels = 3  # Assuming RGB input
    out_channels = 64  # Example output channels for the first ResNet block
    kernel_size = 3
    stride = 1
    padding = 1
    num_epochs = 10
    batch_size = 32

    # Instantiate a ResNet block and the trainer
    model = ResNetBlock(in_channels, out_channels, kernel_size, stride, padding)
    trainer = Trainer(model=model, data_dir=data_dir, labels_file=labels_file, batch_size=32)

    # Run train_mode
    trainer.train(num_epochs=num_epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()