# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
A convolutional autoencoder for image denoising learns to compress images into a lower-dimensional representation and then reconstructs clean images from noisy inputs. It consists of encoder layers that extract features, a bottleneck that captures essential information, and decoder layers that reconstruct the denoised image. The model is trained to minimize the difference between the output and the clean target images. This approach effectively removes noise while preserving important details in the images.

## DESIGN STEPS
### STEP 1: 
Import Libraries and Load Dataset – Import required deep learning libraries and load the image dataset.
### STEP 2: 
Preprocess Images – Normalize the images and reshape them into suitable input format.
### STEP 3: 
Add Noise to Images – Introduce random noise to the original images to create noisy input data.
### STEP 4: 
Construct the Autoencoder – Build the convolutional autoencoder model with encoder and decoder layers.
### STEP 5: 
Train the Model – Train the model using noisy images as input and original images as target output.
### STEP 6: 
Reconstruct and Evaluate – Use the trained model to denoise images and compare the reconstructed images with original images.

## PROGRAM

### Name:  Monika A

### Register Number:  212224240094

```python
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary




# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:Monika A")
    print("Register Number:212224240094 ")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:Monika A  ")
    print("Register Number: 212224240094 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary

![image](https://github.com/Monikaayyalusamy/DL-Autoencoder/blob/main/Screenshot%202026-03-10%20114704.png)

### Training loss
![image](https://github.com/Monikaayyalusamy/DL-Autoencoder/blob/main/Screenshot%202026-03-10%20114714.png)

## Original vs Noisy Vs Reconstructed Image
![image](https://github.com/Monikaayyalusamy/DL-Autoencoder/blob/main/Screenshot%202026-03-10%20114724.png)

## RESULT
The model successfully removed noise from images and produced clear denoised outputs.
