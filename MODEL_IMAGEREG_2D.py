import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torchvision.transforms.functional as FF
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import ants
import scipy.spatial
from monai.losses import GlobalMutualInformationLoss

# Set device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset directory
DATASET_DIR = 'H:/Meu Drive/MESTRADO/MODELO/TESTE/'
os.makedirs(DATASET_DIR, exist_ok=True)

# Define Z-Score normalization transform
class ZScoreNormalize:
    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        return (img - mean) / std

# Define custom random affine transformation with independent probabilities
class RandomAffineWithIndependentProbabilities:
    def __init__(self, rotate_prob, translate_prob, scale_prob, rotate_range, translate_range, scale_range):
        self.rotate_prob = rotate_prob
        self.translate_prob = translate_prob
        self.scale_prob = scale_prob
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() < self.rotate_prob:
            angle = random.uniform(*self.rotate_range)
            img = FF.rotate(img, angle)

        if random.random() < self.translate_prob:
            translate_x = random.randint(-self.translate_range[0], self.translate_range[0])
            translate_y = random.randint(-self.translate_range[1], self.translate_range[1])
            img = FF.affine(img, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)

        if random.random() < self.scale_prob:
            scale_factor = random.uniform(*self.scale_range)
            img = FF.affine(img, angle=0, translate=(0, 0), scale=scale_factor, shear=0)

        return img

# Define optional image augmentations
augment = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3))], p=0.1),
])

# Define transforms for resizing and normalization
transform = T.Compose([
   T.Resize((112, 112), interpolation=InterpolationMode.NEAREST), 
   ZScoreNormalize(),
])

resize = T.Compose([
    T.Resize((112, 112), interpolation=InterpolationMode.NEAREST),
])  # the resize transform is for masks

# Define custom dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, adc_paths, dwi_paths, mask_paths, transform=None, augment=None, resize=None):
        self.adc_paths = adc_paths
        self.dwi_paths = dwi_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment  
        self.resize = resize

        # Initialize random affine transformation
        self.rand_affine = RandomAffineWithIndependentProbabilities(
            rotate_prob=0.70, 
            translate_prob=0.70,
            scale_prob=0.70,
            rotate_range=(-10, 10),  
            translate_range=(6, 6),  
            scale_range=(0.98, 1.02) 
        )

        self.cache = {} 

    def __len__(self):
        return len(self.adc_paths) * 16

    def __getitem__(self, idx):
        patient_idx = idx // 16
        slice_offset = idx % 16

        if idx in self.cache:
            return self.cache[idx]

        # Load NIfTI images
        adc_image = nib.load(self.adc_paths[patient_idx]).get_fdata().astype(np.float32)
        dwi_image = nib.load(self.dwi_paths[patient_idx]).get_fdata().astype(np.float32)
        mask_image = nib.load(self.mask_paths[patient_idx]).get_fdata().astype(np.float32)

        # Extract 2D slices (16 slices from the middle of the 3D volume)
        mid_slice = adc_image.shape[2] // 2
        start_slice = mid_slice - 8
        adc_slice = adc_image[:, :, start_slice + slice_offset]
        dwi_slice = dwi_image[:, :, start_slice + slice_offset]
        mask_slice = mask_image[:, :, start_slice + slice_offset]

        adc_slice = np.expand_dims(adc_slice, axis=0) 
        dwi_slice = np.expand_dims(dwi_slice, axis=0)  
        mask_slice = np.expand_dims(mask_slice, axis=0)

        # Convert slices to PIL Images
        adc_slice_pil = Image.fromarray(adc_slice[0])  
        dwi_slice_pil = Image.fromarray(dwi_slice[0])  
        mask_slice_pil = Image.fromarray(mask_slice[0])

        ground_truth = T.ToTensor()(adc_slice_pil)
        mask_ground = T.ToTensor()(mask_slice_pil)

        # Apply random affine transformations
        angle = random.uniform(*self.rand_affine.rotate_range) if random.random() < self.rand_affine.rotate_prob else 0
        translate_x = random.randint(-self.rand_affine.translate_range[0], self.rand_affine.translate_range[0]) if random.random() < self.rand_affine.translate_prob else 0
        translate_y = random.randint(-self.rand_affine.translate_range[1], self.rand_affine.translate_range[1]) if random.random() < self.rand_affine.translate_prob else 0
        scale = random.uniform(*self.rand_affine.scale_range) if random.random() < self.rand_affine.scale_prob else 1.0

        adc_slice_pil = FF.affine(adc_slice_pil, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=0)
        mask_slice_pil = FF.affine(mask_slice_pil, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=0)

        # Convert to tensors
        adc_slice = T.ToTensor()(adc_slice_pil)
        dwi_slice = T.ToTensor()(dwi_slice_pil)
        mask_slice = T.ToTensor()(mask_slice_pil)

        if self.augment:
             adc_slice = self.augment(adc_slice)
        if self.transform:
             adc_slice = self.transform(adc_slice)
             dwi_slice = self.transform(dwi_slice)
             ground_truth = self.transform(ground_truth)
             mask_ground = self.resize(mask_ground)
             mask_slice = self.resize(mask_slice)

        # Store in cache and return
        self.cache[idx] = {'fixed': dwi_slice, 'moving': adc_slice, 'ground_truth': ground_truth, 'mask': mask_slice, 'mask_ground': mask_ground}
        return self.cache[idx]

# Split dataset into training, validation and testing sets
def create_datasets(adc_paths, dwi_paths, mask_paths, transform=None, augment=None, resize=None):
    dataset = MedicalImageDataset(adc_paths, dwi_paths, mask_paths, transform=transform, augment=augment, resize=resize)
    length = len(dataset)
    train_size = int(0.7 * length)   
    val_size = int(0.2 * length)   
    test_size = length - train_size - val_size 
    return random_split(dataset, [train_size, val_size, test_size])

# Define paths for all images
adc_paths = [f"C:/Users/danny/OneDrive/Área de Trabalho/ESTAGIO/dataset/ISLES-2022/ISLES-2022/sub-strokecase{i:04d}/ses-0001/dwi/sub-strokecase{i:04d}_ses-0001_adc.nii.gz" for i in range(1, 251)]
dwi_paths = [f"C:/Users/danny/OneDrive/Área de Trabalho/ESTAGIO/dataset/ISLES-2022/ISLES-2022/sub-strokecase{i:04d}/ses-0001/dwi/sub-strokecase{i:04d}_ses-0001_dwi.nii.gz" for i in range(1, 251)]
mask_paths= [f"C:/Users/danny/OneDrive/Área de Trabalho/ESTAGIO/dataset/ISLES-2022/ISLES-2022/derivatives/sub-strokecase{i:04d}/ses-0001/sub-strokecase{i:04d}_ses-0001_msk.nii.gz" for i in range(1, 251)]

# Create datasets
train_dataset, val_dataset, test_dataset = create_datasets(adc_paths, dwi_paths, mask_paths, transform=transform, augment=augment, resize=resize)

# Save datasets
torch.save(train_dataset, os.path.join(DATASET_DIR, "train_dataset2.pt"))
torch.save(test_dataset, os.path.join(DATASET_DIR, "test_dataset2.pt"))
torch.save(val_dataset, os.path.join(DATASET_DIR, "val_dataset2.pt"))

# Define CNN model for predicting affine transformation
class MODELCNN(nn.Module):
    def __init__(self):
        super(MODELCNN, self).__init__()
        
        # Feature extractor (input = 2 channels: fixed + moving)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Calculate flatten size
        dummy_input = torch.zeros(1, 2, 112, 112)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        # Regression head to predict affine parameters (6 values)
        self.regression_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )

    def forward(self, fixed, moving):
        combined = torch.cat((fixed, moving), dim=1)
        features = self.feature_extractor(combined)
        transformation_params = self.regression_net(features)
        theta = transformation_params.view(-1, 2, 3)

        # Apply affine transformation
        grid = F.affine_grid(theta, fixed.size(), align_corners=False)
        warped = F.grid_sample(moving, grid, align_corners=False)
        warped = warped[:, 0:1, :, :]  # Keep single channel

        return warped, theta  #save theta for future use 

# Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        fixed = batch['fixed'].to(device)
        moving = batch['moving'].to(device)
        optimizer.zero_grad()
        warped, _ = model(fixed, moving)
        loss = criterion(fixed, warped)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * fixed.size(0)
    return running_loss / len(dataloader.dataset)

# Validation loop
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            fixed = batch['fixed'].to(device)
            moving = batch['moving'].to(device)
            warped, _ = model(fixed, moving)
            loss = criterion(fixed, warped)
            val_loss += loss.item() * fixed.size(0)
    return val_loss / len(dataloader.dataset)

# Hyperparameters
batch_size = 16
initial_lr = 0.001
num_epochs = 1000

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model, optimizer, loss, scheduler
model = MODELCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.001, amsgrad=True)
criterion = GlobalMutualInformationLoss(num_bins=64).to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.90, patience=20, verbose=True)

# Early stopping setup
early_stopping_patience = 50
best_val_loss = float('inf')
patience_counter = 0

# Training loop with early stopping
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Save model weights
torch.save(model.state_dict(), os.path.join(DATASET_DIR, 'FINALMODEL3.pth'))

# Plot loss curve
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATASET_DIR, 'LOSS_FINAL.png'))
