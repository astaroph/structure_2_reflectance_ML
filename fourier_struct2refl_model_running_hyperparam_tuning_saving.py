import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rotate, InterpolationMode
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import *
import os
import torch.nn.functional as F
from time import time
from optparse import OptionParser
import sys
from sklearn.pipeline import make_pipeline
from itertools import product
from time import localtime
###Loading in the structural SEM and reflectance data file pairings
import pandas as pd
import numpy as np
from PIL import Image

start=time()
usage = "usage: %prog [options] arg1 arg2"
parser = OptionParser(usage=usage)
parser.add_option("-f", "--filepath", dest='filepath', type="string",
                 help="file path to csv showing pairs of SEM images and 10channel reflectance values")
parser.add_option("-b", "--batch_size", dest='batch_size', type="int",
                 help="batch size for training the regression algorithm")
parser.add_option("-l", "--lr", dest='lr', type="float",
                 help="the learning rate for training the regression algorithm")
parser.add_option("-e", "--epochs", dest='epochs', type="int",
                 help="the number of epochs over which to train the algorithm")
parser.add_option("-w", "--weight_decay", dest='weight_decay', type="float",
                 help="the weight decay for the adam optimizer")
parser.add_option("-s", "--save_path", dest='save_path', type="string",
                 help="the directory path for saving the best performing machine learning models and training output. Must end with a forward slash")
parser.add_option("-p", "--prefix", dest='prefix', type="string",
                 help="the descriptive file prefix to append to the beginning of saved models")
(options, args) = parser.parse_args()


###Inputs
filepath=options.filepath
batch_size=options.batch_size
lr=options.lr
epochs=options.epochs
weight_decay=options.weight_decay
save_path=options.save_path
prefix=options.prefix
###Inputs
# filepath="./climate_change_solution_structural_test_folder/climate_change_solution_structural_image_reflectancevalues_dataset_updatedstructural.csv"
# batch_size=6
# lr=0.001
# epochs=10
# weight_decay=0.000001
# save_path='/n/holyscratch01/pierce_lab/astaroph/WW_machine_learning/CC_scales/machine_learning_models/structure_to_reflectance/'


# df_pairs2=pd.read_csv("./climate_change_solution_structural_test_folder/climate_change_solution_structural_image_reflectancevalues_dataset.csv")
# df_pairs2=pd.read_csv("./climate_change_solution_structural_test_folder/climate_change_solution_structural_image_reflectancevalues_dataset_updatedstructural.csv")
df_pairs2=pd.read_csv(filepath)

##Splitting the data into training and testing datasets at random
import pandas as pd
from sklearn.model_selection import train_test_split

# # Define the fraction of data to be used for testing (10% in this example)
test_fraction = 0.15
# # Split the data into training and testing sets
train_data, test_data = train_test_split(df_pairs2, test_size=test_fraction, random_state=42)

ref_train_list=[list(train_data.iloc[x,1:11]) for x in range(0,len(train_data))]
SEM_train_list=list(train_data.SEM_filenames)
ref_test_list=[list(test_data.iloc[x,1:11]) for x in range(0,len(test_data))]
SEM_test_list=list(test_data.SEM_filenames)


###initializing the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print()
T = torch.randn(1, 4).to(device)
print(T)

##definitions
class RandomRotationFlipShift(object):
    def __call__(self, img):
        img = rotate(img, angle=torch.randint(-15, 15, size=(1,)).item(), interpolation=Image.BILINEAR)
        img = hflip(img) if torch.rand(1) < 0.5 else img
        # img = vflip(img) if torch.rand(1) < 0.5 else img
        # shift_x = random.randint(-10, 10)
        # shift_y = random.randint(-10, 10)
        # img = img.transform(img.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
        return img
def fourier_transform_image(image):
    # Convert to grayscale
    image_gray = image.convert('L')
    image_np = np.array(image_gray)

    # Apply Fourier transform
    f_transform = np.fft.fft2(image_np)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_shift))

    # Convert back to PIL Image
    magnitude_image = Image.fromarray(np.uint8(magnitude_spectrum))
    return magnitude_image



# Define your dataset class
class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, sem_files, reflectance_values, transform=None):
        self.sem_files = sem_files
        self.reflectance_values = reflectance_values
        self.transform = transform

    def __len__(self):
        return len(self.sem_files)

    def __getitem__(self, idx):
        sem_image = Image.open(self.sem_files[idx])
        fourier_image = fourier_transform_image(sem_image)
        if self.transform:
            sem_image = self.transform(sem_image)
            fourier_image = self.transform(fourier_image)

        reflectance_values = torch.tensor(self.reflectance_values[idx], dtype=torch.float32)
        # print(f"Processing index {idx}, sem_image shape: {sem_image.shape}, reflectance_values shape: {reflectance_values.shape}")

        return sem_image, fourier_image, reflectance_values

    
transform = transforms.Compose([
    RandomRotationFlipShift(),
    transforms.Resize((256, 256)),  # Resize the images to 250x250
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TestPairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, sem_files, reflectance_values, transform=None):
        self.sem_files = sem_files
        self.reflectance_values = reflectance_values
        self.transform = transform

    def __len__(self):
        return len(self.sem_files)

    def __getitem__(self, idx):
        sem_image = Image.open(self.sem_files[idx])
        fourier_image = fourier_transform_image(sem_image)
        if self.transform:
            sem_image = self.transform(sem_image)
            fourier_image = self.transform(fourier_image)
        reflectance_values = torch.tensor(self.reflectance_values[idx], dtype=torch.float32)

        return sem_image, fourier_image, reflectance_values

# Define your data transformation for testing 
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image to have a mean of 0.5 and standard deviation of 0.5
])
def test_accuracy_perchannel(test_loader, model):
    # Evaluate the model on the test dataset
    test_loss = 0.0
    mape_per_channel = np.zeros(10)  # Initialize an array to store MAPE for each channel
    num_samples = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            structural_inputs,fourier_inputs, reflectance_targets = data[0].to(device), data[1].to(device), data[2].to(device)

            # Forward pass
            reflectance_predictions = model(structural_inputs, fourier_inputs)
            # Calculate the loss (you can use different metrics as needed)
            loss = criterion(reflectance_predictions, reflectance_targets)

            # Convert tensors to numpy arrays for easier calculation
            reflectance_targets = reflectance_targets.cpu().numpy()
            reflectance_predictions = reflectance_predictions.cpu().numpy()

            # Calculate the MAPE for each channel
            channel_mape = np.mean(np.abs(reflectance_targets - reflectance_predictions), axis=0) * 100

            # Accumulate loss, MAPE, and count samples
            test_loss += loss.item()
            mape_per_channel += channel_mape
            num_samples += structural_inputs.size(0)

    # Calculate the average test loss and average MAPE per channel
    avg_test_loss = test_loss / num_samples
    avg_mape_per_channel = mape_per_channel / num_samples

    # Define the channel names
    channel_names = ["UV", "B", "G", "R", "740", "940", "fB", "fG", "fR", "poldiff"]

    # Print the results for each channel
    for channel, (name, mape) in enumerate(zip(channel_names, avg_mape_per_channel)):
       print(f"{name}: {mape:.4f}%", end=" | ")

    print(f"Average Test Loss: {avg_test_loss:.4f}")
    return(avg_mape_per_channel.tolist())

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out

# class RegressionModel(nn.Module):
#     def __init__(self):
#         super(RegressionModel, self).__init__()
#         self.in_channels = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)

#         # Increase the number of residual blocks
#         self.layer1 = nn.Sequential(
#             BasicResidualBlock(64, 128, stride=2),
#             BasicResidualBlock(128, 128, stride=1)
#         )
#         self.layer2 = nn.Sequential(
#             BasicResidualBlock(128, 256, stride=2),
#             BasicResidualBlock(256, 256, stride=1),
#             BasicResidualBlock(256, 256, stride=1)
#         )
#         self.layer3 = nn.Sequential(
#             BasicResidualBlock(256, 512, stride=2),
#             BasicResidualBlock(512, 512, stride=1),
#             BasicResidualBlock(512, 512, stride=1)
#         )
#         self.layer4 = nn.Sequential(
#             BasicResidualBlock(512, 1024, stride=2),
#             BasicResidualBlock(1024, 1024, stride=1),
#             BasicResidualBlock(1024, 1024, stride=1)
#         )

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle varying input sizes
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 10)  # 10 output channels for reflectance values

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))

#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         out = self.avgpool(out)  # Global average pooling
#         out = out.view(out.size(0), -1)  # Flatten
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)

#         return out
    
    
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.in_channels = 64

        # Pathway for original images
        self.original_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicResidualBlock(64, 128, stride=2),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 256, stride=2),
            BasicResidualBlock(256, 256),
            BasicResidualBlock(256, 512, stride=2),
            BasicResidualBlock(512, 512),
            BasicResidualBlock(512, 512),
            BasicResidualBlock(512, 1024, stride=2),
            BasicResidualBlock(1024, 1024),
            BasicResidualBlock(1024, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Pathway for Fourier-transformed images
        self.fourier_path = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicResidualBlock(64, 128, stride=2),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 256, stride=2),
            BasicResidualBlock(256, 256),
            BasicResidualBlock(256, 512, stride=2),
            BasicResidualBlock(512, 512),
            BasicResidualBlock(512, 512),
            BasicResidualBlock(512, 1024, stride=2),
            BasicResidualBlock(1024, 1024),
            BasicResidualBlock(1024, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layers after combining features from both paths
        self.fc1 = nn.Linear(1024 * 2, 512)  # *2 because we concatenate features from both paths
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x_original, x_fourier):
        original_features = self.original_path(x_original)
        fourier_features = self.fourier_path(x_fourier)

        # Flatten the output of both paths
        original_features = original_features.view(original_features.size(0), -1)
        fourier_features = fourier_features.view(fourier_features.size(0), -1)

        # Concatenate the features from both paths
        combined_features = torch.cat((original_features, fourier_features), dim=1)

        # Pass the combined features through the fully connected layers
        out = F.relu(self.fc1(combined_features))
        out = self.fc2(out)

        return out    
    

def weighted_mse_loss(input, target, weights):
        # Ensure that the weights are on the correct device
        weights = weights.to(input.device)
        # Reshape weights to be broadcastable to the shape of input/target
        # Assuming input and target have shape [batch_size, num_channels]
        weights = weights.view(1, -1)

        # Calculate the squared error
        squared_error = (input - target) ** 2

        # Apply the weights
        weighted_squared_error = weights * squared_error

        # Calculate the mean of the weighted errors
        loss = weighted_squared_error.mean()
        return loss
#Set up files for dataset    
test_sem_files=SEM_test_list
test_reflectance_values=ref_test_list
structural_files=SEM_train_list
reflectance_values=ref_train_list
###########################################################
# Hyperparameter ranges
# learning_rates = [0.00005,0.00001,0.0001,0.0005]
# batch_sizes = [6, 32, 128]
# batch_sizes = [6,16,32]
rotation_angle=15

# Define weights for each channel ("UV", "B", "G", "R", "740", "940", "fB", "fG", "fR", "poldiff")
channel_weights = torch.tensor([3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0])

# Define loss function
criterion = nn.MSELoss()

def train_and_evaluate_model(lr, batch_size, num_epochs=10,decay=0.00001):
    # Create a test dataset instance
    test_dataset = TestPairedImageDataset(test_sem_files, test_reflectance_values, transform=test_transform)
    # Create a test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define your dataset with paired structural images and reflectance values
    paired_dataset = PairedImageDataset(structural_files, reflectance_values, transform=transform)
    paired_dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the generator
    regression = RegressionModel().to(device)


    # Define optimizer
    optimizer = optim.Adam(regression.parameters(), lr=lr,weight_decay=decay)

    # Training loop
    loss_list=[]
    running_loss = 0.0
    mape_history = []
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(paired_dataloader, 0):
            structural_inputs,fourier_inputs, reflectance_targets = data[0].to(device), data[1].to(device), data[2].to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Add noise to the input data
            noise_factor=0.2
            noise = torch.randn_like(structural_inputs) * noise_factor
            # Scale the noise to match the same range and distribution as the input data
            # Assuming that your data was originally normalized using (0.5, 0.5)
            data_mean = 0.5
            data_std = 0.5
            noise = noise * data_std + data_mean

            # Add the scaled noise to the input data
            structural_inputs = structural_inputs + noise

             # Apply data augmentation
            augmented_structural_inputs = []
            for sem_image in structural_inputs:
                # Random rotation
                angle = random.uniform(-rotation_angle, rotation_angle)
                sem_image = rotate(sem_image, angle, interpolation=InterpolationMode.BILINEAR)
                augmented_structural_inputs.append(sem_image)

            augmented_structural_inputs = torch.stack(augmented_structural_inputs)

            # Forward pass
            reflectance_predictions = regression(augmented_structural_inputs, fourier_inputs)

            # Calculate the loss
            # loss = criterion(reflectance_predictions, reflectance_targets)
            # Calculate the custom weighted loss
            loss = weighted_mse_loss(reflectance_predictions, reflectance_targets, channel_weights)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Print loss
            if (i+1) % int(len(structural_files)/20) == 0:    # print every 1/20th of the size of the training dataset
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(paired_dataloader)}]: loss {running_loss / 100:.4f}")
                # mape_per_channel=test_accuracy_perchannel(test_dataloader)
                loss_list.append(running_loss)
                running_loss = 0.0
                
        mape_per_channel=test_accuracy_perchannel(test_dataloader, regression)
        mape_history.append(mape_per_channel)
        test_loss=np.mean(mape_per_channel)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            rounded=str(np.round(best_test_loss,3)).replace('.','_')
            torch.save(regression.state_dict(), f'{save_path}{prefix}_lr{lr}_bs{batch_size}_{weight_decay}decay_acc{rounded}_epoch{epoch}.pth')
            print(f'Epoch {epoch+1}: Test Loss Improved to {test_loss:.4f}, Model Saved')
    mape_history2 = np.array(mape_history)
    np.savetxt(f'{save_path}{prefix}_lr{lr}_bs{batch_size}_{weight_decay}decay_{epochs}_epochs.csv', mape_history2, delimiter=",")
# Loop over hyperparameters
# for lr in learning_rates:
#     for batch_size in batch_sizes:
#         print(f'Training with lr: {lr}, batch_size: {batch_size}')
#         train_and_evaluate_model(lr, batch_size,100,weight_decay)
        
train_and_evaluate_model(lr, batch_size,epochs,weight_decay)
stop=time()
print("elapsed time:", stop - start)
