import torch
from torch.utils.data import Dataset
import numpy as np

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset Class for DoA Data
class DOADataset(Dataset):
    def __init__(self, data, device):
        """
        data: Input dataset of size Nx(window size+1)x6.
        Splits the input into source (past observations) and target (future observations).
        """
        self.src_data = data[:, :-1, :].to(device)  # First 'window size' timesteps as source
        self.tgt_data = data[:, 1:, :2].to(device)  # Next timestep's DoA angles as target

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# Add NLoS Feature
def add_nlos_feature(blockset):
    """
    Add NLoS feature to the dataset based on extreme velocities and accelerations.

    Parameters:
    - blockset: Input dataset (NumPy array of size NxM).

    Returns:
    - Updated dataset with an additional NLoS feature column.
    """
    az_velocity_threshold = np.mean(blockset[:, :, 2]) + 3 * np.std(blockset[:, :, 2])
    el_velocity_threshold = np.mean(blockset[:, :, 3]) + 3 * np.std(blockset[:, :, 3])
    az_acc_threshold = np.mean(blockset[:, :, 4]) + 3 * np.std(blockset[:, :, 4])
    el_acc_threshold = np.mean(blockset[:, :, 5]) + 3 * np.std(blockset[:, :, 5])

    M, N = blockset.shape[:2]
    los_feature = np.zeros((M, N))  # 0: LoS, 1: NLoS
    for i in range(M):
        for j in range(1, N):  # Start from 1 to compare with the previous step
            if (blockset[i, j, 2] > az_velocity_threshold or
                    blockset[i, j, 3] > el_velocity_threshold or
                    blockset[i, j, 4] > az_acc_threshold or
                    blockset[i, j, 5] > el_acc_threshold):
                los_feature[i, j] = los_feature[i, j - 1] = 1  # Mark current and previous as NLoS

    return np.concatenate([blockset, los_feature[:, :, None]], axis=2)  # Append new feature


# Normalize Data
def normalize_data(data, data_min, data_max):
    """
    Normalize data to the range [-1, 1].

    Parameters:
    - data: Input tensor.
    - data_min: Minimum value of the data.
    - data_max: Maximum value of the data.

    Returns:
    - Normalized data in the range [-1, 1].
    """
    data[data > data_max] = data_max
    data[data < data_min] = data_min
    return -1 + 2 * ((data - data_min) / (data_max - data_min))


# De-normalize Data
def denormalize_data(normalized_data, data_min, data_max):
    """
    De-normalize data from the range [-1, 1] back to the original scale.

    Parameters:
    - normalized_data: Tensor of normalized data.
    - data_min: Minimum value of the original data.
    - data_max: Maximum value of the original data.

    Returns:
    - De-normalized data in the original scale.
    """
    return ((normalized_data + 1) / 2) * (data_max - data_min) + data_min


# Angular Distance in 3D
def angular_distance_3D(theta1, theta2, phi1, phi2):
    """
    Calculate the angular distance between two 3D points (DoAs) based on their azimuth and elevation.

    Parameters:
    - theta1, theta2: Elevation angles in degrees.
    - phi1, phi2: Azimuth angles in degrees.

    Returns:
    - Angular distance in degrees.
    """
    theta1, theta2, phi1, phi2 = torch.deg2rad(torch.tensor([theta1, theta2, phi1, phi2]))
    distance = torch.rad2deg(torch.arccos(
        torch.sin(theta1) * torch.cos(phi1) * torch.sin(theta2) * torch.cos(phi2) +
        torch.sin(theta1) * torch.sin(phi1) * torch.sin(theta2) * torch.sin(phi2) +
        torch.cos(theta1) * torch.cos(theta2)
    ))
    return 0.0 if torch.isnan(distance) else distance


def transform_elevation_azimuth(azimuths, elevations):
    # Initialize transformed arrays
    transformed_elevations = np.zeros_like(elevations)
    transformed_azimuths = np.zeros_like(azimuths)

    # Process azimuth and elevation transformations
    for i in range(len(azimuths)):
        azimuth = azimuths[i]
        elevation = elevations[i]

        if 0 <= azimuth <= 90:
            transformed_azimuths[i] = azimuth  # Remains the same
            transformed_elevations[i] = elevation  # Remains the same
        elif 90 < azimuth <= 180:
            transformed_azimuths[i] = 180 - azimuth
            transformed_elevations[i] = - elevation
        elif 180 < azimuth <=270:
            transformed_azimuths[i] = -(azimuth - 180)
            transformed_elevations[i] = -elevation
        elif 270 < azimuth <= 360:
            transformed_azimuths[i] = -(360-azimuth)
            transformed_elevations[i] = elevation
    return  transformed_azimuths, transformed_elevations


# Inverse Transform Elevation and Azimuth
def inverse_transform_elevation_azimuth(transformed_azimuths, transformed_elevations):
    """
    Convert transformed elevation and azimuth back to their original convention.

    Parameters:
    - transformed_azimuths: Array of transformed azimuth angles.
    - transformed_elevations: Array of transformed elevation angles.

    Returns:
    - Original azimuth and elevation angles.
    """
    single_value_input = np.isscalar(transformed_azimuths)
    if single_value_input:
        transformed_azimuths = np.array([transformed_azimuths])
        transformed_elevations = np.array([transformed_elevations])

    transformed_azimuths = np.clip(transformed_azimuths, -90, 90)
    transformed_elevations = np.clip(transformed_elevations, -90, 90)

    original_azimuths = np.zeros_like(transformed_azimuths)
    original_elevations = np.zeros_like(transformed_elevations)

    for i in range(transformed_azimuths.shape[0]):
        t_azimuth = transformed_azimuths[i]
        t_elevation = transformed_elevations[i]

        if 0 <= t_azimuth <= 90 and 0 <= t_elevation <= 90:
            original_azimuths[i], original_elevations[i] = t_azimuth, t_elevation
        elif -90 <= t_azimuth <= 0 and 0 <= t_elevation <= 90:
            original_azimuths[i], original_elevations[i] = 360 + t_azimuth, t_elevation
        elif 0 < t_azimuth <= 90 and -90 <= t_elevation <= 0:
            original_azimuths[i], original_elevations[i] = 180 - t_azimuth, -t_elevation
        elif -90 < t_azimuth < 0 and -90 <= t_elevation <= 0:
            original_azimuths[i], original_elevations[i] = 180 - t_azimuth, -t_elevation

    if single_value_input:
        return original_azimuths[0], original_elevations[0]
    return original_azimuths, original_elevations
