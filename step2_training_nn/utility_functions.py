import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DOADataset(Dataset):
    def __init__(self, data,device):
        """
        data: Your dataset sized Nx(window size+1)x6
        """
        self.src_data = data[:, :-1, :].to(device)  # Use the first 'window size' timesteps as source
        self.tgt_data = data[:, 1:, :2].to(device)  # Use the next timestep's DOA angles as target

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]




def get_batch(data,batch_size, block_size):
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i,:block_size] for i in ix])
    y = torch.stack([data[i,1:block_size+1] for i in ix])
    return x,y

def numpy_blockenize_dataset(dataset, blocksize, step_ahead=1):
    blocks = []
    for path in dataset:
        for i in range(len(path[0]) - blocksize - step_ahead + 1):
            block = path[0][i:i+blocksize+step_ahead,:]
            blocks.append(block)
    return np.array(blocks, dtype=np.single)

def torch_blockenize_dataset(dataset, blocksize, step_ahead=1):
    blocks = []
    for path in dataset:
        for i in range(len(path[0]) - blocksize - step_ahead + 1):
            block = path[0][i:i+blocksize+step_ahead,:]
            blocks.append(block)
    return torch.from_numpy(np.array(blocks, dtype=np.single))

def normalize_data(data, data_min=None, data_max=None, zerotoone = None):
    # Assuming data is a NumPy array of shape Nx(window size+1)x6
    # Compute the min and max across the entire dataset for each feature
    if data_min==None and data_max==None:
        data_min = data.min(axis=(0, 1), keepdims=True)
        data_max = data.max(axis=(0, 1), keepdims=True)
    else:
        data[data>data_max] = data_max
        data[data<data_min] = data_min
    print(f"Data normalization in range [{data_min}, {data_max}]...")
    # Normalize to [-1, 1]
    if zerotoone==None or zerotoone==False:
        return -1 + 2 * ((data - data_min) / (data_max - data_min))
    elif zerotoone==True:
        return ((data - data_min) / (data_max - data_min))

def denormalize_data(normalized_data, data_min, data_max):
    """
    De-normalize data from the [-1, 1] range back to its original scale.

    Parameters:
    - normalized_data: NumPy array of normalized data.
    - data_min: The minimum value(s) of the original data before normalization.
    - data_max: The maximum value(s) of the original data before normalization.

    Returns:
    - De-normalized data.
    """
    return ((normalized_data + 1) / 2) * (data_max - data_min) + data_min

def angular_distance_3D(theta1, theta2, phi1, phi2):
    theta1, theta2, phi1, phi2 = torch.deg2rad(torch.tensor([theta1, theta2, phi1, phi2]))
    w = torch.tensor(0).to(device)
    w = torch.rad2deg(torch.arccos(torch.sin(theta1) * torch.cos(phi1) * torch.sin(theta2) * torch.cos(phi2) \
                                   + torch.sin(theta1) * torch.sin(phi1) * torch.sin(theta2) * torch.sin(phi2) \
                                   + torch.cos(theta1) * torch.cos(theta2)))
    if w.isnan():
        w = torch.tensor(0.0, dtype=torch.float32)
    return w




def custom_normalize(data, start_col=2, end_col=6, norm_range=(-0.8, 0.8), out_range=(-1, 1),zerotoone=False):
    normalized_data = np.copy(data)
    normalized_data[:, :, 0] = normalize_data(normalized_data[:, :, 0], data_min =0 , data_max=360, zerotoone=zerotoone)
    normalized_data[:, :, 1] = normalize_data(normalized_data[:, :, 1],data_min =0 , data_max=90, zerotoone=zerotoone)
    for i in range(start_col, end_col):
        col_data = data[:, :, i].reshape(-1)  # Flatten the data for the column across all batches
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)

        # Normalize within 1 standard deviation
        inlier_min = mean_val - 1 * std_val
        inlier_max = mean_val + 1 * std_val
        scaled_data = np.interp(col_data, (inlier_min, inlier_max), norm_range)

        # Handle outliers below inlier_min
        outliers_below = col_data < inlier_min
        scaled_data[outliers_below] = np.interp(col_data[outliers_below], (np.min(col_data), inlier_min), (out_range[0], norm_range[0]))

        # Handle outliers above inlier_max
        outliers_above = col_data > inlier_max
        scaled_data[outliers_above] = np.interp(col_data[outliers_above], (inlier_max, np.max(col_data)), (norm_range[1], out_range[1]))

        # Place normalized data back into the array
        normalized_data[:, :, i] = scaled_data.reshape(data.shape[0], data.shape[1])

    return normalized_data

def plot_distribution_with_stats(data, column_index):
    # Flatten the data for the given column index across all rows and samples
    flattened_data = data[:, :, column_index].flatten()

    # Calculate statistics
    mean_val = np.mean(flattened_data)
    std_val = np.std(flattened_data)
    # Assuming outliers are values beyond 3 standard deviations from the mean
    outliers_region = (mean_val - 3 * std_val, mean_val + 3 * std_val)

    # Create histogram plot
    plt.figure(figsize=(8, 4))
    plt.hist(flattened_data, bins=20, alpha=0.75, color='blue')
    plt.title(f'Distribution of Column {column_index}')
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.show()

    # Print the statistics
    print(f"Column {column_index} - Mean: {mean_val:.2f}, Std: {std_val:.2f}, Outliers Region: < {outliers_region[0]:.2f} or > {outliers_region[1]:.2f}")

def plot_continuous_distribution(data, column_index):
    # Flatten the data for the given column index across all rows and samples
    flattened_data = data[:, :, column_index].flatten()

    # Calculate statistics
    mean_val = np.mean(flattened_data)
    std_val = np.std(flattened_data)
    outliers_region = (mean_val - 3 * std_val, mean_val + 3 * std_val)

    # Calculate the KDE for smooth density
    kde = gaussian_kde(flattened_data)
    x_range = np.linspace(flattened_data.min(), flattened_data.max(), 500)
    kde_density = kde(x_range)

    # Plotting the density with a continuous line
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, kde_density, label='Density', color='blue')
    plt.title(f'Distribution of Column {column_index}')
    plt.ylabel('Density')
    plt.xlabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the statistics
    print(f"Column {column_index} - Mean: {mean_val:.2f}, Std: {std_val:.2f}, Outliers Region: < {outliers_region[0]:.2f} or > {outliers_region[1]:.2f}")


def plot_line_from_points(points):
    """
    Plot a line based on an array of points.

    Parameters:
        points (numpy.ndarray): An Nx2 numpy array of x and y coordinates.
    """
    if points.shape[1] != 2:
        raise ValueError("The input array must be Nx2.")

    # Extract the x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Plot the points as a line
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')  # 'o' to mark the points, remove if not needed
    plt.title("Line Plot of Given Points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()

def add_nlos_feature(blockset):
    az_velocity_threshold = np.mean(blockset[:,:,2]) + 3 * np.std(blockset[:,:,2])
    el_velocity_threshold = np.mean(blockset[:,:,3]) + 3 * np.std(blockset[:,:,3])
    az_acc_threshold = np.mean(blockset[:,:,4]) + 3 * np.std(blockset[:,:,4])
    el_acc_threshold = np.mean(blockset[:,:,5]) + 3 * np.std(blockset[:,:,5])

    M = blockset.shape[0]
    N = blockset.shape[1]
    # Placeholder for the new feature
    los_feature = np.zeros((M, N))  # 0 will signify LoS, 1 will signify NLoS
    # Calculate the LoS/NLoS feature
    for i in range(M):
        for j in range(1,N):  # Start from 1 because we'll compare to the previous step
            if blockset[i,j,2] > az_velocity_threshold or blockset[i,j,3] > el_velocity_threshold or blockset[i,j,4] > az_acc_threshold or blockset[i,j,5] > el_acc_threshold:
                los_feature[i, j] = 1  # Mark as NLoS
                los_feature[i, j-1] = 1  # Mark as NLoS

    # Concatenate the new feature to the original dataset
    extended_data = np.concatenate([blockset, los_feature[:, :, None]], axis=2)  # New shape will be (M, N, 7)
    return extended_data
    
    
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


def inverse_transform_elevation_azimuth(transformed_azimuths, transformed_elevations):
    # Check if inputs are arrays or single values and convert single values to arrays
    single_value_input = np.isscalar(transformed_azimuths)
    if single_value_input:
        transformed_azimuths = np.array([transformed_azimuths])
        transformed_elevations = np.array([transformed_elevations])

    transformed_azimuths = np.minimum(np.maximum(transformed_azimuths,-90),90)
    transformed_elevations = np.minimum(np.maximum(transformed_elevations,-90),90)

    # Initialize arrays to store the original azimuths and elevations
    original_azimuths = np.zeros_like(transformed_azimuths)
    original_elevations = np.zeros_like(transformed_elevations)

    # Process the inverse transformations
    for i in range(transformed_azimuths.shape[0]):
        t_azimuth = transformed_azimuths[i]
        t_elevation = transformed_elevations[i]

        if 0 <= t_azimuth <= 90 and 0<= t_elevation<=90:
            original_azimuths[i] = t_azimuth
            original_elevations[i] = t_elevation
        elif -90 <= t_azimuth <= 0 and 0<= t_elevation<=90:
            original_azimuths[i] = 360 + t_azimuth
            original_elevations[i] = t_elevation
        elif 0 < t_azimuth <= 90 and -90<= t_elevation<=0:
            original_azimuths[i] = 180 - t_azimuth
            original_elevations[i] = -t_elevation
        elif -90 < t_azimuth < 0 and -90<= t_elevation<=0:
            original_azimuths[i] = 180 - t_azimuth
            original_elevations[i] = -t_elevation

    # If the input was a single value, return single values instead of arrays
    if single_value_input:
        return original_azimuths[0], original_elevations[0]
    return original_azimuths, original_elevations


import numpy as np
def transform_DoA_convention(pathsMatrix):
    for i in range(len(pathsMatrix)):
        pathsMatrix[i][0][:,0], pathsMatrix[i][0][:,1]= transform_elevation_azimuth(pathsMatrix[i][0][:,0],pathsMatrix[i][0][:,1])
    return pathsMatrix

def calculate_vel_acc(pathsMatrix):
    # Assuming time step
    dt = 1

    # Assume your matrix 'pathsMatrix' is a list of numpy arrays
    #pathsMatrix = []  # This should be filled with your actual data
    numPaths = len(pathsMatrix)  # Number of paths in the dataset

    # Initialize a list to hold the processed data
    processedData = []

    for i in range(numPaths):
        # Extract the current path (Nx2 numpy array)
        currentPath = pathsMatrix[i][0]
        if currentPath.shape[0] < 10:
            continue

        # Calculate the differences in elevation and azimuth to get velocities
        elevationVelocity = np.diff(currentPath[:, 1]) / dt
        azimuthVelocity = np.diff(currentPath[:,0])/dt

        # Calculate acceleration by finding the difference in velocities
        elevationAcceleration = np.diff(elevationVelocity) / dt  # Pad with zero for the last element
        azimuthAcceleration = np.diff(azimuthVelocity) / dt  # Pad with zero for the last element

        # Preallocate the new (N-2)x6 array for the current path
        newData = np.zeros((len(azimuthAcceleration), 6))

        # Populate the new dataset
        newData[:, 0:2] = currentPath[2:, :]  # DOAs (excluding the first two)
        newData[:, 2:4] = np.vstack([azimuthVelocity[1:], elevationVelocity[1:]]).T  # Velocities (excluding the first one)
        newData[:, 4:6] = np.vstack([azimuthAcceleration, elevationAcceleration]).T  # Accelerations

        # Store the processed data
        processedData.append(newData)

    # Keep only non-empty cells
    processedData = [data for data in processedData if data.size > 0]

    return processedData


def blockenize_dataset(dataset, blocksize, step_ahead=1):
    blocks = []
    for path in dataset:
        for i in range(len(path) - blocksize - step_ahead + 1):
            block = path[i:i+blocksize+step_ahead,:]
            blocks.append(block)
    blocked_data = np.array(blocks, dtype=np.single)
    print(f"Blocked dataset size: {blocked_data.shape[0]}")
    return blocked_data

