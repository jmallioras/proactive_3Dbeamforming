#This script processes input data to calculate velocities and accelerations of Direction of Arrival (DoA) signals,
# normalize and prepare the data for inference, and finally uses a pre-trained transformer model (contextDOAEncoder4) to
# predict the DoAs for a given input. The script includes provisions for loading and using two model configurations:
# small and large (although only the small model is currently active).

# Import Libraries
import torch
import os
import numpy as np
import sys
import io

# Add the project path to the system path for module imports
sys.path.insert(1, '/Users/ioannis.mallioras/PycharmProjects/TRANSFORMER')
from utility_functions import *  # Contains functions for data manipulation
from transformer_utils import *  # Contains the transformer model definition

# Save the current stdout so that we can revert sys.stdout after processing
original_stdout = sys.stdout

# Set device to CPU (adjust for GPU if available)
device = 'cpu'

# Function to calculate velocities and accelerations for a single path
def calculate_vel_acc_single(currentPath):
    """
    Processes a single path to compute the velocities and accelerations for azimuth and elevation angles.

    Parameters:
    - currentPath: Nx2 array of azimuth and elevation angles.

    Returns:
    - newData: Nx6 array with azimuth/elevation, velocities, and accelerations.
    """
    dt = 1  # Time step
    # Calculate velocities
    elevationVelocity = np.diff(currentPath[:, 1]) / dt
    azimuthVelocity = np.diff(currentPath[:, 0]) / dt

    # Calculate accelerations
    elevationAcceleration = np.diff(elevationVelocity) / dt
    azimuthAcceleration = np.diff(azimuthVelocity) / dt

    # Create an Nx6 dataset
    newData = np.zeros((len(azimuthAcceleration), 6))
    newData[:, 0:2] = currentPath[2:, :]  # DoAs (excluding the first two points)
    newData[:, 2:4] = np.vstack([azimuthVelocity[1:], elevationVelocity[1:]]).T  # Velocities
    newData[:, 4:6] = np.vstack([azimuthAcceleration, elevationAcceleration]).T  # Accelerations

    return newData

# Model Hyperparameters
model_path = 'model_checkpoint_small_v1.pth'  # Path to the small model checkpoint
batch_size = 64
additional_features_dim = 4
context_size = 2
d_model = 256
input_dim = 2
nhead = 4
num_encoder_layers = 3
dropout = 0.00

# Initialize the small model
model_tnn_small = contextDOAEncoder4(
    input_dim=input_dim, context_size=context_size,
    additional_features_dim=additional_features_dim, d_model=d_model,
    nhead=nhead, num_encoder_layers=num_encoder_layers, dropout=dropout
).to(device)

# Load pre-trained model weights for the small model
checkpoint = torch.load(model_path, map_location=device)
model_tnn_small.load_state_dict(checkpoint['model_state_dict'])

# Placeholder code for the large model (commented out)
'''
# Uncomment and adjust if using the large model
model_path = 'model_checkpoint_big_v1.pth'
d_model = 512
nhead = 8
num_encoder_layers = 4
model_tnn_big = contextDOAEncoder4(
    input_dim=input_dim, context_size=context_size,
    additional_features_dim=additional_features_dim, d_model=d_model, 
    nhead=nhead, num_encoder_layers=num_encoder_layers, dropout=dropout
).to(device)
checkpoint = torch.load(model_path, map_location=device)
model_tnn_big.load_state_dict(checkpoint['model_state_dict'])
'''

# INPUT PREPROCESSING
x = np.array(x, dtype=np.single)  # Input data (DoA observations)
N_SIGNALS = 3  # Number of signals to process
N_STEPS = 5  # Number of time steps
total_input = np.zeros((N_SIGNALS, N_STEPS, 6))  # Placeholder for processed input

# Process each signal to compute velocities and accelerations
for idx in range(N_SIGNALS):
    tmp_az, tmp_el = transform_elevation_azimuth(x[:, 2 * idx], x[:, 2 * idx + 1])  # Transform azimuth and elevation
    tmp = np.array([tmp_az, tmp_el]).T  # Combine into an array
    total_input[idx, :, :] = calculate_vel_acc_single(tmp)  # Calculate features

# Add NLoS indicator to the dataset
total_input = add_nlos_feature(total_input)
total_input = torch.from_numpy(total_input).float()

# Normalization parameters for angles, velocities, and accelerations
az_min, az_max = -89.99993133544922, 89.99988555908203
el_min, el_max = -63.164306640625, 58.51708221435547
v_az_min, v_az_max = -22.083147048950195, 22.39622688293457
v_el_min, v_el_max = -17.6036434173584, 17.59757423400879
a_az_min, a_az_max = -33.608055114746094, 33.583335876464844
a_el_min, a_el_max = -26.34954833984375, 26.35047149658203

# De-normalization tensors
denorm_az = torch.tensor([az_min, az_max])
denorm_el = torch.tensor([el_min, el_max])

# Block printing during normalization
sys.stdout = io.StringIO()

# Normalize angles
total_input[:, :, 0] = normalize_data(total_input[:, :, 0], az_min, az_max)
total_input[:, :, 1] = normalize_data(total_input[:, :, 1], el_min, el_max)

# Normalize velocities
total_input[:, :, 2] = normalize_data(total_input[:, :, 2], v_az_min, v_az_max)
total_input[:, :, 3] = normalize_data(total_input[:, :, 3], v_el_min, v_el_max)

# Normalize accelerations
total_input[:, :, 4] = normalize_data(total_input[:, :, 4], a_az_min, a_az_max)
total_input[:, :, 5] = normalize_data(total_input[:, :, 5], a_el_min, a_el_max)

# Restore stdout
sys.stdout = original_stdout

# Prepare input tensors for the model
total_input = total_input.transpose(0, 1)  # Transpose for correct input shape
input = total_input[:, :, :input_dim].to(device)  # Extract DoA input
context = total_input[:, :, 6].int().to(device)  # Extract NLoS context
cont_feat = total_input[:, :, input_dim:6].to(device)  # Extract continuous features

# Model Evaluation
model_tnn_small.eval()
out_small = model_tnn_small(input, context, cont_feat)  # Predict using the small model
out_small = out_small.detach().cpu()  # Detach and move output to CPU

# De-normalize predicted azimuth and elevation
out_az_small = denormalize_data(out_small[:, 0], denorm_az[0], denorm_az[1])
out_el_small = denormalize_data(out_small[:, 1], denorm_el[0], denorm_el[1])
out_az_small, out_el_small = inverse_transform_elevation_azimuth(out_az_small.tolist(), out_el_small.tolist())

# Prepare output
out_small = np.array([out_az_small, out_el_small]).T
output = out_small[0].tolist(), out_small[1].tolist(), out_small[2].tolist()

# Uncomment for combined output with the large model
# output = out_small[0].tolist(), out_small[1].tolist(), out_small[2].tolist(), out_big[0].tolist(), out_big[1].tolist(), out_big[2].tolist()
