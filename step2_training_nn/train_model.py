# Import Libraries
'''
- `transformer_utils.py` contains the class definition of the transformer (encoder-only) model`contextDOAEncoder4`
 that was used as the proposed architecture.

-`utility_functions.py' contains various important functions for data manipulation, plotting, and more that are used throughout
this script.
'''

import os
from utility_functions import *
from transformer_utils import *
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformer_utils import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# TRAINING the TRANSFORMER model
'''
- In the following script, we import the dataset created in step 1 and prepare the data to later train the transformer
models with.
- The dataset path needs to be changed according to its location in your local directory.
- You can modify parameters such as the
'''

# Device config
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

# Configure Path
pwd = os.getcwd()

# Import Dataset
# CHANGE THE PATH BASED ON YOUR LOCAL DIRECTORY
data = loadmat(pwd + '/Dataset/total_dataset.mat')
block_dataset = data['out']
print(f"Imported Dataset of shape:[{block_dataset.shape[0]} x {block_dataset.shape[1]} x {block_dataset.shape[2]}]")


# Set sliding window size
'''
Depending on your needds you can modify these values, more specifically, the window size, to change the number of past 
observations the NN will use to predict the future one.
'''
window_size = 5
step_ahead = 1

# Add the NLoS feature based on extreme velocities and accelerations
block_dataset = add_nlos_feature(block_dataset)
block_dataset = torch.from_numpy(block_dataset).float().to(device)

# NORMALIZATION TO [-1,1]
'''
IMPORTANT NOTE:

The following normalization is unconvetional in the sense that we derive the min and max values from the dataset and
not by explicitly setting them for each input value. This is because the application is location-based and the angular 
sectors covered at each scenario may be different from the full [-90, 90] degrees of the operational range of the base
station. The DoA trajectories in the dataset cover a plethora of possible incoming signal directions, 
based on the predefined paths and the coverage map of the base station. Thus, by limiting the normalization sector
closer to the range of the recorded DoAs, we utilize the normalization sector better which improves training performance.
'''
# Angles

# Keep the max and min values to be used for de-normalization later
denorm_az = torch.tensor([torch.min(block_dataset[:, :, 0]), torch.max(block_dataset[:, :, 0])])
denorm_el = torch.tensor([torch.min(block_dataset[:, :, 1]), torch.max(block_dataset[:, :, 1])])

# Normalize
block_dataset[:, :, 0] = normalize_data(block_dataset[:, :, 0], torch.min(block_dataset[:, :, 0]),
                                        torch.max(block_dataset[:, :, 0]))
block_dataset[:, :, 1] = normalize_data(block_dataset[:, :, 1], torch.min(block_dataset[:, :, 1]),
                                  torch.max(block_dataset[:, :, 1]))

# Velocities
min_value = torch.mean(block_dataset[:, :, 2]) - 3*torch.std(block_dataset[:, :, 2])
max_value = torch.mean(block_dataset[:, :, 2]) + 3*torch.std(block_dataset[:, :, 2])
block_dataset[:, :, 2] = normalize_data(block_dataset[:, :, 2], min_value, max_value)

min_value = torch.mean(block_dataset[:, :, 3]) - 3*torch.std(block_dataset[:, :, 3])
max_value = torch.mean(block_dataset[:, :, 3]) + 3*torch.std(block_dataset[:, :, 3])
block_dataset[:, :, 3] = normalize_data(block_dataset[:, :, 3], min_value, max_value)

# Accelerations
min_value = torch.mean(block_dataset[:, :, 4]) - 3*torch.std(block_dataset[:, :, 4])
max_value = torch.mean(block_dataset[:, :, 4]) + 3*torch.std(block_dataset[:, :, 4])
block_dataset[:, :, 4] = normalize_data(block_dataset[:, :, 4], min_value, max_value)

min_value = torch.mean(block_dataset[:, :, 5]) - 3*torch.std(block_dataset[:, :, 5])
max_value = torch.mean(block_dataset[:, :, 5]) + 3*torch.std(block_dataset[:, :, 5])
block_dataset[:, :, 5] = normalize_data(block_dataset[:, :, 5], min_value, max_value)


# Train-evaluation split
trainset = block_dataset[:int(0.9 * len(block_dataset))]
valset = block_dataset[int(0.9 * len(block_dataset)):]

# Shuffle the rows of each dataset
trainset = trainset[torch.randperm(trainset.shape[0])]
valset = valset[torch.randperm(valset.shape[0])]

# Test-only
# valset = block_dataset
print("Starting training...")
# Hyperparameters

# SET THE NAMES FOR THE MODEL CHECKPOINT AND THE LEARNING CURVES FIGURE
model_path = 'Checkpoints/model_checkpoint_big_v1.pth'
figure_name = 'training_prog_big_v1.png'

# Training and model Hyperparameters
batch_size = 64
additional_features_dim = 4
context_size = 2
input_dim = 2
dropout = 0.00
lr = 0.0001
num_epochs = 500

# Big model design
#d_model = 512
#context_dim = d_model
#nhead = 8
#num_encoder_layers = 4

# Small model design
d_model = 256
context_dim = d_model
nhead = 4
num_encoder_layers = 2

# Dataloaders
train_dataset = DOADataset(trainset, device)
val_dataset = DOADataset(valset, device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = contextDOAEncoder4(input_dim=input_dim, context_size=context_size,
                           additional_features_dim=additional_features_dim, d_model=d_model, nhead=nhead,
                           num_encoder_layers=num_encoder_layers, dropout=dropout)
# Send to gpu
model.to(device)

# Try loading previous progress
try:
    # Load the saved weights
    checkpoint = torch.load(model_path, map_location="cuda:0")
    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    lr = checkpoint['learning_rates']
    elapsed_time = checkpoint['elapsed_time']
    print("Model loaded!")
except:
    print("New model!")
    # Initialize lists to store the losses
    train_losses = []
    val_losses = []
    elapsed_time = 0

# Training tools
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)  # Adjust lr as needed
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

# Main training loop
start_time = time.time()  # Start time measurement
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch_id, (src_batch, tgt_batch) in enumerate(train_loader):
        src_batch = src_batch.transpose(0, 1)  # Prepare input in the shape [batch_size, sequence_length, input_size]
        tgt_batch = tgt_batch[:, -1, :]  # Keep the last value out of the shifted inout sequence
        optimizer.zero_grad()  # Clear gradients
        context = src_batch[:, :, 6].int()  # Extract binary NLoS indicator
        cont_feat = src_batch[:, :, 2:6]  # Extract continuous features (velocity,acceleration)
        input = src_batch[:, :, :2]  # Extract input sequence (DoAs only)
        output = model(input, context, cont_feat)  # Forward pass
        loss = torch.sqrt(criterion(output, tgt_batch))  # RMSE loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item()

    if batch_id % 5000 == 0:
        print(f"\nTRAIN Batch: {batch_id} (Prediction Example)")
        print(f"TNN hat:{tgt_batch[-1, :]}")
        print(f"TNN Out: {output[-1, :]}")
        print(f"Loss: {total_loss / (batch_id + 1)}")

    # Gather epoch losses
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for batch_id, (src_batch, tgt_batch) in enumerate(val_loader):
            src_batch = src_batch.transpose(0, 1)
            tgt_batch = tgt_batch[:, -1, :]
            context = src_batch[:, :, 6].int()
            cont_feat = src_batch[:, :, 2:6]
            input = src_batch[:, :, :2]
            output = model(input, context, cont_feat)  # Forward pass
            loss = torch.sqrt(criterion(output, tgt_batch))  # RMSE loss
            total_val_loss += loss.item()
            if batch_id % 5000 == 0:
                print(f"\nTEST Batch: {batch_id} (Prediction Example)")
                print(f"TNN hat:{tgt_batch[-1, :]}")
                print(f"TNN Out: {output[-1, :]}")
                print(f"Loss: {total_val_loss / (batch_id + 1)}")
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    # Call scheduler.step() after the validation phase
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - start_time  # Calculate elapsed time

    # Plotting
    clear_output(wait=True)  # Clear the previous plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title(
        f"Training and Validation RMSE Progression [Time Elapsed: {(elapsed_time + epoch_time) / 60:.2f}min, LR: {current_lr}]\nWindow:{window_size} Batch Size:{batch_size} Emb. Dim:{d_model} Heads:{nhead} Enc Layers:{num_encoder_layers} Dropout:{dropout}")
    plt.legend()
    plt.grid(True)

    # Annotations
    plt.annotate(f"{avg_train_loss:.4f}", (len(train_losses) - 1, train_losses[-1]), textcoords="offset points",
                 xytext=(0, 10), ha='center', color='blue')
    plt.annotate(f"{avg_val_loss:.4f}", (len(val_losses) - 1, val_losses[-1]), textcoords="offset points",
                 xytext=(0, 10), ha='center', color='orange')
    plt.savefig(pwd+f"/Plots/{figure_name}", dpi=400)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': current_lr,
        'elapsed_time': elapsed_time + epoch_time
    }, model_path)

    # Save the model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': current_lr,
        'elapsed_time': elapsed_time + epoch_time
    }, model_path)

print("Training Completed!")