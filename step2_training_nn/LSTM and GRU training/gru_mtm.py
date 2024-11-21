import torch
import torch.nn as nn
import numpy as np
from transformer_utils import *
from dataset_functions import *
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
import os
from scipy.io import loadmat

class DoAGRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, max_sequence_length=5):
        super(DoAGRUModel, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer to predict elevation and azimuth

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # Forward pass through GRU layer
        packed_output, hidden = self.gru(packed_input)
        # Unpack the sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Applying the fully connected layer to each time step
        output = self.fc(output)

        # Pad the output to max_sequence_length if necessary
        if output.shape[1] < self.max_sequence_length:
            pad_size = self.max_sequence_length - output.shape[1]
            padding = torch.zeros((x.shape[0], pad_size, output.shape[2]), device=output.device)
            output = torch.cat((output, padding), dim=1)

        # Applying tanh activation to ensure output is within the desired range [-1,1]
        output = torch.tanh(output)
        return output
        

# Device config
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

# Dataset prep
# Dataset preparation
pwd = os.getcwd()
window_size = 5
step_ahead = 1
data = loadmat(pwd + '/Dataset/total_dataset.mat')
block_dataset = data['out']
print(f"Imported Dataset of shape:[{block_dataset.shape[0]} x {block_dataset.shape[1]} x {block_dataset.shape[2]}]")

# Convert to tensor
block_dataset = torch.from_numpy(block_dataset).float()

# Angles
denorm_az = torch.tensor([torch.min(block_dataset[:,:,0]), torch.max(block_dataset[:,:,0])])
denorm_el = torch.tensor([torch.min(block_dataset[:,:,1]), torch.max(block_dataset[:,:,1])])

block_dataset[:,:,0] = normalize_data(block_dataset[:,:,0], torch.min(block_dataset[:,:,0]), torch.max(block_dataset[:,:,0]))
block_dataset[:,:,1] = normalize_data(block_dataset[:,:,1], torch.min(block_dataset[:,:,1]), torch.max(block_dataset[:,:,1]))

block_dataset = block_dataset[:,:,:2]


import torch

# Sample dataset generation
N = block_dataset.shape[0]  # Total records
K = block_dataset.shape[1]    # Timesteps per record
data = block_dataset

# Create input and target datasets
inputs = data[:, :-1, :]  # Using the first K-1 timesteps as input
targets = data[:, 1:, :]  # Using the remaining timesteps as the target

# Apply random masking for variability in training
max_length = inputs.shape[1]  # Max length is now K-1
lengths = torch.randint(1, max_length + 1, (N,))  # Lengths vary from 1 to K-1

# Mask the inputs based on lengths
masked_inputs = torch.zeros_like(inputs)
masked_targets = torch.zeros_like(targets)
for i in range(N):
    length = lengths[i]
    masked_inputs[i, :length, :] = inputs[i, :length, :]
    masked_targets[i, :length, :] = targets[i, :length, :]
    

# Assuming masked_inputs, targets, and lengths are already defined
dataset = TensorDataset(masked_inputs, masked_targets, lengths)

# Define the size of the test set
test_size = int(0.1 * len(dataset))  # 20% for testing
train_size = len(dataset) - test_size  # 80% for training

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Set the batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Assume input_seq, target_seq, and seq_lengths are your batched input sequences, target sequences, and sequence lengths, respectively
#model = GRUModel(input_dim=2, hidden_dim=128, output_dim=2, num_layers=2)
input_dim = 2
hidden_dim = 160
num_layers = 2
lr = 0.0001
model_path = pwd+'/gru_mtm_checkpoint2.pth'
plotname = 'gru_mtm_training_prog2.png'

# Initialize the model
model = DoAGRUModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers, max_sequence_length=5).to(device)

# Example forward pass - include this in your training and validation loops
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from IPython.display import clear_output

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

num_epochs = 200
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
criterion = nn.MSELoss()

start_time = time.time()  # Start time measurement
model.to(device)
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    #print(f"Epoch: {epoch}")
    for batch_id,(inputs, targets, lengths) in enumerate(train_loader):
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        optimizer.zero_grad()
        lengths = lengths.to('cpu')
        #print(f"Inputs shape: {inputs.shape}")
        outputs = model(inputs, lengths)

        #print(f"Output shape: {outputs.shape}")
        #print(f"Targets shape: {targets.shape}")
        mask = create_mask(lengths,K-1)
        loss = masked_loss(outputs,targets,mask,criterion)


        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item()



    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for batch_id,(inputs, targets, lengths) in enumerate(val_loader):
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            optimizer.zero_grad()
            lengths = lengths.to('cpu')
            outputs = model(inputs, lengths)

            mask = create_mask(lengths,K-1)
            loss = masked_loss(outputs,targets,mask,criterion)

            total_val_loss += loss.item()
            if batch_id%2000==0:
                print(f"\nBatch: {batch_id}")
                #print(f"Input DoAs :{src_batch[-1,:,:2]}")
                #print(f"Target Ouput DoAs: {tgt_batch[-1,:,:]}")
                print(f"RNN hat:{targets[-1,:,:]}")
                #print(f"RNN Input (Last of batch): {src_batch[-1]}")
                print(f"RNN Output (Last of batch): {outputs[-1]}")
                print(f"Loss: {total_val_loss/(batch_id+1)}")

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
    plt.title(f"Training and Validation RMSE Progression [Time Elapsed: {(elapsed_time+epoch_time)/60:.2f}min, LR: {current_lr}, Hidden Size: {hidden_dim}, Num_Layers:{num_layers}]")
    plt.legend()
    plt.grid(True)

    # Annotations
    plt.annotate(f"{avg_train_loss:.4f}", (len(train_losses)-1, train_losses[-1]), textcoords="offset points", xytext=(0,10), ha='center',color='blue')
    plt.annotate(f"{avg_val_loss:.4f}", (len(val_losses)-1, val_losses[-1]), textcoords="offset points", xytext=(0,10), ha='center',color='orange')
    plt.savefig(pwd+f"/Plots/{plotname}",dpi = 400)
    #plt.show()

    torch.save({
     'model_state_dict': model.state_dict(),
     'optimizer_state_dict': optimizer.state_dict(),
     'scheduler_state_dict': scheduler.state_dict(),
     'train_losses': train_losses,
     'val_losses': val_losses,
     'learning_rates': current_lr,
     'elapsed_time': elapsed_time+epoch_time
     }, model_path)

# Save the model checkpoint
torch.save({
 'model_state_dict': model.state_dict(),
 'optimizer_state_dict': optimizer.state_dict(),
 'scheduler_state_dict': scheduler.state_dict(),
 'train_losses': train_losses,
 'val_losses': val_losses,
 'learning_rates': current_lr,
 'elapsed_time': elapsed_time+epoch_time
 }, model_path)