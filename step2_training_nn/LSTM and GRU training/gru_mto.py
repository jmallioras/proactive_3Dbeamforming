import torch
import torch.nn as nn
from transformer_utils import *
from dataset_functions import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
from IPython.display import clear_output
import os
from scipy.io import loadmat
from torch.utils.data import random_split, DataLoader, TensorDataset

# Device config
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    

class GRUModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Mapping to 2 DOA values

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # lengths: [batch_size] containing the actual lengths of sequences for masking
        #print(f"NN Input size:{x.shape}")
        #print(f"Input:{x}")
        #packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(x)
        #output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply linear layer to each time step
        output = self.fc(packed_output[:,-1,:])
        #print(f'Output shape: {output.shape}')
        #print(f"Output: {output}")
        # Normalize outputs to [0, 1] range
        output = torch.tanh(output)  # Note: adjust this based on your normalization strategy
        return output


class DoAModel(nn.Module):
    def __init__(self,input_dim=2, hidden_dim=128, output_dim=2, num_layers=2):
        super(DoAModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer to predict elevation and azimuth

    def forward(self, x):
        # Forward pass through LSTM layer
        # x should be of shape (batch, sequence, features)
        output, (hidden, cell) = self.lstm(x)
        # Take the output from the last time step
        last_time_step = output[:, -1, :]
        # Pass the output through the linear layer
        output = self.fc(last_time_step)
        # Applying tanh activation function to match the output range with your target
        output = torch.tanh(output)
        return output



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
denorm_az = torch.tensor([-89.99993133544922, 89.99988555908203])
denorm_el = torch.tensor([-63.164306640625, 58.51708221435547])

block_dataset[:,:,0] = normalize_data(block_dataset[:,:,0], -89.99993133544922, 89.99988555908203)
block_dataset[:,:,1] = normalize_data(block_dataset[:,:,1],-63.164306640625, 58.51708221435547)

block_dataset = block_dataset[:,:,:2]

trainset = block_dataset[:int(0.9*len(block_dataset))]
valset = block_dataset[int(0.9*len(block_dataset)):]
print(f"Blocked dataset size: {block_dataset.shape[0]}")
# Shuffle the rows of each dataset
trainset = trainset[torch.randperm(trainset.shape[0])]
valset = valset[torch.randperm(valset.shape[0])]


# Assume input_seq, target_seq, and seq_lengths are your batched input sequences, target sequences, and sequence lengths, respectively
#model = GRUModel(input_dim=2, hidden_dim=128, output_dim=2, num_layers=2)
hidden_dim = 160
num_layers = 3
lr = 0.0001
batch_size = 64
model_path = 'gru_checkpoint2.pth'
plotname = 'gru_training_prog2.png'

train_dataset = DOADataset(trainset, device=device)
val_dataset = DOADataset(valset, device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#model = DoAModel(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers)
model = GRUModel(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers)
model.to(device)



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

num_epochs = 500
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
criterion = nn.MSELoss()

start_time = time.time()  # Start time measurement
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    #print(f"Epoch: {epoch}")
    for batch_id,(src_batch, tgt_batch) in enumerate(train_loader):
        #src_batch = src_batch.transpose(0, 1)
        optimizer.zero_grad()  # Clear gradients
        sequence_sizes = torch.ones(src_batch.shape[0])*src_batch.shape[1]
        output = model(src_batch) # Forward pass
        loss = torch.sqrt(criterion(output, tgt_batch[:,-1,:2]))  # RMSE loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item()

        if batch_id%2000==0:
            print(f"\nBatch: {batch_id}")
            #print(f"Input DoAs :{src_batch[-1,:,:2]}")
            #print(f"Target Ouput DoAs: {tgt_batch[-1,:,:]}")
            print(f"RNN hat:{tgt_batch[-1,-1,:]}")
            #print(f"RNN Input (Last of batch): {src_batch[-1]}")
            print(f"RNN Output (Last of batch): {output[-1]}")
            print(f"Loss: {total_loss/(batch_id+1)}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            sequence_sizes = torch.ones(src_batch.shape[0])*src_batch.shape[1]
            output = model(src_batch) # Forward pass

            loss = torch.sqrt(criterion(output, tgt_batch[:,-1,:]))  # RMSE loss
            total_val_loss += loss.item()

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
    plt.savefig(f"Plots/{plotname}",dpi = 400)
    

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