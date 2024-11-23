import torch
import os
import numpy as np
import sys
import io
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/ioannis.mallioras/PycharmProjects/TRANSFORMER')
from dataset_functions import *
from lstm_functions import*
# Save the current stdout so that we can revert sys.stdout after we are done
original_stdout = sys.stdout


device='cpu'

class DoAModel_MTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, max_sequence_length=5, dropout = 0.0):
        super(DoAModel_MTM, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer to predict elevation and azimuth

    def forward(self, x, lengths=None):
        if lengths==None:
            output, (hidden, cell) = self.lstm(x)
            output = self.fc(output)
            output = torch.tanh(output)
            return output
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # Forward pass through LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # Unpack the sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Applying the fully connected layer to each time step
        output = self.fc(output)

        if output.shape[1] < self.max_sequence_length:
            temp_pad = torch.zeros((x.shape[0], 1, 2)).to(output.device)
            output = torch.cat((output, temp_pad), dim=1)
        # Applying tanh activation to ensure output is within the desired range [-1,1]
        output = torch.tanh(output)
        return output




# INPUT PREPROCESSING
x = np.array(x, dtype= np.single)
N_SIGNALS = 3
N_STEPS = 5
total_input = np.zeros((N_SIGNALS, N_STEPS, 2))

for idx in range(N_SIGNALS):
    tmp_az, tmp_el = transform_elevation_azimuth(x[2:,2*idx], x[2:,2*idx+1])
    total_input[idx,:,:] = np.array([tmp_az, tmp_el]).T

total_input = torch.from_numpy(total_input).float()
az_min, az_max = -89.99993133544922, 89.99988555908203
el_min, el_max = -63.164306640625, 58.51708221435547

# Angles
denorm_az = torch.tensor([az_min, az_max])
denorm_el = torch.tensor([el_min, el_max])

# Block printing
sys.stdout = io.StringIO()

total_input[:,:,0] = normalize_data(total_input[:,:,0], az_min, az_max)
total_input[:,:,1] = normalize_data(total_input[:,:,1],el_min, el_max)

# Restore stdout to original
sys.stdout = original_stdout

# Import LSTM-MTO
input_dim = 2
hidden_dim = 320
num_layers = 3
model_path = '/Users/ioannis.mallioras/PycharmProjects/TRANSFORMER/RESULTS/lstm_mto_checkpoint1.pth'
# Initialize the model
model_lstm_mto = DoAModel_MTO(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers).to(device)
checkpoint = torch.load(model_path, map_location="cpu")
model_lstm_mto.load_state_dict(checkpoint['model_state_dict'])

# Import LSTM-MTM
hidden_dim = 608
num_layers = 1
model_path = '/Users/ioannis.mallioras/PycharmProjects/TRANSFORMER/RESULTS/lstm_mtm_checkpoint3.pth'

# Initialize the model
model_lstm_mtm = DoAModel_MTM(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers).to(device)
checkpoint = torch.load(model_path, map_location="cpu")
model_lstm_mtm.load_state_dict(checkpoint['model_state_dict'])

# LSTM_MTO OUTPUT
total_input = total_input.to(device)
out_mto = model_lstm_mto(total_input) # Forward pass
out_mto = out_mto.detach().cpu()

out_az_mto = denormalize_data(out_mto[:,0],denorm_az[0],denorm_az[1])
out_el_mto = denormalize_data(out_mto[:,1],denorm_el[0],denorm_el[1])
out_az_mto,out_el_mto = inverse_transform_elevation_azimuth(out_az_mto.tolist(),out_el_mto.tolist())
out_mto = np.array([out_az_mto,out_el_mto]).T
#print(f"MTO Out: AZ:{out_az_mto}, EL:{out_el_mto}")

# LSTM_MTM OUTPUT
N = 3  # Total records
K = 5  # Timesteps per record
# Create a tensor of shape (N,) where each element is K
lengths = torch.full((N,), K)
lengths = lengths.to('cpu')
out_mtm = model_lstm_mtm(total_input) # Forward pass
out_mtm = out_mtm[:,K-1,:]
out_mtm = out_mtm.detach().cpu()

out_az_mtm = denormalize_data(out_mtm[:,0],denorm_az[0],denorm_az[1])
out_el_mtm = denormalize_data(out_mtm[:,1],denorm_el[0],denorm_el[1])
out_az_mtm,out_el_mtm = inverse_transform_elevation_azimuth(out_az_mtm.tolist(),out_el_mtm.tolist())
out_mtm = np.array([out_az_mtm,out_el_mtm]).T
#print(f"MTM Out: AZ:{out_az_mtm}, EL:{out_el_mtm}")
output = out_mto[0].tolist() ,out_mto[1].tolist() ,out_mto[2].tolist(), out_mtm[0].tolist(),out_mtm[1].tolist(), out_mtm[2].tolist()
