import torch
import torch.nn as nn

class LSTM_MTO_model(nn.Module):
    def __init__(self,input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, dropout = 0.0):
        super(LSTM_MTO_model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
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


class LSTM_MTM_model(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, max_sequence_length=5, dropout = 0.0):
        super(LSTM_MTM_model, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer to predict elevation and azimuth

    def forward(self, x, lengths):
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


def random_masking(block_dataset):
    # Masking
    N = block_dataset.shape[0]  # Total records
    K = block_dataset.shape[1]  # Timesteps per record
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

    return masked_inputs, masked_targets, lengths

def create_mask(lengths, K):
    """ Create a boolean mask from sequence lengths.

    Args:
        lengths (torch.Tensor): A tensor of shape [batch_size] containing sequence lengths.
        K (int): The maximum length of sequences in the batch.

    Returns:
        torch.Tensor: A boolean mask of shape [batch_size, K].
    """
    # Create a range tensor from 0 to K-1
    range_tensor = torch.arange(K)[None, :]  # [1, K]

    # Expand lengths to match the range tensor shape
    lengths = lengths[:, None]  # Reshape to [batch_size, 1]

    # Compare range tensor with expanded lengths tensor
    mask = range_tensor == lengths-1  # [batch_size, K]

    return mask


def masked_loss(outputs, targets, mask, criterion):
    mask = mask.to(outputs.device)
    # Mask outputs and targets
    masked_outputs = outputs * mask.unsqueeze(-1)
    masked_targets = targets * mask.unsqueeze(-1)

    # Compute the loss only on the masked outputs and targets
    loss = torch.sqrt(criterion(masked_outputs, masked_targets))
    return loss.mean()