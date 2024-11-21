import torch
import torch.nn as nn

class GRUModel_MTO(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2, num_layers=2, dropout=0.0):
        super(GRUModel_MTO, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Mapping to 2 DOA values

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # lengths: [batch_size] containing the actual lengths of sequences for masking
        packed_output, _ = self.gru(x)
        # Apply linear layer to each time step
        output = self.fc(packed_output[:,-1,:])
        output = torch.tanh(output)  # Note: adjust this based on your normalization strategy
        return output




class GRUModel_MTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_layers=2, max_sequence_length=5,dropout=0.0):
        super(GRUModel_MTM, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer to predict elevation and azimuth

    def forward(self, x, lengths=None):
        if lengths == None:
            output, hidden = self.gru(x)
            output = self.fc(output)
            output = torch.tanh(output)
            return output
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
