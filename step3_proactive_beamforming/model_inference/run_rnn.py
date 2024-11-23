# Import required libraries
import torch
import torch.nn as nn

# Function to de-normalize output data
def denorm(y, x_min, x_max):
    """
    De-normalizes the input tensor `y` using the specified min and max values.

    Parameters:
    - y: Normalized input tensor.
    - x_min: Minimum value of the original data range.
    - x_max: Maximum value of the original data range.

    Returns:
    - De-normalized tensor.
    """
    return y * (x_max - x_min) + x_min

# Define the RNN model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type, dropout):
        """
        Initializes an RNN-based model, supporting both GRU and LSTM.

        Parameters:
        - input_size: Number of input features per time step.
        - hidden_size: Number of hidden units in each recurrent layer.
        - num_layers: Number of recurrent layers.
        - num_classes: Number of output features (final layer output size).
        - rnn_type: Type of recurrent layer ('GRU' or 'LSTM').
        - dropout: Dropout probability for regularization.
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Define GRU or LSTM layers based on the chosen recurrent type
        if self.rnn_type == 'GRU':
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the RNN.

        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
        - Output tensor of shape (batch_size, num_classes).
        """
        # Initialize hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Hidden state
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Cell state (only for LSTM)

        # Pass input through the chosen recurrent layer
        if self.rnn_type == 'GRU':
            out, _ = self.gru(x, h0)
        elif self.rnn_type == 'LSTM':
            out, _ = self.lstm(x, (h0, c0))

        # Use the output from the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Pass through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, num_classes)

        return out

# Recurrent model hyperparameters
TIMESTEP_INPUT_SIZE = 2  # Number of input features per time step
OUTPUT_SIZE = 128  # Number of output features (final layer size)
DROPOUT = 0.0  # Dropout probability
RNN_TYPE = 'LSTM'  # Type of recurrent layer ('GRU' or 'LSTM')
HIDDEN_SIZE = 464  # Number of hidden units per recurrent layer
N_LAYERS = 4  # Number of recurrent layers

# Initialize the RNN model
model = RNN(
    input_size=TIMESTEP_INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=N_LAYERS,
    num_classes=OUTPUT_SIZE,
    rnn_type=RNN_TYPE,
    dropout=DROPOUT
)

# Load pre-trained model weights
model.load_state_dict(torch.load("LSTM_PRETRAINED.pt", map_location=torch.device('cpu')))

# Prepare sample input
input = torch.zeros((1, 3, 2))  # Input shape: (batch_size, sequence_length, input_size)

# Populate input with example data
input[0, :, 0] = torch.tensor(x[0:3])  # Populate elevation data
input[0, :, 1] = torch.tensor(x[3:])  # Populate azimuth data

# Define normalization parameters
th_max, th_min = 59.9, 0.0  # Elevation range
ph_max, ph_min = 359.9, 0.0  # Azimuth range

# Uncomment the following line for debugging normalized inputs
# print(f"ELEVATION: {denorm(input[0,:,0], th_min, th_max)}  AZIMUTH: {denorm(input[0,:,1], ph_min, ph_max)}")

# Evaluate the model
model.eval()  # Set the model to evaluation mode
output = model(input)  # Perform inference
output = output.flatten()  # Flatten the output tensor

# Scale output to the range [-1, 1]
output = ((output.detach().numpy()) * 2) - 1

# Convert output to a list
output = output.tolist()
