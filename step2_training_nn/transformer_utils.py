import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=28):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, d_model]
        """
        #print(x.size(0))
        x = x + self.pe[:,x.size(0)]
        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(F.tanh(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output


class DOAEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dropout):
        super(DOAEncoder, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward = 4*d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_proj = nn.Linear(d_model, 2)  # Output layer for DOA prediction

    def forward(self, src):
        """
        src: Tensor, shape [seq_len, batch_size, input_dim]
        """
        # Project input dimension to d_model
        src = self.input_proj(src) * math.sqrt(self.d_model)  # Scale embeddings
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_proj(output[-1])  # Predict next DOA based on the last encoder output
        return output

class contextDOAEncoder(nn.Module):
    def __init__(self, input_dim=6, context_size=2, context_dim=16, d_model=512, nhead=8, num_encoder_layers=6,
                 dropout=0.1):
        super(contextDOAEncoder, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim + context_dim, d_model)  # Adjust for concatenated context
        self.context_embedding = nn.Embedding(num_embeddings=context_size, embedding_dim=context_dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_proj = nn.Linear(d_model, 2)  # Output layer for DOA prediction

    def forward(self, src, context):
        """
        src: Tensor, shape [seq_len, batch_size, input_dim]
        context: Tensor, shape [seq_len, batch_size] (contains indices for context, e.g., 0 for LoS and 1 for NLoS)
        """
        # Context embedding for each time step
        context_emb = self.context_embedding(context)  # [seq_len, batch_size, context_dim]
        # Concatenate context embedding with input data
        src = torch.cat([src, context_emb], dim=-1)  # Ensure src and context_emb are compatible
        src = self.input_proj(src) * math.sqrt(self.d_model)  # Scale embeddings
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_proj(output[-1])  # Predict next DOA based on the last encoder output
        return output



class ExtendedContextualEmbedding(nn.Module):
    def __init__(self, context_size, embedding_dim, additional_features_dim):
        super(ExtendedContextualEmbedding, self).__init__()
        # Context for binary features like LoS/NLoS
        self.context_embedding = nn.Embedding(num_embeddings=context_size, embedding_dim=embedding_dim)
        # Linear layer to embed continuous features like velocity and acceleration
        self.continuous_feature_embedding = nn.Linear(additional_features_dim, embedding_dim)

    def forward(self, context, continuous_features):
        # context: [batch_size, seq_len] - Binary features
        # continuous_features: [batch_size, seq_len, feature_dim] - Continuous features like velocity and acceleration
        context_emb = self.context_embedding(context)  # Embed binary context
        continuous_features_emb = self.continuous_feature_embedding(continuous_features)  # Embed continuous features
        # Combine the embeddings, could be by addition, concatenation etc. Here we add them.
        combined_embedding = context_emb + continuous_features_emb
        return combined_embedding

# Adjust the transformer model to use the ExtendedContextualEmbedding
class contextDOAEncoder2(nn.Module):
    def __init__(self, input_dim, context_size, additional_features_dim, d_model, nhead, num_encoder_layers, dropout):
        super(contextDOAEncoder2, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)  # DoA features projection
        self.extended_context_embedding = ExtendedContextualEmbedding(context_size, d_model, additional_features_dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_proj = nn.Linear(d_model, 2)  # Output prediction for DoA

    def forward(self, src, context, continuous_features):
        src = self.input_proj(src) * math.sqrt(self.d_model)  # Scale embeddings
        context_emb = self.extended_context_embedding(context, continuous_features)  # Get combined context embedding
        src += context_emb  # Combine input features with context
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_proj(output[-1])  # Predict next DoA
        return output



class ExtendedContextualEmbedding2(nn.Module):
    def __init__(self, d_model, num_binary_features, num_continuous_features):
        super(ExtendedContextualEmbedding2, self).__init__()
        # Optional: Embedding layer for binary features if you choose to embed them
        self.binary_embedding = nn.Embedding(num_binary_features, d_model)  # if needed
        self.continuous_projection = nn.Linear(num_continuous_features, d_model)
        # self.binary_projection = nn.Linear(num_binary_features, d_model)

    def forward(self, binary_data, continuous_data):
        # Handle binary features
        # If binary features are embedded:
        binary_embedded = self.binary_embedding(binary_data)
        # If directly using binary features, project them:
        # binary_embedded = self.binary_projection(binary_data.float())  # Ensure binary data is float

        # Handle continuous features
        continuous_embedded = self.continuous_projection(continuous_data)
        # Combine embedded features
        combined_context = binary_embedded + continuous_embedded  # Element-wise addition or concatenation
        return combined_context


class contextDOAEncoder4(nn.Module):
    def __init__(self, input_dim, context_size, additional_features_dim, d_model, nhead, num_encoder_layers, dropout):
        super(contextDOAEncoder4, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, int(d_model/2))  # DoA features projection
        self.extended_context_embedding = ExtendedContextualEmbedding2(int(d_model/2), context_size,
                                                                       additional_features_dim)
        self.pos_encoder = PositionalEncoding(int(d_model/2), dropout)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_proj = nn.Linear(d_model, 2)  # Output prediction for DoA

    def forward(self, src, context, continuous_features):
        src = self.input_proj(src) * math.sqrt(self.d_model/2)  # Scale embeddings
        context_emb = self.extended_context_embedding(context, continuous_features)  # Get combined context embedding
        src = self.pos_encoder(src)
        context_emb = self.pos_encoder(context_emb)
        src = torch.cat([src, context_emb], dim=-1)  # Concatenate along the feature dimension
        output = self.transformer_encoder(src)
        output = torch.tanh(self.output_proj(output[-1]))  # Predict next DoA
        return output
        
class DOAEncoder4(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dropout):
        super(DOAEncoder4, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, int(d_model))  # DoA features projection
        self.pos_encoder = PositionalEncoding(int(d_model), dropout)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_proj = nn.Linear(d_model, 2)  # Output prediction for DoA

    def forward(self, src):
        src = self.input_proj(src) * math.sqrt(self.d_model/2)  # Scale embeddings
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.tanh(self.output_proj(output[-1]))  # Predict next DoA
        return output

