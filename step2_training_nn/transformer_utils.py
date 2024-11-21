import torch
import torch.nn as nn
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
        x = x + self.pe[:, x.size(0)]
        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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
        src2 = self.linear2(self.dropout(torch.tanh(self.linear1(src))))
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


class ExtendedContextualEmbedding2(nn.Module):
    def __init__(self, d_model, num_binary_features, num_continuous_features):
        super(ExtendedContextualEmbedding2, self).__init__()
        self.binary_embedding = nn.Embedding(num_binary_features, d_model)
        self.continuous_projection = nn.Linear(num_continuous_features, d_model)

    def forward(self, binary_data, continuous_data):
        binary_embedded = self.binary_embedding(binary_data)
        continuous_embedded = self.continuous_projection(continuous_data)
        combined_context = binary_embedded + continuous_embedded
        return combined_context


class contextDOAEncoder4(nn.Module):
    def __init__(self, input_dim, context_size, additional_features_dim, d_model, nhead, num_encoder_layers, dropout):
        super(contextDOAEncoder4, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, int(d_model / 2))  # DoA features projection
        self.extended_context_embedding = ExtendedContextualEmbedding2(int(d_model / 2), context_size,
                                                                       additional_features_dim)
        self.pos_encoder = PositionalEncoding(int(d_model / 2), dropout)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_proj = nn.Linear(d_model, 2)  # Output prediction for DoA

    def forward(self, src, context, continuous_features):
        src = self.input_proj(src) * math.sqrt(self.d_model / 2)  # Scale embeddings
        context_emb = self.extended_context_embedding(context, continuous_features)  # Get combined context embedding
        src = self.pos_encoder(src)
        context_emb = self.pos_encoder(context_emb)
        src = torch.cat([src, context_emb], dim=-1)  # Concatenate along the feature dimension
        output = self.transformer_encoder(src)
        output = torch.tanh(self.output_proj(output[-1]))  # Predict next DoA
        return output
