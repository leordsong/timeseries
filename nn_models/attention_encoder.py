from typing import Tuple
import torch.nn as nn


class AttentionBlock(nn.Module):

    def __init__(self, input_len:int, input_dim:int, attn_heads:int, hidden_dim:int):
        super(AttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(input_dim, attn_heads, batch_first=True)
        self.norm = nn.BatchNorm1d(input_len)
        self.norm2 = nn.BatchNorm1d(input_len)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        attn_x, _ = self.mha(x, x, x)
        attn_x = self.norm(x + attn_x)

        linear_x = self.linear(attn_x)
        linear_x = self.relu(linear_x)
        linear_x = self.linear2(linear_x)
        y = self.norm2(attn_x + linear_x)
        return y


class AttentionEncoder(nn.Module):

    def __init__(self, inputs:Tuple[int], attn_dims:Tuple[int], hidden_dims:Tuple[int], outputs:Tuple[int]):
        super(AttentionEncoder, self).__init__()

        self.input_len, self.input_dim = inputs
        self.attn_dims, self.attn_heads = attn_dims
        self.hidden_dim, self.hidden_dim2 = hidden_dims
        self.output_len, self.output_dim = outputs

        self.linear = nn.Linear(self.input_dim, self.attn_dims)
        self.attn_block = AttentionBlock(self.input_len, self.attn_dims, self.attn_heads, self.hidden_dim)

        self.mha = nn.MultiheadAttention(self.attn_dims, self.attn_heads, batch_first=True)
        self.linear2 = nn.Linear(self.attn_dims, self.hidden_dim2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.hidden_dim2, self.output_dim)
        
        
    def forward(self, x):
        x = self.linear(x)
        x = self.attn_block(x)
        x, att_weight = self.mha(x[:, -self.output_len:, :], x, x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.output(x)
