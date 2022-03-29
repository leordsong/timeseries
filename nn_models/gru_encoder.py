from typing import Tuple
import torch.nn as nn


class GRUEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims:Tuple[int], outputs:Tuple[int]):
        super(GRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim, self.hidden_dim2 = hidden_dims
        self.output_len, self.output_dim = outputs

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=2, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.hidden_dim2, self.output_dim)
        
        
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -self.output_len:, :]
        x = self.linear(x)
        x = self.relu(x)
        return self.output(x)


gru = GRUEncoder(3, (64, 32), (5, 2))
print(gru)
import torch
gru(torch.rand(8, 20, 3))
