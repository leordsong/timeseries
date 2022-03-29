from typing import Tuple
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(self, input_len, input_dim: int, hidden_dims:Tuple[int], outputs:Tuple[int]):
        super(CNNEncoder, self).__init__()

        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim, self.hidden_dim2 = hidden_dims
        self.output_len, self.output_dim = outputs

        self.cnn = nn.Conv1d(self.input_len, self.output_len, self.input_dim, padding='same')
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=2, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.hidden_dim2, self.output_dim)
        
        
    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.gru(x)
        x = self.linear(x)
        x = self.relu(x)
        return self.output(x)
