import torch.nn as nn
import torch

class DeepAR(nn.Module):

    def __init__(self, input_dim, hidden_dims):

        super(DeepAR, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dims,
                            num_layers=2,
                            bias=True,
                            batch_first=True)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         start, end = n // 4, n // 2
        #         bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(hidden_dims, 1)
        self.distribution_presigma = nn.Linear(hidden_dims, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        # use h from all three layers to calculate mu and sigma
        mu = self.distribution_mu(output)

        pre_sigma = self.distribution_presigma(output)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return torch.squeeze(mu), torch.squeeze(sigma)

    def predict(self, x):
        mu, sigma = self.forward(x)
        return torch.normal(mu, sigma)


def loss_fn(mu, sigma, y_true):
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(y_true)
    return -torch.mean(likelihood)
