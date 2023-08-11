import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


metrics = {
    'Mean Squared Error': mean_squared_error
}