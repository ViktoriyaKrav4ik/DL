# Making a PyTorch Dataset
from torch.utils.data import Dataset, DataLoader
import torch


class LifeExpectancyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # return size of a dataset
        return len(self.y)

    def __getitem__(self, idx):
        # supports indexing using dataset[i] to get the ith row in a dataset

        X = torch.tensor(self.X[idx], dtype=torch.float32)
        # One-hot-encoding over y for the correct loss propagation
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return X, y
