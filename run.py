# Set the random seed for reproducible experiments
import torch
import logging
from data_loader import LifeExpectancyDataset
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from model import LinearModel
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from engine import train_and_evaluate
from model import metrics
import numpy as np
import pickle


@dataclass
class params():
    cuda = torch.cuda.is_available()
    save_summary_steps = 1

    model_dir = 'C:/Users/la.vikravchenko/PycharmProjects/pythonProject/experiments'

    learning_rate = 0.05
    num_epochs = 120
    batch_size = 64


torch.manual_seed(42)
if params.cuda:
    torch.cuda.manual_seed(42)

# Specify a computing device
device = 'cuda' if params.cuda else None

# Set the logger
# set_logger(os.path.join(params.model_dir, 'train.log'))

logging.info("Loading the datasets...")

# Create datasets
df = pd.read_csv('data/df_prepared.csv')
X_norm = df.drop('Life expectancy ', axis=1)
y = df['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(X_norm.values, y.values, test_size=0.33, random_state=42)
train_dataset = LifeExpectancyDataset(X_train, y_train)
test_dataset = LifeExpectancyDataset(X_test, y_test)

# Create dataloaders
train_dataloader = DataLoader(train_dataset,
                              batch_size=params.batch_size,
                              shuffle=True
                              )

test_dataloader = DataLoader(test_dataset,
                             batch_size=params.batch_size,
                             shuffle=False
                             )

logging.info("- done.")

# Define the model and optimizer
model = LinearModel(in_dim=X_norm.shape[1], out_dim=X_norm.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

# Define loss function
loss_fn = nn.MSELoss().to(device)

# Train the model
logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
train_stats, val_stats = train_and_evaluate(model, train_dataloader, test_dataloader,
                                            optimizer, loss_fn, metrics, params)

# Open a file and use dump()
with open('experiments/train_stats.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(train_stats, file)

with open('experiments/val_stats.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(val_stats, file)