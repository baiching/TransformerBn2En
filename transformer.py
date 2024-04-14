import math
import torch
import torch.nn as nn
import torch.nn.functional as f

#initialization method
def initialize_weights(x):
    nn.init.xavier_normal_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

#FeedForward block for the transformer
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, x):
        x = self.layer2(self.dropout(self.relu(self.layer1(x))))
        return x

