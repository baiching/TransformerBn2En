# code source: https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/model/transformer.py#L201

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

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super().__init__(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

        initialize_weights(self.linear_q)
        initialize_weights(self.linear_k)
        initialize_weights(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)
        initialize_weights(self.output_layer)

    def
