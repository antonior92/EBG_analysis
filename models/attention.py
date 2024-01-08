import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eegnet import EEGNet


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class AttentionEEGNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eegnet = EEGNet(**kwargs)
        self.attention = SelfAttention(kwargs['n_channels'])

    def forward(self, x):
        x = x.squeeze().permute((0, 2, 1))
        attn_weights = self.attention(x)
        attn_weights = attn_weights.permute((0, 2, 1)).unsqueeze(dim=1)
        out = self.eegnet(attn_weights)

        return out

class AttentionMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential()