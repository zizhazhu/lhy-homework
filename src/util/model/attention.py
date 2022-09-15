import torch
from torch import nn as nn


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, query):
        x, _ = self.attention.forward(query, query, query, need_weights=False)
        return x


class AttClassifier(nn.Module):
    def __init__(self, att_dim, att_head, att_layer, n_concat, input_dim=39, output_dim=41, dropout=0.0):
        super(AttClassifier, self).__init__()
        self.n_concat = n_concat
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.transformer = torch.nn.Transformer(
            att_dim, att_head, num_encoder_layers=att_layer, num_decoder_layers=att_layer, dropout=dropout,
            batch_first=True)
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, att_dim),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Linear(self.n_concat * att_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.n_concat, self.input_dim)
        att_input = self.input_layer(x)
        att_output = self.transformer(att_input)
        att_output = att_output.reshape(-1, self.att_dim * self.n_concat)
        all_output = self.output_layer(att_output)
        return all_output

    def move_to(self, device):
        self.input_layer.to(device)
        self.output_layer.to(device)
        for layer in self.attention:
            layer.to(device)
        self.pe.to(device)
