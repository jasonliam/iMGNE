#!python3

"""
LSTM data flow (ref: http://colah.github.io/posts/2015-08-Understanding-LSTMs)
NOTE: using m for LSTM dimension, n for input dimension
- input layer (n+m): horizontally concatenated h(t-1)(m) and X(t)(n)
- forget gate (m): linear layer w/sigmoid; input from input layer (n+m);
  multiplied to cell state (m)
- learning gate (m): linear layer w/tanh; input from input layer (n+m)
- update gate (m): linear layer w/sigmoid, input from input layer (n+m)
- learning delta (m): learning gate multiplied with update gate; added to
  cell state (m)
- output gate (m): linear layer w/sigmoid, input from input layer
- X(t+1) and h(t) (m): cell state passed through tanh then multiplied with
  output gate

NOTE that X(t)(n) and X(t+1)(m) aren't necessarily of the same dimension;
  thus it's necessary to append a linear layer after the LSTM to restore its output to input space, before feeding the output back to the LSTM input. The linear layer should learn the necessary mapping to correctly rebase the LSTM
  outputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBasic(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1):
        super(LSTMBasic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Encoding layer
        self.encode = nn.Linear(input_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers)

        # Decoding layer: decode LSTM output back to input space
        self.decode = nn.Linear(hidden_dim, input_dim)

        # init the state tuple for LSTM
        self.curr_state = self.init_hidden()

    def init_hidden(self, random=False):
        if random:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
            c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        else:
            # init hidden states and cell states of the LSTM to zero
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            h0, c0 = h0.to(torch.device("cuda")), c0.to(torch.device("cuda"))
        return (h0, c0)

    def forward(self, chunk, prev_state):

        self.lstm.flatten_parameters()

        # encode inputs
        chunk = self.encode(chunk)

        # run the LSTM, get outputs and updates
        # Note: expects the data to already be batched along axis 1
        output, self.curr_state = self.lstm(chunk, prev_state)

        # decode LSTM outputs back to input space
        output = self.decode(output)

        return output, self.curr_state


# Linear feature extraction before LSTM
# Source: https://arxiv.org/abs/1612.04928, FC model
class LSTMFC(LSTMBasic):

    def __init__(self, input_dim, fc_dim, hidden_dim, num_layers=1, batch_size=1, dropout_p=0.5):
        super(LSTMFC, self).__init__(
            input_dim, hidden_dim, num_layers, batch_size)

        self.dropout_p = dropout_p

        # FC layer for feature extraction
        self.fc = nn.Linear(input_dim, fc_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        self.encode = nn.Linear(fc_dim, hidden_dim)

    def forward(self, chunk, prev_state):

        self.lstm.flatten_parameters()

        # linear feature extraction
        chunk = self.fc(chunk)

        # dropout
        chunk = self.dropout(chunk)

        # run the LSTM, get outputs and updates (hidden states i.e. recurrent input, and cell states)
        output, self.curr_state = self.lstm(chunk, prev_state)

        # decode LSTM outputs back to input space
        output = self.decode(output)

        return output, self.curr_state


# CNN feature extraction before LSTM
class LSTMCNN(LSTMBasic):

    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1, decoder="vanilla"):
        super(LSTMCNN, self).__init__(
            input_dim, hidden_dim, num_layers, batch_size)

        self.decoder = decoder

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm_input_dim = int(self.input_dim/2/2/2/2/2/2/2*128)
        self.lstm = nn.LSTM(self.lstm_input_dim,
                            hidden_dim, num_layers=num_layers)

        # CNN layers for decoding
        if self.decoder == "cnn":
            self.lstm_decode = nn.Linear(hidden_dim, self.lstm_input_dim)
            self.pad1 = nn.Upsample(scale_factor=2)
            self.pad2 = nn.Upsample(scale_factor=2)
            self.conv5_t = nn.ConvTranspose1d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv4_t = nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv3_t = nn.ConvTranspose1d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv2_t = nn.ConvTranspose1d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv1_t = nn.ConvTranspose1d(16, 1, kernel_size=7, padding=3)
            self.bn5_t = nn.BatchNorm1d(128)
            self.bn4_t = nn.BatchNorm1d(64)
            self.bn3_t = nn.BatchNorm1d(32)
            self.bn2_t = nn.BatchNorm1d(16)
        elif self.decoder == "2fc":
            self.decode1 = nn.Linear(hidden_dim, input_dim)
            self.decode2 = nn.Linear(input_dim, input_dim)

    def forward(self, chunk, prev_state):

        self.lstm.flatten_parameters()

        # CNN feature extraction
        # concat all minibatches to fit Conv1d's shape req
        x = chunk.view(-1, 2, self.input_dim)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.max_pool2(x)
        # break out the minibatches again for LSTM
        x = x.view(chunk.shape[0], chunk.shape[1], -1)

        # run the LSTM, get outputs and updates (hidden states i.e. recurrent input, and cell states)
        output, self.curr_state = self.lstm(x, prev_state)

        # decode LSTM outputs back to input space
        if self.decoder == "cnn":
            output = self.lstm_decode(output)
            # concat minibatches to make space for conv_t
            output = output.view(-1, 128, self.hidden_dim//128)
            output = self.pad1(output)
            output = F.relu(self.bn5_t(self.conv5_t(output)))
            output = F.relu(self.bn4_t(self.conv4_t(output)))
            output = F.relu(self.bn3_t(self.conv3_t(output)))
            output = F.relu(self.bn2_t(self.conv2_t(output)))
            output = self.pad2(output)
            output = self.conv1_t(output)
            # break out minibatches again
            output = output.view(chunk.shape[0], chunk.shape[1], -1)
        elif self.decoder == "2fc":
            output = F.tanh(self.decode1(output))
            output = self.decode2(output)
        else:
            output = self.decode(output)

        return output, self.curr_state
