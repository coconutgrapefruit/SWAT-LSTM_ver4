import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):

        super(Model1, self).__init__()

        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.dropout_rate = dropout_rate

        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, dropout=0.2,
                            num_layers=2, bias=True, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=128, hidden_size=32,
        #                      num_layers=1, bias=True, batch_first=True)
        # self.lstm3 = nn.LSTM(input_size=32, hidden_size=8,
        #                      num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=64, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (h_n, c_n) = self.lstm1(x)

        pred = self.fc1(self.dropout(self.relu(h_n[-1, :, :])))
        pred = self.fc2(self.dropout(self.relu(pred)))
        return pred
