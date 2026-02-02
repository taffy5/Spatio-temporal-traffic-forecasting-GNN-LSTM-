import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GNNLSTM(nn.Module):
    def __init__(self, num_nodes, in_features=1, hidden=64):
        super().__init__()

        self.gcn1 = GCNConv(in_features, hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

        self.fc = nn.Linear(hidden, 1)
        self.num_nodes = num_nodes

    def forward(self, x, edge_index):
        # x: [batch, seq_len, nodes]

        batch, seq_len, nodes = x.shape
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :].reshape(-1, 1)

            h = self.gcn1(xt, edge_index)
            h = torch.relu(h)
            h = self.gcn2(h, edge_index)

            outputs.append(h.view(batch, nodes, -1))

        gnn_out = torch.stack(outputs, dim=1)

        lstm_out, _ = self.lstm(gnn_out)
        out = self.fc(lstm_out[:, -1])

        return out.squeeze()
