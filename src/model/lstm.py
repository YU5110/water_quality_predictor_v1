from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """与训练脚本 `LSTM模型预测污水厂出水水质.py` 中保持一致的网络结构。"""

    def __init__(self, input_dim: int, hidden_dim: int, n2: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_dim, batch_first=True))
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_dim, hidden_dim, batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, n2)
        self.fc2 = nn.Linear(n2, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "LSTMModel":
        return cls(
            input_dim=len(metadata["features"]),
            hidden_dim=int(metadata["hidden_dim"]),
            n2=int(metadata["n2"]),
            dropout=float(metadata["dropout"]),
            num_layers=int(metadata.get("num_layers", 2)),
        )
