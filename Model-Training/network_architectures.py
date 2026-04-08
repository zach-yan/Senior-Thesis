import pyepo
import torch
import torch.nn as nn
import torch.optim as optim
import gurobipy as gp
from gurobipy import GRB

import gurobipy as gp
from gurobipy import GRB
import pyepo
from pyepo.model.grb import optGrbModel


class DirectRewardNet(nn.Module):
    def __init__(self, input_dim=46, T=3, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        last_dim = hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, last_dim), nn.LayerNorm(last_dim), nn.ReLU(),
        )
        self.target_head = nn.Linear(last_dim, T + 1)

    def forward(self, x):
        return self.target_head(self.trunk(x))


class ClassifierNet(nn.Module):
    def __init__(self, input_dim=46, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.num_classes = 4
        last_dim = hidden_dim // 2

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, last_dim), nn.LayerNorm(last_dim), nn.ReLU(),
        )
        # Outputs 4 raw logits for CrossEntropyLoss
        self.target_head = nn.Linear(last_dim, self.num_classes)

    def forward(self, x):
        return self.target_head(self.trunk(x))

class NeuroICUSchedulingModel(optGrbModel):
    def __init__(self, N, T, R):
        self.N = N
        self.T = T
        self.R = R
        super().__init__()

    def _getModel(self):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("NeuroICUScheduling_Rolling", env=env)

        # Flat array of length N * (T + 1)
        x = m.addVars(self.N * (self.T + 1), vtype=GRB.BINARY, name="x")

        def get_x(i, t):
            return x[i * (self.T + 1) + t]

        def get_x_later(i):
            return x[i * (self.T + 1) + self.T]

        # 1. Capacity Constraint R at each timestep t
        for t in range(self.T):
            m.addConstr(gp.quicksum(get_x(i, t) for i in range(self.N)) <= self.R)

        # 2. Mutual Exclusivity: Schedule at most once within T, or defer
        for i in range(self.N):
            m.addConstr(
                gp.quicksum(get_x(i, t) for t in range(self.T)) + get_x_later(i) == 1
            )

        return m, x