import torch
import torch.nn as nn

class NNetwork(nn.Module):
    
    def __init__(self, dim_input, dim_output, dim_hidden, n_layers):
        super().__init__()
        activation = nn.Tanh
        self.input_layer = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            activation()
        )
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                activation()
            )
            for _ in range(n_layers - 1)
        ])
        self.output_layer = nn.Linear(dim_hidden, dim_output)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x