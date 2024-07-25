import torch

def exp_scale(x):
    return 1 - torch.exp(-x)

class TSGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, scaling='exp'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_gate = torch.nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.r_gate = torch.nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.h_gate = torch.nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.scaling = exp_scale if scaling == 'exp' else torch.nn.Identity()
    
    def forward(self, x, dt, h=None):
        # x is B x N
        # h is B x H
        # dt is B or B x 1
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        z = torch.sigmoid(self.z_gate(torch.cat([x, h], dim=-1)))
        r = torch.sigmoid(self.r_gate(torch.cat([x, h], dim=-1)))
        h_hat = torch.tanh(self.h_gate(torch.cat([x, r * h], dim=-1)))
        dtz = self.scaling(dt) * z
        return (1 - dtz) * h + dtz * h_hat