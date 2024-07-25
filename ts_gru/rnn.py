import torch
from .cell import TSGRUCell

class TSGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, scaling='exp'):
        super().__init__()
        self.cells = torch.nn.ModuleList(
            [TSGRUCell(input_size, hidden_size, bias=bias, scaling=scaling)] + [
            TSGRUCell(hidden_size, hidden_size, bias=bias, scaling=scaling)
            for _ in range(num_layers-1)
        ])
    
    def forward(self, x, dt, h_init=None, return_sequences=True):
        # x is B x T x N
        # dt is B x T
        h = h_init 
        if h is None:
            h = torch.zeros(
                x.size(0),
                len(self.cells),
                self.cells[0].hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        steps = x.size(1)
        sequences = torch.empty(
            x.size(0),
            steps,
            h.size(-1),
            device=x.device,
            dtype=x.dtype,
        )
        for step in range(steps):
            h = self.step(x[:, step], dt[:, step], h)
            sequences[:, step] = h[:, -1]
        if return_sequences:
            return sequences
        else:
            return h[:, -1]
    
    def step(self, x, dt, h):
        # x is B x N
        # dt is B or B x 1
        # h is B x L x H
        new_h = torch.empty_like(h, dtype=h.dtype, device=h.device)
        inp = x # first cells input is x
        for i, cell in enumerate(self.cells):
            new_h[:, i] = cell(inp, dt, h[:, i])
            inp = new_h[:, i] # next cells input is the output of this cell
        return new_h
