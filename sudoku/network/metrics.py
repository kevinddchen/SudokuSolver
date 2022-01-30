import torch

class Mean(object):

    '''Aggregate mean. Inputs are 0D or 1D tensors.'''

    def __init__(self):
        self.s = torch.tensor(0).float()
        self.N = torch.tensor(0).float()

    def accumulate(self, x: torch.Tensor):
        if len(x.size()) == 0:
            self.s += x
            self.N += 1
        else:
            self.s += torch.sum(x)
            self.N += x.size()[0]

    def result(self) -> torch.float:
        return self.s / self.N
