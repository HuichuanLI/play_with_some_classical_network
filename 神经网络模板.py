import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_0 = nn.Linear(100, 10, bias=False)
        self.fc_1 = nn.Linear(10, 5, bias=False)
        self.fc_2 = nn.Linear(5, 1, bias=False)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


if __name__ == '__main__':
    model = Model()
    print(model)

    input = torch.rand(100)

    output = model(input)
    print(output)

    summary(model, (100,))
