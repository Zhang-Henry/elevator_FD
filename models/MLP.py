import torch
import torch.nn as nn

class MLP2(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # 添加输入层到隐藏层之间的线性层
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size

        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(900, 512),
        nn.ReLU(),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        # nn.Linear(64, 32),
        # nn.ReLU(),
        # nn.Linear(32, 16),
        # nn.ReLU()
        )


  def forward(self, x):
    x = self.encoder(x)
    return x

