import torch.nn as nn
import torch.nn.functional as F


class CartPoleQNet(nn.Module):

    def __init__(self, n_out, batchnorm):
        super(CartPoleQNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.head = nn.Linear(448, n_out)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(32)
        self._batchnorm = batchnorm

    def forward(self, x):
        if self._batchnorm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))
