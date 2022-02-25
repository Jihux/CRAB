from torch import nn
import torch.nn.functional as F

from utils.custom_models.layers import drop


class lenet(nn.Module):

    def __init__(self, **kwargs):
        super(lenet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=kwargs['drop_rate'])

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):

        if x.shape[1] == 2:
            x = x[:, :1, ...]

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x_ = F.relu(self.fc2(x))
        x = self.dropout(x_)
        x = self.fc3(x_)
        if with_latent:
            return x, x_
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
