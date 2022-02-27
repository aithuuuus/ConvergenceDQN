import numpy as np
from copy import deepcopy

import torch
from torch import nn


class ShallowNet(nn.Module):
    def __init__(
        self, 
        obs_sample, 
        action_shape, 
        device='cuda', 
    ):
        super(ShallowNet, self).__init__()
        self.observation_sample = torch.from_numpy(
            np.expand_dims(obs_sample, 0))
        self.action_shape = action_shape
        self.device = device

        self.conv = nn.Sequential(
                nn.Conv2d(obs_sample.shape[0], 32, kernel_size=8, stride=4), 
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), 
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU())

        _out_conv = deepcopy(self.conv)(self.observation_sample)
        self._out_conv_size = np.prod(_out_conv.shape[1:])

        self.fc = nn.Sequential(
                nn.Linear(self._out_conv_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_shape)
                )
        self.to(device)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(self.device)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, state


if __name__ == "__main__":
    from wrappers import make_env
    env = make_env()
    obs = env.reset()
    net = ShallowNet(obs, env.action_space.n)
    obs = obs.repeat(32, axis=0).reshape(32, *env.observation_space.shape)
    Q = net(obs)[0]
    print(Q.shape)
