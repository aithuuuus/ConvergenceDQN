from lib.wrappers import make_env
from lib.dqn_model import DQN
import torch

env = make_env()
obs = env.reset()
net = DQN().to('cuda')
net.load_state_dict(torch.load('2/w214'))

for x in range(9000):
    obs = torch.from_numpy(obs)
    obs = obs.unsqueeze(0)
    Q = net(obs)[0]
    action = Q.argmax().item()
    obs, reward, is_done, info = env.step(action)
    env.render()
    if is_done:
        obs = env.reset()
