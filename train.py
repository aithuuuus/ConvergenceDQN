import gym
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from lib.wrappers import make_env
from lib.model import ShallowNet
from lib.parser import parse_args
from lib.policy.cdqn import CDQNPolicy
from lib.collector import Collector
from lib.offpolicy import offpolicy_trainer

import tianshou as ts
from tianshou.utils import TensorboardLogger

args = parse_args()

# Make the envs
sample_env = make_env(**args.__dict__)
obs_sample = sample_env.observation_space.sample()
action_shape = sample_env.action_space.n

train_envs = ts.env.SubprocVectorEnv(
    [make_env for _ in range(args.train_parallel)])
test_envs = ts.env.SubprocVectorEnv(
    [make_env for _ in range(args.test_parallel)])

# Make net
if args.is_shallow:
    net = ShallowNet(obs_sample, action_shape)
else:
    pass
optim = torch.optim.Adam(
    net.parameters(), 
    lr=args.learning_rate)

if args.origin:
    policy = ts.policy.DQNPolicy(net, optim, \
        discount_factor=args.discount_rate, 
        estimation_step=args.obs_merge, 
        target_update_freq=args.sync_steps, 
        max_step=args.max_step, 
    )
else:
    policy = CDQNPolicy(net, optim, \
        discount_factor=args.discount_rate, 
        estimation_step=args.obs_merge, 
        target_update_freq=args.sync_steps, 
        max_step=args.max_step, 
    )

train_collector = Collector(
    policy, train_envs, \
    ts.data.VectorReplayBuffer(args.buffer_size, args.batch_size), 
    exploration_noise=True)
test_collector = ts.data.Collector(
    policy, test_envs, exploration_noise=False)

writer = SummaryWriter('log/'+args.log_path)
logger = TensorboardLogger(writer)

result = offpolicy_trainer(
        policy, train_collector, test_collector, 
        max_epoch=50, step_per_epoch=10000, step_per_collect=8, 
        update_per_step=0.1, episode_per_test=3, 
        batch_size=args.batch_size, 
        logger=logger, writer=writer)

if not os.path.exists('./weights/{}'.format(args.weight_path)):
    os.makedirs('./weights/{}'.format(args.weight_path), exist_ok=True)

torch.save(policy.state_dict(), './weights/{}/weights.pth'.format(args.name))
