import gym
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import json
import collections


class FireEnv(gym.Wrapper):
    '''For the env have to fire to start'''
    def __init__(self, env=None):
        super(FireEnv, self).__init__(env)
        assert 'FIRE' in env.unwrapped.get_action_meanings()
        self.Fire = env.unwrapped.get_action_meanings().index('FIRE')

    def reset(self):
        self.env.reset()
        is_done = True
        while is_done:
            obs, reward, is_done, reward = self.env.step(self.Fire)

        return obs


class MakeNewFrame(gym.ObservationWrapper):
    '''Make the obs as a F * F * 1 gray graph'''
    def __init__(self, env=None, F=84):
        super(MakeNewFrame, self).__init__(env)
        self.F = F
        self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(F, F, 1), dtype=np.uint8)

    def observation(self, obs):
        return MakeNewFrame.trans(obs, self.F)

    @staticmethod
    def trans(obs, F=84):
        img = Image.fromarray(obs)
        img = img.convert('L')
        img = img.resize((F, F))
        img = np.array(img)
        img = np.expand_dims(img, -1)
        return img

class RandomPlay(gym.Wrapper):
    '''Randomly play the game to check the wrapper'''
    def random_play(self, time_step=2000):
        fourcc = VideoWriter_fourcc(*'MP42')
        video = VideoWriter('./randomplay.avi', fourcc, 64., \
                (84, 84), 3)
        self.env.reset()
        for i in range(time_step):
            action = self.env.action_space.sample()
            obs, reward, is_done, info = self.step(action)
            obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR)
            obs = obs.astype(np.uint8)
            video.write(obs)

            if is_done:
                self.env.reset()
        self.env.reset()

        video.release()

class MaxAndSkip(gym.Wrapper):
    def __init__(self, env=None, skip=3):
        '''return only every `skip`-th frame'''
        super(MaxAndSkip, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        total_info = []
        for x in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_info.append(info)
            total_reward += reward
            self._obs_buffer.append(obs)
            if done:
                break

        obs = np.max(np.stack(self._obs_buffer), axis=0)
        return obs, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ScaleFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32) / 255

class MoveAxis(gym.ObservationWrapper):
    '''move the axis to fit the pytorch'''
    def __init__(self, env=None):
        super(MoveAxis, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                high=np.moveaxis(self.observation_space.high, -1, 0), 
                low=np.moveaxis(self.observation_space.low, -1, 0)
            )

    def observation(self, obs):
        return np.moveaxis(obs, -1, 0)

class BufferWrapper(gym.ObservationWrapper):
    '''Wrapper the neibor obs'''
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.observation_space = gym.spaces.Box(
                self.observation_space.low.repeat(n_steps, axis=0), 
                self.observation_space.high.repeat(n_steps, axis=0), 
                dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
                self.observation_space.low, dtype=self.dtype
            )
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer = self.buffer.copy()
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(
    task='ALE/Pong-v5', 
    skip_frame=3, 
    frame=84,
    obs_merge=3
):
    env = gym.make(task)
    env = MaxAndSkip(env, skip=skip_frame)
    env = FireEnv(env)
    env = MakeNewFrame(env, frame)
    env = RandomPlay(env)
    env = MoveAxis(env)
    env = ScaleFloatFrame(env)
    env = BufferWrapper(env, obs_merge)
    return env

if __name__ == '__main__':
    env = make_env()
    env.random_play()
    obs = env.reset()
    import matplotlib.pyplot as plt
    plt.imshow(obs[-1], vmin=0, vmax=1)
    plt.show()
