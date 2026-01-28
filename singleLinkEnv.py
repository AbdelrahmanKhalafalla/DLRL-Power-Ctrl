import gymnasium as gym
from gymnasium import spaces
import numpy as np

class wirelessPowerEnv(gym.Env):
    def __init__(self , seed=42):
        super().__init__()
        self.power_levels = np.array([0.1,0.2,0.3,0.4,0.5])
        self.action_space = spaces.Discrete(len(self.power_levels)) # choose power index
        self.observation_space = spaces.Box(
            low = 0.0 , high= 1.0 , shape = (2,) , dtype= np.float32
        )
        self.noise = 0.1
        self.lamda_pen = 0.2
        self.seed = seed
        self.np_random , _ = gym.utils.seeding.np_random(seed)


    def reset(self , seed = None , options = None):
        if seed is not None:
            self.np_random , _ = gym.utils.seeding.np_random(seed)

        self.channel_gain = self.np_random.uniform(0.1 , 1.0)
        self.interference = self.np_random.uniform(0.1 , 1.0)

        state = np.array([self.channel_gain , self.interference] , dtype=np.float32)

        return state , {}


    def step(self, action):
        power = self.power_levels[action]

        sinr = (power * self.channel_gain) / (self.interference + self.noise)
        reward = np.log2(1 + sinr) - self.lamda_pen * power

        self.channel_gain = self.np_random.uniform(0.1, 1.0)
        self.interference = self.np_random.uniform(0.1, 1.0)

        next_state = np.array(
            [self.channel_gain, self.interference], dtype=np.float32
        )

        terminated = False
        truncated = False

        return next_state, reward, terminated, truncated, {}

