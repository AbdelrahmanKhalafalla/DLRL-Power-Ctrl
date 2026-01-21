import math
import simpy 
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class wirelessPowerEnv(gym.Env):
    def __init__(self):
        super().__init__()
 # -------- Action space --------
# 0: low power, 1: medium power, 2: high power
 # ------------- observations-------------
 # channel params: 0: low, 1: medium, 2: high
 # interference params: 0: low, 1: high
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([3, 2])
        self.power_levels = np.array([0.1, 0.5, 1.0])
        self.noise = 0.1
        self.lamda_power = 0.1
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        channel = np.random.randint(0, 3)
        interference = np.random.randint(0,2)

        self.state = np.array([channel , interference ], dtype=np.int32)
        return self.state , {}
    def step(self , action):
        power = self.power_levels[action]

        #####--------- Channel fading -------------######
        h = np.random.exponential(scale=0.1)

        IP = 0.1 if self.state[1]== 0 else 0.5

        SINR = (power * h) / (self.noise + IP) 

        throughput = np.log2(1 + SINR)
         #### -------  Reward Function ------#####
        reward =  throughput - self.lamda_power * power

          ########## ---- next state --------#####
        next_channel = np.random.randint(0, 3)
        next_interference = np.random.randint(0, 2)
        self.state = np.array([next_channel, next_interference], dtype=np.int32)

        # (no terminal state)
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}
