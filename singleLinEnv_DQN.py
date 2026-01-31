import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DQN_Env(gym.Env):
    """
    DQN environment for wireless power control.
    State: [channel_gain, interference] (continuous)
    Action: discrete power levels
    Reward: log2(1 + SINR) - penalty * power
    """

    def __init__(self, seed=42):
        super().__init__()
        # Discrete power levels
        self.power_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.action_space = spaces.Discrete(len(self.power_levels))

        # Continuous state space: channel gain and interference
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Environment parameters
        self.noise = 0.1
        self.lambda_pen = 0.2
        self.seed = seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.state = None  # placeholder for current state

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Initialize channel gain and interference randomly
        self.channel_gain = self.np_random.uniform(0.1, 1.0)
        self.interference = self.np_random.uniform(0.1, 1.0)

        self.state = np.array([self.channel_gain, self.interference], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # Map action to power
        power = self.power_levels[action]

        # Compute SINR and reward
        sinr = (power * self.channel_gain) / (self.interference + self.noise)
        reward = np.log2(1 + sinr) - self.lambda_pen * power

        # Update channel and interference randomly for next step
        self.channel_gain = self.np_random.uniform(0.1, 1.0)
        self.interference = self.np_random.uniform(0.1, 1.0)
        self.state = np.array([self.channel_gain, self.interference], dtype=np.float32)

        terminated = False  # continuous environment
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info
