from singleLinkEnv import wirelessPowerEnv
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# =========================
# 1. Reproducibility Settings
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

env = wirelessPowerEnv(seed=RANDOM_SEED)
env.reset(seed=RANDOM_SEED)

# =========================
# 2. Discretization
# =========================
NUM_BINS = 10
bins = np.linspace(0, 1, NUM_BINS)

def discretize(value):
    idx = np.digitize(value, bins) - 1
    return np.clip(idx, 0, NUM_BINS - 1)

# =========================
# 3. Q-table
# =========================
num_actions = env.action_space.n
Q = np.zeros((NUM_BINS, NUM_BINS, num_actions))

# =========================
# 4. Hyperparameters
# =========================
alpha = 0.1           # learning rate
gamma = 0.95          # discount factor
epsilon = 1.0         # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

EPISODES = 500
MAX_STEPS = 50

rewards_per_episode = []

# =========================
# 5. Training Loop
# =========================
for episode in range(EPISODES):
    state, _ = env.reset()
    s0 = discretize(state[0])
    s1 = discretize(state[1])

    total_reward = 0

    for step in range(MAX_STEPS):
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[s0, s1])

        next_state, reward, _, _, _ = env.step(action)

        ns0 = discretize(next_state[0])
        ns1 = discretize(next_state[1])

        # Q-learning update
        best_next_q = np.max(Q[ns0, ns1])
        Q[s0, s1, action] += alpha * (
            reward + gamma * best_next_q - Q[s0, s1, action]
        )

        s0, s1 = ns0, ns1
        total_reward += reward

    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode:3d} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

window = 20
moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10,6))
plt.plot(moving_avg, color='blue', label='Moving Avg Reward')
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Performance")
plt.grid(True)
plt.legend()

save_dir = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\Q-Learning"
os.makedirs(save_dir, exist_ok=True)

plot_path = os.path.join(save_dir, "Moving_Average_Reward_500_episodes.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved at: {plot_path}")
