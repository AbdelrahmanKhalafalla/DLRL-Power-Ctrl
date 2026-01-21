import gymnasium
from gymnasium import spaces 
from singleLinkEnv import wirelessPowerEnv
import numpy as np
import matplotlib.pyplot as plt
import os

env = wirelessPowerEnv()

# ------- Q-learning parameters-----#####
episodes = 500
steps_per_episode= 50
Alpha = 0.1 # learning rate
Gamma = 0.9 # discount factor
epsilon = 1.0 # initial
epsilon_min = 0.05
epsilon_decay = 0.995
n_states = 6
n_action = 3


def state_to_index(state):
    channel, interference = state
    return channel * 2 + interference

Q= np.zeros((n_states , n_action))
episode_rewards = []

for episode in range(episodes):
    obs , _ = env.reset()
    state = state_to_index(obs)
    total_reward = 0

    for step in range(steps_per_episode):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_obs , reward , terminated , truncated , _ = env.step(action)
        next_state = state_to_index(next_obs)

        ############ ------- Q- Learning update ---------###########
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + Gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += Alpha * td_error

        state = next_state
        total_reward += reward

    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# -------- Results --------
print("\nLearned Q-table:")
rewards = episode_rewards

# Optional: smooth with a moving average
window = 10
smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

plt.plot(rewards, label="Episode Reward", alpha=0.4)
plt.plot(range(window - 1, len(smoothed) + window - 1), smoothed, label="Smoothed", color="red")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Convergence Over Training")
plt.legend()
plt.grid(True)

# Create simulationimages folder if it doesn't exist
os.makedirs("simulationimages", exist_ok=True)

# Save the plot
plt.savefig("simulationimages/reward_convergence.png", dpi=300, bbox_inches="tight")
print(f"\nPlot saved to simulationimages/reward_convergence.png")
print(Q)
plt.show()


