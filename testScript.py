import gymnasium
from gymnasium import spaces
from singleLinkEnv import wirelessPowerEnv
import numpy as np
import matplotlib.pyplot as plt
import os

env = wirelessPowerEnv()

# ------- Q-learning parameters-----#####
episodes = 500  # Increased for better convergence
steps_per_episode = 50
Alpha = 0.1  # learning rate
Alpha_min = 0.01  # minimum learning rate
Alpha_decay = 0.999  # learning rate decay
Gamma = 0.9  # discount factor
epsilon = 1.0  # initial
epsilon_min = 0.01  # Lower minimum for better exploitation
epsilon_decay = 0.995
n_states = 6
n_action = 3

# Evaluation parameters
eval_episodes = 20
eval_frequency = 50  # Evaluate every N episodes


def state_to_index(state):
    channel, interference = state
    return int(channel * 2 + interference)


Q = np.zeros((n_states, n_action))
episode_rewards = []
eval_rewards = []  # Evaluation rewards
episode_throughputs = []
episode_powers = []

# Track Q-value changes for convergence
q_value_history = []

for episode in range(episodes):
    obs, _ = env.reset()
    state = state_to_index(obs)
    total_reward = 0
    total_throughput = 0
    total_power = 0

    for step in range(steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = state_to_index(next_obs)

        # Calculate metrics (approximate from reward)
        power = env.power_levels[action]
        total_power += power
        # Throughput approximation: reward + lambda*power
        throughput = reward + env.lamda_power * power
        total_throughput += throughput

        ############ ------- Q- Learning update ---------###########
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + Gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += Alpha * td_error

        state = next_state
        total_reward += reward

    # Decay exploration and learning rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    Alpha = max(Alpha_min, Alpha * Alpha_decay)
    
    episode_rewards.append(total_reward)
    episode_throughputs.append(total_throughput / steps_per_episode)
    episode_powers.append(total_power / steps_per_episode)
    
    # Track Q-value magnitude for convergence
    if episode % 10 == 0:
        q_value_history.append(np.mean(np.abs(Q)))

    # Periodic evaluation (no exploration)
    if episode % eval_frequency == 0:
        eval_reward_sum = 0
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            state = state_to_index(obs)
            eval_reward = 0
            for _ in range(steps_per_episode):
                action = np.argmax(Q[state])  # Greedy policy
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = state_to_index(next_obs)
                eval_reward += reward
                state = next_state
            eval_reward_sum += eval_reward
        eval_rewards.append((episode, eval_reward_sum / eval_episodes))
        print(f"Episode {episode}, Train Reward: {total_reward:.2f}, "
              f"Eval Reward: {eval_reward_sum/eval_episodes:.2f}, "
              f"Epsilon: {epsilon:.3f}, Alpha: {Alpha:.3f}")

# -------- Enhanced Results Visualization --------
print("\n" + "="*50)
print("Training Statistics:")
print(f"Mean Reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}")
print(f"Std Reward (last 50 episodes): {np.std(episode_rewards[-50:]):.2f}")
print(f"Mean Throughput: {np.mean(episode_throughputs[-50:]):.2f}")
print(f"Mean Power: {np.mean(episode_powers[-50:]):.2f}")
print("\nLearned Q-table:")
print(Q)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Reward convergence
window = 20
smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
ax1.plot(episode_rewards, label="Episode Reward", alpha=0.3, color='blue')
ax1.plot(range(window - 1, len(smoothed) + window - 1), smoothed, 
         label=f"Smoothed (window={window})", color="red", linewidth=2)
if eval_rewards:
    eval_eps, eval_rews = zip(*eval_rewards)
    ax1.scatter(eval_eps, eval_rews, color='green', s=50, zorder=5, 
                label='Evaluation (greedy)', marker='x')
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Reward Convergence Over Training")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Q-value convergence
ax2.plot(q_value_history, color='purple')
ax2.set_xlabel("Episode (x10)")
ax2.set_ylabel("Mean |Q-value|")
ax2.set_title("Q-Value Convergence")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plot_path = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\Q-Learning\reward_convergence.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {plot_path}")
plt.show()

# Additional analysis: Throughput vs Power trade-off
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Throughput over episodes
window = 20
smoothed_throughput = np.convolve(episode_throughputs, np.ones(window) / window, mode="valid")
ax1.plot(episode_throughputs, alpha=0.3, color='blue', label='Raw')
ax1.plot(range(window - 1, len(smoothed_throughput) + window - 1), 
         smoothed_throughput, color='red', linewidth=2, label='Smoothed')
ax1.set_xlabel("Episode")
ax1.set_ylabel("Average Throughput")
ax1.set_title("Throughput Over Training")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power usage over episodes
smoothed_power = np.convolve(episode_powers, np.ones(window) / window, mode="valid")
ax2.plot(episode_powers, alpha=0.3, color='blue', label='Raw')
ax2.plot(range(window - 1, len(smoothed_power) + window - 1), 
         smoothed_power, color='red', linewidth=2, label='Smoothed')
ax2.set_xlabel("Episode")
ax2.set_ylabel("Average Power")
ax2.set_title("Power Usage Over Training")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path2 = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\Q-Learning\throughput_power.png"
os.makedirs(os.path.dirname(plot_path2), exist_ok=True)
plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
print(f"Additional plot saved to {plot_path2}")
plt.show()