# ===============================
# Imports
# ===============================
from singleLinEnv_DQN import DQN_Env
import gymnasium as gym
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle

from baseline import fixed_power_policy, random_power_policy, rule_based_policy
from evaluate import evaluate_policy


# ===============================
# Reproducibility
# ===============================
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)


# ===============================
# Hyperparameters (IMPROVED)
# ===============================
episodes = 3000
max_step = 30

gamma = 0.95
learning_rate = 3e-4

epsilon_start = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05

batch_size = 64
memory_size = 10000
train_freq = 1
target_update_freq = 100


# ===============================
# Save paths
# ===============================
save_dir = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\DQN"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "dqn_model.keras")
epsilon_path = os.path.join(save_dir, "epsilon.pkl")
rewards_path = os.path.join(save_dir, "rewards_history.npy")
csv_log_path = os.path.join(save_dir, "Training_Log._3000_episodes.csv")


# ===============================
# DQN Agent
# ===============================
class DQNagent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.train_freq = train_freq
        self.step_count = 0

        self.policy_net = self.build_model()
        self.target_net = self.build_model()
        self.update_target_network()

        if os.path.exists(model_path):
            print("Loading saved model...")
            self.policy_net = models.load_model(model_path, compile=False)
            self.target_net = models.load_model(model_path, compile=False)

            self.policy_net.compile(
                optimizer=optimizers.Adam(learning_rate, clipnorm=1.0),
                loss="mse"
            )
            self.target_net.compile(
                optimizer=optimizers.Adam(learning_rate, clipnorm=1.0),
                loss="mse"
            )

            if os.path.exists(epsilon_path):
                with open(epsilon_path, "rb") as f:
                    self.epsilon = pickle.load(f)

            print(f"Resuming with epsilon = {self.epsilon:.3f}")

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_dim, activation="linear")
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate, clipnorm=1.0),
            loss="mse"
        )
        return model

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def action_selection(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        q_values = self.policy_net.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.step_count += 1

        # Warm-up
        if len(self.memory) < 1000:
            return

        if self.step_count % self.train_freq == 0:
            self.train()

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        q_current = self.policy_net.predict(states, verbose=0)
        q_next = self.target_net.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_current[i, actions[i]] = target

        self.policy_net.fit(states, q_current, epochs=1, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ===============================
# Training
# ===============================
env = DQN_Env()
agent = DQNagent(env.observation_space.shape[0], env.action_space.n)

rewards_history = []
log_data = []
moving_avg_history = []

print("Starting Training...")

for episode in range(episodes):
    state, _ = env.reset(seed=random_seed + episode)
    total_reward = 0

    for step in range(max_step):
        action = agent.action_selection(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if episode % target_update_freq == 0:
        agent.update_target_network()

    rewards_history.append(total_reward)

    avg_reward = np.mean(rewards_history[-10:])
    moving_avg_history.append(avg_reward)

    log_data.append({
        "Episode": episode,
        "Reward": total_reward,
        "Epsilon": agent.epsilon
    })

    if episode % 100 == 0:
        print(f"Ep {episode:4d} | Reward {total_reward:.2f} | Avg10 {avg_reward:.2f} | Eps {agent.epsilon:.3f}")


# ===============================
# Evaluation
# ===============================
def dqn_policy(agent):
    def policy(state):
        return agent.action_selection(state)
    return policy

agent.epsilon = 0.0

mean_dqn, std_dqn = evaluate_policy(env, dqn_policy(agent), episodes=200, max_steps=max_step)
mean_fixed, std_fixed = evaluate_policy(env, lambda s: fixed_power_policy(s, env.action_space.n))
mean_random, std_random = evaluate_policy(env, lambda s: random_power_policy(s, env.action_space.n))
mean_rule, std_rule = evaluate_policy(env, lambda s: rule_based_policy(s, env.action_space.n))

print("\n===== BASELINE COMPARISON =====")
print(f"DQN     → {mean_dqn:.2f} ± {std_dqn:.2f}")
print(f"Random  → {mean_random:.2f} ± {std_random:.2f}")
print(f"Rule    → {mean_rule:.2f} ± {std_rule:.2f}")
print(f"Fixed   → {mean_fixed:.2f} ± {std_fixed:.2f}")


# ===============================
# Plot Baselines
# ===============================
plt.figure(figsize=(10,6))
labels = ["DQN", "Random", "Rule", "Fixed"]
means = [mean_dqn, mean_random, mean_rule, mean_fixed]

plt.bar(labels, means)
plt.ylabel("Average Reward")
plt.title("Baseline Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()


# ===============================
# Save Everything
# ===============================
agent.policy_net.save(model_path)
with open(epsilon_path, "wb") as f:
    pickle.dump(agent.epsilon, f)

np.save(rewards_path, np.array(rewards_history))
pd.DataFrame(log_data).to_csv(csv_log_path, index=False)

plt.figure(figsize=(10,6))
plt.plot(rewards_history, alpha=0.3, label="Reward")
plt.plot(moving_avg_history, label="10-Ep Avg", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=300)
plt.show()

print("✅ TRAINING COMPLETE")
