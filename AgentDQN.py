from singleLinEnv_DQN import DQN_Env
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import os

# --- Reproducibility ---
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

# --- Parameters ---
episodes = 500
max_step = 20  # Increased slightly to give the agent more time to learn
target_update_freq = 10
gamma = 0.95
learning_rate = 0.001 # Reduced from 0.1 to 0.001 for stability
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 10000

class DQNagent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.train_freq = 4 
        self.step_count = 0
        
        self.policy_net = self.build_model()
        self.target_net = self.build_model()
        self.update_target_network()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_dim, activation="linear")
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss="mse")
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
        # Only train inside the store logic to respect train_freq
        if len(self.memory) >= self.batch_size and self.step_count % self.train_freq == 0:
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

# --- Training Loop ---
env = DQN_Env()
agent = DQNagent(env.observation_space.shape[0], env.action_space.n, lr=learning_rate)
rewards_history = []
moving_avg_history = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_step):
        action = agent.action_selection(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # This handles the train_freq internally now
        agent.store(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        if done: break

    agent.decay_epsilon()
    if episode % target_update_freq == 0:
        agent.update_target_network()

    rewards_history.append(total_reward)
    
    # Calculate 10-episode moving average
    if len(rewards_history) >= 10:
        avg = np.mean(rewards_history[-10:])
        moving_avg_history.append(avg)

    if episode % 50 == 0:
        print(f"Episode {episode:3d} | Reward: {total_reward:.2f} | Avg (last 10): {np.mean(rewards_history[-10:]):.2f} | Eps: {agent.epsilon:.3f}")


plt.plot(rewards_history, alpha=0.3, label="Raw Reward")
plt.plot(range(9, episodes), moving_avg_history, label="10-Ep Moving Avg", color='red')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

save_dir = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\DQN"
os.makedirs(save_dir, exist_ok=True)

plot_path = os.path.join(save_dir, "DQN_model_500_episodes.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved at: {plot_path}")
