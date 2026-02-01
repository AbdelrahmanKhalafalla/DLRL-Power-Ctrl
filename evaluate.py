import numpy as np

def evaluate_policy(env, policy_fn, episodes=100, max_steps=20):
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)
