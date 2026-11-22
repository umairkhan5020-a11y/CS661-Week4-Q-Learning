
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Q-learning settings
episodes = 10000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Store the reward from each episode
rewards = []

# Train the agent
for episode in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:

        # Choose between exploration and exploitation
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        # Perform the selected action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the Q-table using the Q-learning equation
        best_next_action_value = np.max(q_table[next_state])

        q_table[state, action] += learning_rate * (
            reward
            + discount_factor * best_next_action_value * (not terminated)
            - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

env.close()

# Calculate results
success_rate = np.mean(rewards) * 100

print("Training completed.")
print("Total episodes:", episodes)
print(f"Training success rate: {success_rate:.2f}%")
print("Final Q-table:")
print(q_table)

# Calculate moving average
window = 100
moving_average = np.convolve(
    rewards, np.ones(window) / window, mode="valid"
)

# Plot training performance
plt.figure(figsize=(10, 5))
plt.plot(moving_average, label="Baseline Q-learning")
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 Episodes)")
plt.title("FrozenLake Baseline Q-learning Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_rewards.png", dpi=300)
plt.show()
