
import gymnasium as gym
import numpy as np

# Evaluation settings
evaluation_episodes = 1000
max_steps = 100
evaluation_seed = 2026

def evaluate_model(filename, model_name):
    q_table = np.load(filename)

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
        max_episode_steps=max_steps
    )

    successes = 0
    total_rewards = []

    for episode in range(evaluation_episodes):
        state, info = env.reset(
            seed=evaluation_seed + episode
        )

        episode_reward = 0

        for step in range(max_steps):
            # Use the learned policy without random exploration
            action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        if episode_reward > 0:
            successes += 1

    env.close()

    success_rate = successes / evaluation_episodes * 100

    print(f"\n{model_name}")
    print(f"Evaluation episodes: {evaluation_episodes}")
    print(f"Successful episodes: {successes}")
    print(f"Evaluation success rate: {success_rate:.2f}%")
    print(f"Average reward: {np.mean(total_rewards):.4f}")

    return success_rate


# Evaluate both saved models
baseline_rate = evaluate_model(
    "baseline_q_table.npy",
    "Baseline Q-learning"
)

improved_rate = evaluate_model(
    "improved_q_table.npy",
    "Improved Q-learning"
)

print("\nEvaluation comparison")
print(f"Baseline: {baseline_rate:.2f}%")
print(f"Improved: {improved_rate:.2f}%")
print(f"Difference: {improved_rate - baseline_rate:+.2f} percentage points")
