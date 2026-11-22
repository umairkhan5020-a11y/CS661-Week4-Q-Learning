
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import csv

# Experiment settings
EPISODES = 10000
EVALUATION_EPISODES = 1000
SEEDS = [10, 20, 30, 40, 50]

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

def train_agent(seed, use_decay):
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True
    )

    rng = np.random.default_rng(seed)
    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    epsilon = 1.0 if use_decay else 0.1
    rewards = []

    for episode in range(EPISODES):
        state, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            target = reward
            if not terminated:
                target += DISCOUNT_FACTOR * np.max(
                    q_table[next_state]
                )

            q_table[state, action] += LEARNING_RATE * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if use_decay:
            epsilon = max(0.01, epsilon * 0.9995)

    env.close()
    return q_table, np.array(rewards)


def evaluate_agent(q_table):
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True
    )

    successes = 0

    for episode in range(EVALUATION_EPISODES):
        state, _ = env.reset(seed=50000 + episode)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(
                action
            )
            successes += reward

    env.close()
    return successes / EVALUATION_EPISODES * 100


results = []
learning_curves = {}

for model_name, use_decay in [
    ("Baseline", False),
    ("Improved", True)
]:
    model_curves = []

    for seed in SEEDS:
        q_table, rewards = train_agent(seed, use_decay)
        evaluation_rate = evaluate_agent(q_table)
        training_rate = np.mean(rewards) * 100

        results.append([
            model_name,
            seed,
            round(training_rate, 2),
            round(evaluation_rate, 2)
        ])

        model_curves.append(rewards)

        print(
            f"{model_name}, seed {seed}: "
            f"Training {training_rate:.2f}%, "
            f"Evaluation {evaluation_rate:.2f}%"
        )

    learning_curves[model_name] = np.mean(
        model_curves, axis=0
    )

# Save numerical results
with open("controlled_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Model",
        "Seed",
        "Training Success (%)",
        "Evaluation Success (%)"
    ])
    writer.writerows(results)

# Plot smoothed learning curves
plt.figure(figsize=(10, 5))

window = 200

for model_name, rewards in learning_curves.items():
    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        np.arange(window, EPISODES + 1),
        moving_average,
        label=model_name
    )

plt.xlabel("Training Episode")
plt.ylabel("Mean Reward (200-Episode Moving Average)")
plt.title("FrozenLake: Controlled Q-learning Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("controlled_comparison.png", dpi=300)
plt.close()

# Print summary
print("\nCONTROLLED EXPERIMENT SUMMARY")

for model_name in ["Baseline", "Improved"]:
    rates = [
        row[3] for row in results
        if row[0] == model_name
    ]

    print(
        f"{model_name}: "
        f"Mean evaluation success = {np.mean(rates):.2f}%, "
        f"SD = {np.std(rates, ddof=1):.2f}"
    )

print("\nResults saved to controlled_results.csv")
print("Graph saved to controlled_comparison.png")
