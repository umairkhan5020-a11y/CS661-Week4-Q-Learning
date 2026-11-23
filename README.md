# CS 661 Week 4: Q-Learning in FrozenLake

**Student:** Umair Moin  
**Course:** Artificial Intelligence and Machine Learning (CS 661 45)  
**Professor:** Jaime Fuentes  
**Assignment:** Week 4, Reinforcement Learning

## Project Overview

This project demonstrates reinforcement learning using the Q-learning algorithm in the Gymnasium FrozenLake environment. The objective is to train an agent to navigate a slippery 4 × 4 grid and reach the goal while avoiding holes. The experiment compares a baseline model using a fixed exploration rate with an improved model that gradually reduces exploration through epsilon decay.

## Implementation

Both models were trained for 10,000 episodes using a learning rate of 0.1 and a discount factor of 0.99. The baseline model used a fixed epsilon of 0.1, while the improved model started with epsilon 1.0 and reduced it gradually to a minimum of 0.01.

To evaluate the performance more reliably, a controlled experiment was conducted using five random seeds. Each trained model was evaluated over 1,000 episodes without random exploration.

## Experimental Results

| Model | Mean Evaluation Success Rate |
|---|---|
| Baseline Q-learning | 14.96% |
| Improved Q-learning | 75.94% |

The improved model achieved a 60.98 percentage point increase in mean evaluation success under the tested configurations. However, the experiment does not isolate the effect of epsilon decay because the two models also used different initial exploration rates.

## Python Requirements

Install the required libraries:

```bash
pip install gymnasium numpy matplotlib
```

## Running the Project

Run the following scripts from the repository directory:

```bash
python baseline_q_learning.py
python improved_q_learning.py
python evaluate_models.py
python controlled_experiment.py
```

The scripts generate Q-tables, training graphs and evaluation results. The controlled experiment saves its numerical results to `controlled_results.csv` and its comparison graph to `controlled_comparison.png`.

## Project Files

The repository contains Python scripts for training and evaluating the agents, saved Q-tables, training graphs, controlled experiment results and an AI assistance log.

## AI Assistance Disclosure

ChatGPT was used to support explanations of Q-learning, Python implementation, debugging, epsilon decay, performance evaluation and critical interpretation. The generated suggestions were reviewed against the experiment results, and the limitations of the comparison are discussed in the assignment report.
