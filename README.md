# Humanoid PPO Model

This project trains a **Humanoid agent** using **Proximal Policy Optimization (PPO)** in the **Humanoid-v5** environment provided by `gymnasium`. The agent learns to walk, balance, and move efficiently using reinforcement learning in a simulated physics environment powered by **MuJoCo**.

---

## Project Overview

- **Objective:** Train a humanoid agent to walk using reinforcement learning.
- **Algorithm:** Proximal Policy Optimization (PPO) from `stable-baselines3`.
- **Environment:** Humanoid-v5 (`gymnasium`) with physics simulation via **MuJoCo**.
- **Video Recording:** Capture agent performance using `imageio`.

---

## Installation

Make sure you have **Python 3.14** or higher installed.

1. Install Python libraries:

```bash
pip install gymnasium stable-baselines3 mujoco pyvirtualdisplay imageio[ffmpeg]
sudo apt-get install xvfb ffmpeg xorg-dev libsdl2-dev libosmesa6-dev libglew-dev mesa-utils
1. Training the Agent
Initialize the Humanoid-v5 environment with render_mode='rgb_array'.
Create a PPO agent (MlpPolicy) and train it:State → Action → Reward → Next State → Policy Update → Repeat
Train for 1,000,000 timesteps.
Save the trained model:model.save("humanoid_ppo_model")
Loading the Trained Modelfrom stable_baselines3 import PPO
model = PPO.load("humanoid_ppo_model")
Generating Video in VSCode
Reset the environment.
Run the trained agent, collecting frames:
Workflow / Environment
Virtual Display: Handles rendering in headless systems.
Humanoid Environment: Provides state vectors, action space, and reward function.
PPO Agent: Neural network maps states → actions and updates policy based on rewards.
Training Loop: Observes state → chooses action → receives reward → updates policy.
Video Generation: Captures frames and saves a video to visualize agent behavior.Features
Trains a humanoid agent to walk using PPO reinforcement learning.
Works on VSCode, headless servers, or local machines.
Generates video output of the trained agent.
Saves trained models for reuse without retraining.
