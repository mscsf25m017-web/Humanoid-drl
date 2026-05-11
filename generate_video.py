import os
import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
import time

# Ensure the video folder exists
video_folder = "./humanoid_videos"
os.makedirs(video_folder, exist_ok=True)

# Create the environment for Humanoid-v5
env = gym.make("Humanoid-v5", render_mode="rgb_array")

# Load the trained PPO model
model = PPO.load("humanoid_ppo_model-2.zip")

# Initialize frames list for video recording
frames = []

# Initialize rewards list
rewards = []

# Gymnasium reset returns (obs, info)
obs, info = env.reset()

# Inspect the structure of the observation to understand how to extract the agent's position
print("Observation structure:", obs)

# Run for 10,000 timesteps (or more for a longer video)
for _ in range(10000):  # Running for 10,000 timesteps for a longer video
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # **Reward for speed** (encourage faster movement)
    velocity_reward = np.linalg.norm(obs[0])  # Assuming obs[0] is the position of the agent
    reward += velocity_reward  # Add velocity reward to the original reward

    # **Balance Reward** (encourage balance)
    # Assuming that obs[1] contains the vertical position (y-axis). Adjust this if needed after inspecting `obs`
    balance_reward = -np.abs(obs[1])  # Penalize the agent for falling (if vertical position is obs[1])
    reward += balance_reward  # Add balance reward to the total reward

    rewards.append(reward)  # Store reward

    done = terminated or truncated
    if done:  # If episode ends
        obs, info = env.reset()  # Reset environment for next episode

    # Render the frame and store it
    frame = env.render()
    frames.append(frame)

# Close the environment after simulation
env.close()

# Save the video with a different name
imageio.mimsave("ppo_humanoid_speed_video.mp4", frames, fps=30)

print("Video saved: ppo_humanoid_speed_video.mp4")