import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Parallel environments
vec_env = gym.make("CartPole-v1")
env = NormalizeObservation(vec_env)
check_env(env)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")
check_env(env)
# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")