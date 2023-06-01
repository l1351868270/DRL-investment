import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from drl_investment.data.tdx import unpack_data
from drl_investment.envs.stocks_v3 import StocksEnvV3
# from gym.envs.registration import register as gym_register
# from gymnasium.envs.registration import register as gymnasium_register


# gymnasium_register(
#     id='drl_investment/StocksEnv-v2',
#     entry_point="drl_investment.envs.stocks_v2:StocksEnvV2",
# )

# Parallel environments
df = unpack_data(r'E:\code\github\l1351868270\DRL-investment\drl_investment\tests\assets\sh000001.day')['2006-01-01':]
env_config = {
    'data': df
}

# vec_env = make_vec_env("drl_investment/StocksEnv-v1", env_kwargs=env_config, n_envs=4)
env = StocksEnvV3(config=env_config)

check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_stocks_v2_tensorboard/")
print(model.policy)
model.learn(total_timesteps=250000)
model.save("ppo_stocks_v2")