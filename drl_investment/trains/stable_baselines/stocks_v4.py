import os
import logging
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from drl_investment.data.tdx import unpack_data
from drl_investment.envs.stocks_v4 import StocksEnvV4
# import drl_investment.envs.stocks_v4.StocksEnvV4
# from gym.envs.registration import register as gym_register
# from gymnasium.envs.registration import register as gymnasium_register


# gymnasium_register(
#     id='drl_investment/StocksEnv-v2',
#     entry_point="drl_investment.envs.stocks_v2:StocksEnvV2",
# )

LOG = logging.getLogger(__name__)

# Parallel environments
data_path = os.path.join(os.path.dirname(__file__), '../..', 'tests/assets/sh000001.day')
df = unpack_data(data_path)['2006-01-01':]
env_config = {
    'data': df
}

# vec_env = make_vec_env("drl_investment/StocksEnv-v1", env_kwargs=env_config, n_envs=4)
env = StocksEnvV4(config=env_config)

check_env(env)

tensorboard_log = os.path.expanduser('~/sb3_results/ppo_stocks_v4')


model = PPO('MlpPolicy', env, verbose=2, tensorboard_log=tensorboard_log)
print(model.policy)
TIMESTEPS = 1500000
begin_time = datetime.datetime.now()
model.learn(total_timesteps=TIMESTEPS)
model.save(os.path.join(model.logger.dir, f'model'))
print(f'The begin loop cost {datetime.datetime.now() - begin_time}s')

for i in range(1, 1000):
    begin_time = datetime.datetime.now()
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(os.path.join(model.logger.dir, f'model_{i*TIMESTEPS}'))
    print(f'The {i} loop ({i*TIMESTEPS}-{(i+1)*TIMESTEPS-1} steps) cost {datetime.datetime.now() - begin_time}s')
