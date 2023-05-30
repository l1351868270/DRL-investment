'''
refer to: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
'''
import numpy as np
import matplotlib.pyplot as plt
from ray.rllib.policy.policy import Policy

from drl_investment.models.ray.fc_net import TorchFC
from drl_investment.envs.stocks_v1 import StocksEnvV1
from drl_investment.data.tdx import unpack_data

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


df = unpack_data(r'E:\code\github\l1351868270\DRL-investment\drl_investment\tests\assets\sh000001.day')['2006-01-01':]
env_config = {
    'data': df
}

policies = Policy.from_checkpoint(r'C:\Users\DELL\ray_results\PPO_StocksEnvV1_2023-05-17_18-12-16ci1dvfcb\checkpoint_002701')
policy = policies['default_policy']
print(f'policy: {policy}')
env = StocksEnvV1(config=env_config)
observation, _ = env.reset()
done = False
rewards = []
while not done:
    action = policy.compute_single_action(observation)
    observation, reward, done, _, info = env.step(action[0])
    rewards.append(reward)
    print(f'observation: {observation}, reward: {reward}, info: {info}')

    if env.render_mode == 'human':
        env.render()

rewards = np.cumsum(np.array(rewards))
fig, ax = plt.subplots()
ax.plot(rewards)
fig.show()


    


