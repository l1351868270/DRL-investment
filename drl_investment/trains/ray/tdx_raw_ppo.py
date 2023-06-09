'''
refer to: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
'''
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPOConfig

from drl_investment.envs.tdx_raw import TDXRawEnv
from drl_investment.data.tdx import unpack_data

df = unpack_data(r'E:\code\github\l1351868270\DRL-investment\drl_investment\tests\assets\sh000001.day')['2006-01-01':]
env_config = {
    'data': df.to_numpy(),
    'columns': df.columns.to_list(),
}

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .framework(framework="torch")
    .environment(TDXRawEnv, env_config=env_config)
    .training(model={"fcnet_hiddens": [512, 512, 256, 256, 128, 128, 64, 64, 32, 32]},)
    .build()
)

policy = algo.get_policy()
model = policy.model

print("Using exploration strategy:", policy.exploration)
print("Using model:", model)

for i in range(100000):
    result = algo.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")