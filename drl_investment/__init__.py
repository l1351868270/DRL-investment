from gym.envs.registration import register as gym_register
from gymnasium.envs.registration import register as gymnasium_register

# raise Exception("hello")
gymnasium_register(
    id="drl_investment/TDXRaw-v0",
    entry_point="drl_investment.envs.tdx_raw:TDXRawEnv",
)