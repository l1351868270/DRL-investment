from gym.envs.registration import register

# raise Exception("hello")
register(
    id="drl_investment/TDXRaw-v0",
    entry_point="drl_investment.envs.tdx_raw:TDXRawEnv",
)