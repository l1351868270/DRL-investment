from gymnasium.envs.registration import register as gymnasium_register


gymnasium_register(
    id="drl_investment/TDXRaw-v0",
    entry_point="drl_investment.envs.tdx_raw:TDXRawEnv",
)

gymnasium_register(
    id='drl_investment/StocksEnv-v1',
    entry_point="drl_investment.envs.stocks_v1:StocksEnvV1",
)

gymnasium_register(
    id='drl_investment/StocksEnv-v2',
    entry_point="drl_investment.envs.stocks_v2:StocksEnvV2",
)
