from gym.envs.registration import register

register(
    id="GraphEnv/GraphEnv-v0",
    entry_point="GraphEnv.env:GraphEnv",
)