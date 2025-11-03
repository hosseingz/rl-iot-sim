from stable_baselines3.common.env_checker import check_env
from env import SmartClimateControlEnv



env = SmartClimateControlEnv()
env.reset()

check_env(env)