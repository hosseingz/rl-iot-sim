from stable_baselines3.common.callbacks import CheckpointCallback
from env import SmartClimateControlEnv
from stable_baselines3 import SAC
import torch as th
import os


# models_dir = f'models/DQN'
models_dir = f'models/SAC'
logdir = 'logs'

for path in [models_dir, logdir]:
    os.makedirs(path, exist_ok=True)



env = SmartClimateControlEnv()
env.reset()


policy_kwargs = dict(
    activation_fn=th.nn.SiLU,
    net_arch=dict(
        pi=[256, 128, 128],
        qf=[256, 128, 128]
    )
)

# model = DQN('MlpPolicy', env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=logdir, device='cuda')
model = SAC('MlpPolicy', env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=logdir, device='cuda')


callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    verbose=1
)

# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='DQN', callback=RewardLoggerCallback())
model.learn(
    total_timesteps=10000000,
    reset_num_timesteps=False, tb_log_name='SAC',
    callback=callback
)

