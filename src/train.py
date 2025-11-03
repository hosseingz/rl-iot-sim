from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from env import SmartClimateControlEnv
import numpy as np
import os



mean_reward_steps = 9999
TIMESTEPS = mean_reward_steps + 1


# models_dir = f'models/DQN'
models_dir = f'models/SAC'
logdir = 'logs'

for path in [models_dir, logdir]:
    os.makedirs(path, exist_ok=True)



env = SmartClimateControlEnv()
env.reset()

# model = DQN('MlpPolicy', env, verbose=2, tensorboard_log=logdir, device='cuda')
model = SAC('MlpPolicy', env, verbose=2, tensorboard_log=logdir, device='cuda')



class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []
        

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'])
        
        if len(self.rewards) % mean_reward_steps == 0: 
            mean_reward = np.mean(self.rewards)
            total_reward = np.sum(self.rewards)
                
            with open('Reward_Log.txt', 'a') as f:
                f.write(f'total reward: {total_reward} - mean reward: {mean_reward}\n')
            
            self.rewards.clear()
            
        return True



for i in range(1, 101):
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='DQN', callback=RewardLoggerCallback())
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='SAC', callback=RewardLoggerCallback())
    model.save(f"{models_dir}/{TIMESTEPS * i}")
