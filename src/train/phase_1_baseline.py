"""
Phase 1 - Baseline
- Fixed target (explicit values)
- rate_scale_range = (1, 1) # 0% randomness
- No noise enabled
- Train from scratch
"""


from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
import torch as th
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env import SmartClimateControlEnv



# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(ROOT, 'logs', 'SAC_phase1')
MODEL_PATH = os.path.join(ROOT, 'models', 'SAC_phase1')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Environment
env = SmartClimateControlEnv(
    time_limit=300,
    options={
        "noise_config": None,
        "target_temp": 22.0,
        "target_hum": 50.0
    },
    rate_scale_range=(1, 1)
)


# Wrap with Monitor for logging
info_keywords=(
    'avg_error', 'temp_error',
    'hum_error', 'energy_weight',
    'energy_norm', 'error_norm'
)

env = Monitor(env, LOG_DIR, info_keywords=info_keywords)


# Model
policy_kwargs = dict(
    activation_fn=th.nn.SiLU,
    net_arch=dict(
        pi=[256, 128, 128],
        qf=[256, 128, 128]
    )
)

model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)


# Save periodic checkpoints (optional)
callback = CheckpointCallback(
    save_freq=10_000, save_path=os.path.dirname(MODEL_PATH),
    name_prefix='sac_phase1'
)

# Learn
TIMESTEPS = 300_000
model.learn(total_timesteps=TIMESTEPS, callback=callback)


# Final save
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")