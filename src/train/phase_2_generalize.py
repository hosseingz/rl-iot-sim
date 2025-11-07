"""
Phase 2 - Generalization
- Random targets each episode
- rate_scale_range = (0.9, 1.1) # 10% randomness
- No noise enabled
- Train starting from phase1 (incremental): load model from phase1 and continue training
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
LOG_DIR = os.path.join(ROOT, 'logs', 'SAC_phase2')
MODEL_PATH = os.path.join(ROOT, 'models', 'SAC_phase2')
PREV_MODEL = os.path.join(ROOT, 'models', 'SAC_phase1')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Environment
env = SmartClimateControlEnv(
    time_limit=300,
    options={
        "noise_config": {"random_target": {'enabled': True}}
    },
    rate_scale_range=(0.9, 1.1)
)

# Wrap with Monitor for logging
info_keywords=(
    'avg_error', 'temp_error',
    'hum_error', 'energy_weight',
    'energy_norm', 'error_norm'
)
env = Monitor(env, LOG_DIR, info_keywords=info_keywords)


# Try to load previous model; if not present, start from scratch

policy_kwargs = dict(
    activation_fn=th.nn.SiLU,
    net_arch=dict(
        pi=[256, 128, 128],
        qf=[256, 128, 128]
    )
)

if os.path.exists(PREV_MODEL + ".zip") or os.path.exists(PREV_MODEL):
    try:
        model = SAC.load(PREV_MODEL, env=env, verbose=1, policy_kwargs=policy_kwargs,)
        print(f"Loaded model from {PREV_MODEL}")
    except Exception as e:
        print(f"Failed to load previous model ({PREV_MODEL}): {e}\nTraining from scratch.")
        model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)
    else:
        model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)


# Save periodic checkpoints (optional)
callback = CheckpointCallback(
    save_freq=10_000, save_path=os.path.dirname(MODEL_PATH),
    name_prefix='sac_phase3'
)

# Learn
TIMESTEPS = 300_000
model.learn(total_timesteps=TIMESTEPS, callback=callback)

# Final save
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")