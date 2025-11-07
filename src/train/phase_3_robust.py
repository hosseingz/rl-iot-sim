"""
Phase 3 - Robustness (Noise)
- Random targets
- rate_scale_range = (0.7, 1.3) # 30% randomness
- Noise enabled (sensor, env disturbance, power fluctuation, adaptive noise)
- Continue training from phase2 model
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
LOG_DIR = os.path.join(ROOT, 'logs', 'SAC_phase3')
MODEL_PATH = os.path.join(ROOT, 'models', 'SAC_phase3')
PREV_MODEL = os.path.join(ROOT, 'models', 'SAC_phase2')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Noise configuration
noise_cfg = {
    "random_target": {'enabled': True},
    "sensor_noise": {"enabled": True},
    "env_disturbance": {"enabled": True},
    "power_fluctuation": {"enabled": True},
    "adaptive_noise": {"enabled": True}
}

# Environment
env = SmartClimateControlEnv(
    time_limit=300,
    options={
        "noise_config": noise_cfg
    },
    rate_scale_range=(0.7, 1.3)
)

# Wrap with Monitor for logging
info_keywords=(
    'avg_error', 'temp_error',
    'hum_error', 'energy_weight',
    'energy_norm', 'error_norm'
)
env = Monitor(env, LOG_DIR, info_keywords=info_keywords)


# Try loading previous phase2 model


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