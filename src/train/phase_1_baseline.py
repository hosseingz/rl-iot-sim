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
from callbacks import PerformanceThresholdCallback, LoggerCallback
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
        "target_hum": 50.0,
        
        # Optimized hyperparameters obtained through hyperparameter sweep.
        # Further testing may yield improved values for alpha and beta.
        "alpha": 6,
        "beta": 3,
    },
    rate_scale_range=(1, 1)
)


# Wrap with Monitor for logging
info_keywords=(
    'temp_error','hum_error',
    'error','energy_norm'
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


# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=os.path.dirname(MODEL_PATH),
    name_prefix='sac_phase1'
)

params_logger_callback = LoggerCallback()

performanceTh_callback = PerformanceThresholdCallback(
    metric_weights={"error": 0.8, "energy_norm": 0.2},
    thresholds={"error": 0.5, "energy_norm": 0.1},
    patience=30_000,
    fail_patience=60_000,
    warmup_steps=1500,
    success_streak=1000,
    adaptive=True,
    verbose=1
)


# Learn
TIMESTEPS = 3_000_000
model.learn(
    total_timesteps=TIMESTEPS,
    callback=[
        checkpoint_callback,
        params_logger_callback,
        performanceTh_callback
    ]
)


# Final save
final_save_path = os.path.join(MODEL_PATH, 'final_save.zip')
model.save(final_save_path)
print(f"Saved model to {final_save_path}")