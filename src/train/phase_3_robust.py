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
from callbacks import PerformanceThresholdCallback, LoggerCallback
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
        "noise_config": noise_cfg,
        
        # Optimized hyperparameters obtained through hyperparameter sweep.
        # Further testing may yield improved values for alpha and beta.
        "alpha": 6,
        "beta": 3,
    },
    rate_scale_range=(0.7, 1.3)
)

# Wrap with Monitor for logging
info_keywords=(
    'temp_error','hum_error',
    'error','energy_norm'
)
env = Monitor(env, LOG_DIR, info_keywords=info_keywords)



policy_kwargs = dict(
    activation_fn=th.nn.SiLU,
    net_arch=dict(
        pi=[256, 128, 128],
        qf=[256, 128, 128]
    )
)

# Load pre-trained model for continuation; exit if not found
if os.path.exists(PREV_MODEL + ".zip") or os.path.exists(PREV_MODEL):
    try:
        model = SAC.load(PREV_MODEL, env=env, verbose=1, policy_kwargs=policy_kwargs)
        print(f"Loaded model from {PREV_MODEL}")
    except Exception as e:
        print(f"Failed to load previous model ({PREV_MODEL}): {e}")
        print("Exiting execution.")
        exit(1)
else:
    print(f"Previous model ({PREV_MODEL}) not found.")
    print("Exiting execution.")
    exit(1)


# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, save_path=os.path.dirname(MODEL_PATH),
    name_prefix='sac_phase3'
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