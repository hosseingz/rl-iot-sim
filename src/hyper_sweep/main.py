from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC
import torch as th
import itertools
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env import SmartClimateControlEnv


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SWEEP_DIR = os.path.join(ROOT, 'results', 'hyper_sweep')

LOG_DIR = os.path.join(SWEEP_DIR, 'logs')

for path in [SWEEP_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)
    
    
# Number of parallel envs
NUM_ENVS = 15

# Number of timesteps per run
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 20_000



# Sweep hyperparameters
alpha_list = [8.0, 12.0, 16]
beta_list  = [0.5, 1.0, 2]
gamma_list = [0.02, 0.05]

hyper_combinations = list(itertools.product(alpha_list, beta_list, gamma_list))


info_keywords=(
    'avg_error', 'temp_error',
    'hum_error', 'energy_weight',
    'energy_norm', 'error_norm'
)


def env_wrapper(alpha, beta, gamma):
    def make_env():
        env = SmartClimateControlEnv(
            time_limit=300,
            options={
                "noise_config": None,
                "target_temp": 22.0,
                "target_hum": 50.0,
                
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma   
            },
            rate_scale_range=(1, 1)
        )

        # return Monitor(env, LOG_DIR, info_keywords=info_keywords)
        return env

    return make_env

    
def make_model(phase, env):

    if phase == 1:
        policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=dict(
            pi=[256, 128, 128],
            qf=[256, 128, 128]
        )
        )

        model = SAC(
            'MlpPolicy', env,
            verbose=2, policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR, device='cuda'
        )
        return model
    else:
        print('Invalid phase')


class LoggerCallback(BaseCallback):
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "avg_error" in info:
                self.logger.record("metrics/avg_error", info["avg_error"])
        
            if "total_power" in info:
                self.logger.record("metrics/total_power", info["total_power"])
                
            if "energy_norm" in info:
                self.logger.record("metrics/energy_norm", info["energy_norm"])
                
            if "error_norm" in info:
                self.logger.record("metrics/error_norm", info["error_norm"])
        
        return True
                
                

# Run sweep
for idx, (alpha, beta, gamma) in enumerate(hyper_combinations, start=1):
    run_name = f"run_a{alpha}_b{beta}_g{gamma}"
    
    run_dir = os.path.join(SWEEP_DIR, 'runs', run_name)
    model_dir = os.path.join(run_dir, 'models')
    
    for path in [run_dir, model_dir]:
        os.makedirs(path, exist_ok=True)

    print(f"=== Running sweep {idx}/{len(hyper_combinations)}: {run_name} ===")
    print(f"Hyperparameters: alpha={alpha}, beta={beta}, gamma={gamma}")


    # Create parallel envs
    make_env = env_wrapper(alpha, beta, gamma)
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    try:
        model = make_model(phase=1, env=env,)

        # callback
        checkpoint_callback = CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=model_dir,
            verbose=2,
            name_prefix=f'hps-{run_name}'
        )

        params_logger_callback = LoggerCallback(
            save_dir=run_dir,
            params={'alpha': alpha, 'beta': beta, 'gamma': gamma},
            verbose=1
        )

        # Training
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, params_logger_callback],
            tb_log_name=f"a{alpha}_b{beta}_g{gamma}"
        )

        # Save final model
        model_file = os.path.join(model_dir, 'model.zip')
        model.save(model_file)
        print(f"[Sweep] Saved final model at {model_file}\n")
    
    finally:  
        env.close()