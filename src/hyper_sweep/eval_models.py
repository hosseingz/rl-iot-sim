from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env import SmartClimateControlEnv



ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SWEEP_DIR = os.path.join(ROOT, 'results', 'hyper_sweep')

RUNS_DIR = os.path.join(SWEEP_DIR, 'runs')
TB_DIR = os.path.join(SWEEP_DIR, 'tb_eval')

MEAN_EPISODE_CSV = os.path.join(SWEEP_DIR, 'mean_episode.csv')
MEAN_SCALAR_CSV  = os.path.join(SWEEP_DIR, 'mean_scalar.csv')


EPISODES = 100
EP_LEN = 300


# skip if exists 
if os.path.exists(MEAN_EPISODE_CSV) and os.path.exists(MEAN_SCALAR_CSV):
    print("✅ CSV files already exist. Skip evaluation.")
    exit()



df_mean_episode = []
df_mean_scalar = []


def eval_model(model_path):
    model_name = model_path.split(os.sep)[-3]  # runs/<run_name>/models/model.zip  => model_name = run_name
    match = re.match(r"run_a([0-9.]+)_b([0-9.]+)_g([0-9.]+)", model_name)

    tb_writer = SummaryWriter(os.path.join(TB_DIR, model_name))

    if not match:
        print(f"'{model_name}' does not match the pattern. Skipping.")
        return
    
    try:
        env = SmartClimateControlEnv(
            time_limit=EP_LEN,
            options={
                "noise_config": None,
                "target_temp": 22.0,
                "target_hum": 50.0,
                
                "alpha": float(match.group(1)),
                "beta": float(match.group(2)),
                "gamma": float(match.group(3)),
        })

        model = SAC.load(model_path, env=env, verbos=0)

        avg_error_all = np.zeros((EPISODES, EP_LEN))
        energy_norm_all = np.zeros((EPISODES, EP_LEN))


        print(f'\nModel: {model_name}')
        for ep in tqdm(range(EPISODES)):
            obs,_ = env.reset()
            for t in range(EP_LEN):
                action,_ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)

                avg_error_all[ep, t] = info.get("avg_error", 0.0)
                energy_norm_all[ep, t] = info.get("energy_norm", 0.0)

                if term or trunc:
                    break

        mean_avg_error = avg_error_all.mean(axis=0)
        mean_energy_norm = energy_norm_all.mean(axis=0)
        
        for step in range(EP_LEN):
            tb_writer.add_scalar(f"avg_error", mean_avg_error[step], step)
            tb_writer.add_scalar(f"energy_norm", mean_energy_norm[step], step)

        for t in range(EP_LEN):
            df_mean_episode.append({
                "model": model_name,
                "metric": "avg_error",
                "step": t,
                "value": mean_avg_error[t]
            })
            df_mean_episode.append({
                "model": model_name,
                "metric": "energy_norm", 
                "step": t,
                "value": mean_energy_norm[t]
            })
        
        df_mean_scalar.append({
            "model": model_name,
            "avg_error_mean": mean_avg_error.mean(),
            "energy_norm_mean": mean_energy_norm.mean()
        })
        
        env.close()
        tb_writer.close()
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")




# find all *.zip in SWEEP_DIR/runs/**/models
model_paths = glob.glob(os.path.join(RUNS_DIR, "**", "models", "*.zip"), recursive=True)

print(f"\nFound {len(model_paths)} models.")

for mp in model_paths:
    eval_model(mp)

# save csv
pd.DataFrame(df_mean_episode).to_csv(MEAN_EPISODE_CSV, index=False)
pd.DataFrame(df_mean_scalar).to_csv(MEAN_SCALAR_CSV, index=False)


print("\n✅ all done.")
print(f"CSV: {MEAN_EPISODE_CSV}")
print(f"CSV: {MEAN_SCALAR_CSV}")
print(f"TensorBoard logdir: {TB_DIR}")
