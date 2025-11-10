import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SWEEP_DIR = os.path.join(ROOT, 'results', 'hyper_sweep')

MEAN_SCALAR_CSV  = os.path.join(SWEEP_DIR, 'mean_scalar.csv')
RANKING_CSV = os.path.join(SWEEP_DIR, 'ranking.csv')

df = pd.read_csv(MEAN_SCALAR_CSV)

df['err'] = df['avg_error_mean'] / 200.0
df['eng'] = df['energy_norm_mean']

df['score'] = 0.6 * df['err'] + 0.4 * df['eng']

df = df.sort_values('score')

df.to_csv(RANKING_CSV, index=False)

print("✅ ranking saved at:", RANKING_CSV)
print(df.head(5))
