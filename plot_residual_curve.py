"""
plot_residual_curve.py  —  plots the PDE residual over training steps.
Run from the repo root:  python plot_residual_curve.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUN_NAME   = 'run_1'
OUTPUT_DIR = f'./eval_results/{RUN_NAME}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

stat_files = sorted(
    Path(f'./trained_models/{RUN_NAME}/training').glob('step_*/sample_statistics.csv'),
    key=lambda p: int(p.parent.name.replace('step_', ''))
)

if not stat_files:
    print('No sample_statistics.csv files found.')
    exit()

steps, mean_res, min_res, max_res = [], [], [], []

for sf in stat_files:
    step = int(sf.parent.name.replace('step_', ''))
    df   = pd.read_csv(sf)
    # strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    col = 'Residuals (abs)'
    if col not in df.columns:
        print(f'Column "{col}" not found in {sf}, skipping. Columns: {list(df.columns)}')
        continue
    vals = df[col].dropna().values
    steps.append(step)
    mean_res.append(vals.mean())
    min_res.append(vals.min())
    max_res.append(vals.max())

steps    = np.array(steps)
mean_res = np.array(mean_res)
min_res  = np.array(min_res)
max_res  = np.array(max_res)

fig, ax = plt.subplots(figsize=(9, 5))

ax.fill_between(steps, min_res, max_res, alpha=0.2, color='steelblue', label='Min–Max range')
ax.semilogy(steps, mean_res, marker='o', markersize=4, color='steelblue',
            linewidth=2, label='Mean residual |∇²u − f|')

ax.set_xlabel('Training iteration', fontsize=12)
ax.set_ylabel('Absolute PDE residual  |∇²u − f|  (log scale)', fontsize=12)
ax.set_title('PDE Residual over Training\n'
             'Poisson equation  ∇²u = f  on [0,1]²', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()

save_path = f'{OUTPUT_DIR}/residual_curve.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path}')
