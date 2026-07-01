"""
evaluate.py  —  Evaluation and plotting script for the trained Poisson diffusion model.

Loads the trained checkpoint, runs reverse diffusion on held-out validation samples,
plots f / true u / predicted u side by side, and prints relative L2 errors.

Run from the repo root:
    python evaluate.py
"""

import os, yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from src.data_utils import Dataset
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_poisson import ResidualsPoisson
from torch.utils.data import DataLoader

# ─── Config ────────────────────────────────────────────────────────────────────
RUN_NAME       = 'run_1'
CHECKPOINT     = 56000          # which saved checkpoint to load (match a saved step number)
N_EVAL_SAMPLES = 8              # how many samples to evaluate and plot
PIXELS         = 64
DOMAIN_LENGTH  = 1.
PIXELS_AT_BOUNDARY = True
FD_ACC         = 2
DEVICE         = 'cpu'
OUTPUT_DIR     = f'./eval_results/{RUN_NAME}/'

# ─── Load config and model ─────────────────────────────────────────────────────
config = yaml.safe_load(Path(f'./trained_models/{RUN_NAME}/model/model.yaml').read_text())

diff_steps = config['diff_steps']

model = Unet3D(
    dim      = 32,
    channels = 2,        # u (output, 1 ch) + f (conditioning, 1 ch)
    out_dim  = 1,        # only predict u
    sigmoid_last_channel = False,
).to(DEVICE)

checkpoint_path = f'./trained_models/{RUN_NAME}/model/checkpoint_{CHECKPOINT}.pt'
state = torch.load(checkpoint_path, map_location=DEVICE)
# support both raw state_dict and wrapped checkpoints
if isinstance(state, dict) and 'model' in state:
    model.load_state_dict(state['model'])
else:
    model.load_state_dict(state)
model.eval()
print(f'Loaded checkpoint: {checkpoint_path}')

# ─── Diffusion + residual objects ──────────────────────────────────────────────
diffusion = DenoisingDiffusion(n_steps=diff_steps, device=DEVICE)
residuals  = ResidualsPoisson(
    model              = model,
    fd_acc             = FD_ACC,
    pixels_per_dim     = PIXELS,
    pixels_at_boundary = PIXELS_AT_BOUNDARY,
    device             = DEVICE,
    domain_length      = DOMAIN_LENGTH,
)

# ─── Validation data ───────────────────────────────────────────────────────────
ds_valid = Dataset(
    ('./data/poisson/valid/f_data.csv', './data/poisson/valid/u_data.csv'),
    use_double=False,
)
dl_valid = DataLoader(ds_valid, batch_size=N_EVAL_SAMPLES, shuffle=False)
batch = next(iter(dl_valid)).to(DEVICE)           # [N, 2, 64, 64]
f_cond  = batch[:, :1]                            # [N, 1, 64, 64]  source field
u_true  = batch[:, 1:2]                           # [N, 1, 64, 64]  exact solution

# ─── Run reverse diffusion (inference) ─────────────────────────────────────────
print(f'Running reverse diffusion on {N_EVAL_SAMPLES} validation samples …')
sample_shape = (N_EVAL_SAMPLES, 1, PIXELS, PIXELS)
with torch.no_grad():
    # conditioning_input for poisson: (f_cond, None, u_true)
    # p_sample_loop only uses f_cond inside p_sample; u_true is stored but not fed to model
    x_seq, _ = diffusion.p_sample_loop(
        conditioning_input = (f_cond, None, u_true),
        shape              = sample_shape,
        save_output        = True,
        surpress_noise     = True,
        residual_func      = residuals,
        eval_residuals     = False,
    )
u_pred = x_seq[-1].to(DEVICE)                     # [N, 1, 64, 64]

# ─── Relative L2 error ─────────────────────────────────────────────────────────
def rel_l2(pred, true):
    diff  = (pred - true).reshape(pred.shape[0], -1)
    denom = true.reshape(true.shape[0], -1)
    return (diff.norm(dim=1) / (denom.norm(dim=1) + 1e-10)).numpy()

errors = rel_l2(u_pred.cpu(), u_true.cpu())
print('\nRelative L2 errors per sample:')
for i, e in enumerate(errors):
    print(f'  sample {i}: {e:.4f}  ({e*100:.2f}%)')
print(f'\nMean relative L2 error: {errors.mean():.4f}  ({errors.mean()*100:.2f}%)')

# ─── Plotting ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Side-by-side comparison for each sample
for i in range(N_EVAL_SAMPLES):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    fi    = f_cond[i, 0].cpu().numpy()
    utrue = u_true[i, 0].cpu().numpy()
    upred = u_pred[i, 0].cpu().numpy()
    err   = np.abs(upred - utrue)

    vmin_u = min(utrue.min(), upred.min())
    vmax_u = max(utrue.max(), upred.max())

    im0 = axes[0].imshow(fi,    cmap='RdBu_r', origin='lower', aspect='equal')
    axes[0].set_title('Source field  f(x,y)', fontsize=12)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(utrue, cmap='viridis', origin='lower', aspect='equal',
                          vmin=vmin_u, vmax=vmax_u)
    axes[1].set_title('True u  (exact solution)', fontsize=12)
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(upred, cmap='viridis', origin='lower', aspect='equal',
                          vmin=vmin_u, vmax=vmax_u)
    axes[2].set_title(f'Predicted u  (diffusion model)\nRel. L2 = {errors[i]*100:.2f}%', fontsize=12)
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle(f'Poisson Equation — Sample {i}  |  ∇²u = f  on [0,1]²', fontsize=13)
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/comparison_sample_{i}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

# 2. Error bar chart across all samples
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(N_EVAL_SAMPLES), errors * 100, color='steelblue', edgecolor='white')
ax.axhline(errors.mean() * 100, color='red', linestyle='--', label=f'Mean = {errors.mean()*100:.2f}%')
ax.set_xlabel('Sample index', fontsize=11)
ax.set_ylabel('Relative L2 error (%)', fontsize=11)
ax.set_title('Per-sample error: Diffusion model vs exact solution', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/error_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUTPUT_DIR}error_bar_chart.png')

# 3. Training loss curve from sample_statistics.csv files
stat_files = sorted(Path(f'./trained_models/{RUN_NAME}/training').glob('step_*/sample_statistics.csv'))
if stat_files:
    steps, mean_residuals = [], []
    for sf in stat_files:
        import pandas as pd
        step = int(sf.parent.name.replace('step_', ''))
        df   = pd.read_csv(sf)
        if 'Residual' in df.columns:
            steps.append(step)
            mean_residuals.append(df['Residual'].mean())
    if steps:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(steps, mean_residuals, marker='o', color='steelblue', linewidth=1.5)
        ax.set_xlabel('Training iteration', fontsize=11)
        ax.set_ylabel('Mean PDE residual  |∇²u − f|  (log scale)', fontsize=11)
        ax.set_title('PDE residual over training', fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/residual_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUTPUT_DIR}residual_curve.png')

print('\nDone. All figures saved to', OUTPUT_DIR)
