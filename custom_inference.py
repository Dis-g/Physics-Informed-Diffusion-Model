"""
custom_inference.py — Run the trained Poisson diffusion model on a
custom, physically-designed source field f that was never in training data.

Scenario: Point charge at the centre of a grounded 2D box
  - f(x,y) ≈ delta(x-0.5, y-0.5)  (approximated as a narrow Gaussian)
  - u = 0 on all four walls (Dirichlet BC)
  - Physical meaning: electric potential due to a point charge
    inside a grounded conducting box

The model was trained on random sine-series f fields. This Gaussian
point-charge f is completely outside the training distribution —
so this tests genuine generalisation.

Run from repo root:
    python custom_inference.py
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_poisson import ResidualsPoisson

# ── Config ──────────────────────────────────────────────────────────────────
RUN_NAME   = 'run_1'
CHECKPOINT = 56000
PIXELS     = 64
DOMAIN     = 1.0
DEVICE     = 'cpu'
OUTPUT_DIR = f'./eval_results/{RUN_NAME}/custom_inference/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
config = yaml.safe_load(
    Path(f'./trained_models/{RUN_NAME}/model/model.yaml').read_text()
)
model = Unet3D(dim=32, channels=2, out_dim=1,
               sigmoid_last_channel=False).to(DEVICE)
state = torch.load(
    f'./trained_models/{RUN_NAME}/model/checkpoint_{CHECKPOINT}.pt',
    map_location=DEVICE
)
model.load_state_dict(state['model'] if 'model' in state else state)
model.eval()
print(f'Loaded checkpoint: checkpoint_{CHECKPOINT}.pt')

diffusion = DenoisingDiffusion(n_steps=config['diff_steps'], device=DEVICE)
residuals = ResidualsPoisson(
    model=model, fd_acc=2, pixels_per_dim=PIXELS,
    pixels_at_boundary=True, device=DEVICE, domain_length=DOMAIN,
)

# ── Build custom source field f: point charge at centre ──────────────────────
coords = np.linspace(0.0, DOMAIN, PIXELS)
X, Y   = np.meshgrid(coords, coords, indexing='ij')

# Narrow Gaussian centred at (0.5, 0.5) — approximates a point charge
# sigma controls how sharp the peak is; smaller = more point-like
sigma  = 0.04
cx, cy = 0.5, 0.5
amplitude = 50.0   # controls the charge strength
f_np = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

# Zero at boundary (model expects this)
f_np[ 0, :] = 0; f_np[-1, :] = 0
f_np[:,  0] = 0; f_np[:, -1] = 0

# Tensor: [1, 1, 64, 64]
f_tensor = torch.tensor(f_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ── Run reverse diffusion ─────────────────────────────────────────────────────
print('Running reverse diffusion on point-charge source field...')
sample_shape = (1, 1, PIXELS, PIXELS)
with torch.no_grad():
    x_seq, _ = diffusion.p_sample_loop(
        conditioning_input=(f_tensor, None, torch.zeros(sample_shape)),
        shape=sample_shape,
        save_output=True,
        surpress_noise=True,
        residual_func=residuals,
        eval_residuals=False,
    )
u_pred = x_seq[-1][0, 0].numpy()   # [64, 64]

# ── Compute PDE residual of the prediction ────────────────────────────────────
# Finite-difference Laplacian
dx = DOMAIN / (PIXELS - 1)
laplacian_u = (
    np.roll(u_pred,  1, axis=0) + np.roll(u_pred, -1, axis=0) +
    np.roll(u_pred,  1, axis=1) + np.roll(u_pred, -1, axis=1) -
    4 * u_pred
) / dx**2
residual_field = laplacian_u - f_np
mean_residual  = np.mean(np.abs(residual_field[1:-1, 1:-1]))  # ignore boundaries
print(f'Mean PDE residual |∇²u − f|: {mean_residual:.4f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    'Custom Inference: Point Charge at Centre  |  ∇²u = f  on [0,1]²\n'
    'Source field never seen during training — testing generalisation',
    fontsize=13
)

# Source field
im0 = axes[0].imshow(f_np.T, cmap='RdBu_r', origin='lower', aspect='equal',
                      extent=[0, 1, 0, 1])
axes[0].set_title('Source field  f(x,y)\n(Point charge at centre)', fontsize=11)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0])

# Predicted potential
im1 = axes[1].imshow(u_pred.T, cmap='plasma', origin='lower', aspect='equal',
                      extent=[0, 1, 0, 1])
axes[1].set_title('Predicted  u(x,y)\n(Electric potential)', fontsize=11)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
plt.colorbar(im1, ax=axes[1])

# PDE residual map
im2 = axes[2].imshow(np.abs(residual_field[1:-1, 1:-1]).T,
                      cmap='hot', origin='lower', aspect='equal',
                      extent=[dx, 1-dx, dx, 1-dx])
axes[2].set_title(
    f'PDE residual  |∇²u − f|\nMean = {mean_residual:.4f}', fontsize=11
)
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
save_path = f'{OUTPUT_DIR}/point_charge_inference.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path}')

# ── Also plot the denoising sequence ─────────────────────────────────────────
# Show how the model goes from noise → solution in steps
steps_to_show = [0, 20, 50, 80, 99]
fig2, axes2 = plt.subplots(1, len(steps_to_show), figsize=(14, 3.5))
fig2.suptitle('Reverse Diffusion Process: Noise → Solution', fontsize=13)
for ax, step in zip(axes2, steps_to_show):
    frame = x_seq[step][0, 0].numpy()
    vmin, vmax = u_pred.min(), u_pred.max()
    ax.imshow(frame.T, cmap='plasma', origin='lower', aspect='equal',
              vmin=vmin, vmax=vmax)
    label = 'Pure noise' if step == 0 else (
        'Final solution' if step == steps_to_show[-1] else f'Step {step}'
    )
    ax.set_title(label, fontsize=10)
    ax.axis('off')
plt.tight_layout()
save_path2 = f'{OUTPUT_DIR}/denoising_sequence.png'
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path2}')

print('\nDone. Check eval_results/run_1/custom_inference/')
