# Physics-Informed Diffusion Models — Toy PDE Experiments

This repository is a fork/adaptation of a **Physics-Informed Diffusion Model** framework for solving partial differential equations (PDEs). It embeds physical constraints (PDE residuals) into the reverse diffusion process, combining score-based generative modeling with numerical PDE solving.

This fork focuses on **toy problem experiments** for the **2D Poisson** and **Helmholtz** equations, built on top of the original codebase's architecture and training pipeline.

> **Note:** This project is adapted from the [PhysicsInformedDiffusionModels](https://github.com/) codebase accompanying Bastek et al., *"Physics-Informed Diffusion Models"* (ICLR 2025).

## Overview

Traditional numerical methods for solving PDEs (Finite Difference, Finite Element, etc.) can be computationally expensive for high-dimensional or geometrically complex domains. This project uses a **score-based diffusion model** (U-Net backbone) constrained by governing physical equations, so that generated solution fields are both statistically plausible *and* physically consistent.

**Current status:** the diffusion pipeline has been adapted and validated on 2D Poisson and Helmholtz toy problems, achieving sub-1% relative L2 error against exact/reference solutions.

## Repository Structure

### `src/` — Core Codebase
* **Model Architecture & Diffusion**
    * `unet_model.py` — U-Net architecture used for noise prediction.
    * `denoising_utils.py`, `denoising_toy_utils.py` — Forward (noising) and reverse (denoising) diffusion process utilities.
* **Physics Constraints (Residuals)**
    * `residuals_poisson.py` — Physics-informed loss for Poisson's equation ($\nabla^2 u = f$).
    * `residuals_darcy.py` — Residuals for Darcy flow.
    * `residuals_mechanics_K.py` — Residuals for solid mechanics.
* **Data Processing & Generation**
    * `poisson_data_generation.py`, `darcy_data_generation.py` — Scripts to generate/preprocess training data.
    * `data_utils.py` — Data loading, batching, and normalization utilities.
* **Helper Utilities**
    * `grad_utils.py` — Automatic differentiation / gradient utilities for the physics loss.
    * `helper_plot.py` — Visualization tools for residual curves, error bars, and field comparisons.

### `run_1/` — Results & Evaluation
Outputs from training and inference runs:
* `residual_curve.png` — Physics-informed loss over training.
* `error_bar_chart.png` — Absolute/relative error bounds vs. exact solutions.
* `comparison_sample_*.png` — Predicted vs. exact field comparisons.
* `custom_inference/` — Step-by-step denoising sequence visualizations and specific test cases (e.g., `point_charge_inference.png`).

## What This Solves

Both problems are posed on a 2D domain and solved via reverse diffusion instead of a classical solver:

- **Poisson's equation:** $\nabla^2 u = f$ — recovering the field $u$ (e.g. an electrostatic potential) given a source term $f$ (e.g. a point charge distribution). See `run_1/custom_inference/point_charge_inference.png` for a worked example.
- **Helmholtz equation:** the frequency-domain wave equation, testing the model on oscillatory solution fields rather than the purely elliptic Poisson case.

The model is trained to denoise a random field into a valid solution while the PDE residual is enforced at each reverse diffusion step, so the output isn't just visually plausible — it's checked against the governing equation.

## What Was Adapted

Starting from the original codebase's architecture and training loop, the main adaptation work for this toy setup was:
- Wiring up `residuals_poisson.py` and the corresponding data generation for the specific Poisson test case (point-charge source).
- Extending/validating the pipeline on the Helmholtz case.
- Evaluation: comparing predicted vs. exact fields and tracking relative L2 error through the denoising trajectory.

## Results

On the 2D Poisson and Helmholtz toy problems, the model achieves **sub-1% relative L2 error** compared to reference solutions. See `run_1/error_bar_chart.png` and `run_1/comparison_sample_*.png` for quantitative and qualitative results, and `run_1/custom_inference/` for step-by-step denoising visualizations.

## Acknowledgments & Attribution

This repository builds on the codebase accompanying:
> Bastek, J.-H., et al. *"Physics-Informed Diffusion Models."* ICLR 2025.
