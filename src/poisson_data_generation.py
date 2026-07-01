"""
Generates exact (f, u) training pairs for the 2D Poisson equation

    laplacian(u) = f      on (x, y) in [0, 1] x [0, 1]
    u = 0                 on the boundary  (Dirichlet)

using the closed-form sine-series solution. We never need to call a
numerical solver: if

    f(x, y) = sum_{m,n} c_mn * sin(m*pi*x) * sin(n*pi*y)

then the exact solution is

    u(x, y) = sum_{m,n} [ -c_mn / ((m*pi)**2 + (n*pi)**2) ] * sin(m*pi*x) * sin(n*pi*y)

because laplacian( sin(m pi x) sin(n pi y) ) = -((m pi)^2 + (n pi)^2) sin(m pi x) sin(n pi y).

Each sample uses a random handful of modes (m, n) with random coefficients,
so f looks like a smooth, randomly-shaped source field, and u is its EXACT
analytic solution (no discretization error in the ground truth).
"""

import os
import numpy as np
import pandas as pd

def make_grid(pixels_per_dim, pixels_at_boundary=True, domain_length=1.0):
    if pixels_at_boundary:
        coords = np.linspace(0.0, domain_length, pixels_per_dim)
    else:
        pixel_size = domain_length / pixels_per_dim
        coords = np.linspace(pixel_size / 2, domain_length - pixel_size / 2, pixels_per_dim)
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    return X, Y

def generate_one_sample(X, Y, n_modes=5, max_mode=6, coeff_scale=20.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    f = np.zeros_like(X)
    u = np.zeros_like(X)

    modes_m = rng.integers(1, max_mode + 1, size=n_modes)
    modes_n = rng.integers(1, max_mode + 1, size=n_modes)
    coeffs = rng.uniform(-coeff_scale, coeff_scale, size=n_modes)

    for m, n, c in zip(modes_m, modes_n, coeffs):
        basis = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        f += c * basis
        eigenvalue = -((m * np.pi) ** 2 + (n * np.pi) ** 2)
        u += (c / eigenvalue) * basis

    return f, u

def main():
    pixels_per_dim = 64
    pixels_at_boundary = True
    domain_length = 1.0

    n_train = 4000
    n_valid = 400

    n_modes = 5      # how many sine terms are summed per sample
    max_mode = 6      # highest mode index (m, n) allowed
    coeff_scale = 20.0  # magnitude of random source coefficients

    seed = 42
    rng = np.random.default_rng(seed)

    X, Y = make_grid(pixels_per_dim, pixels_at_boundary, domain_length)

    for split, n_samples in [('train', n_train), ('valid', n_valid)]:
        f_data, u_data = [], []
        for _ in range(n_samples):
            f, u = generate_one_sample(X, Y, n_modes=n_modes, max_mode=max_mode,
                                        coeff_scale=coeff_scale, rng=rng)
            f_data.append(f.flatten())
            u_data.append(u.flatten())

        save_dir = f'./data/poisson/{split}/'
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(f_data).to_csv(save_dir + 'f_data.csv', index=False, header=False)
        pd.DataFrame(u_data).to_csv(save_dir + 'u_data.csv', index=False, header=False)
        print(f'{split}: saved {n_samples} samples to {save_dir}')

    print('Poisson data generation finished.')

if __name__ == '__main__':
    main()
