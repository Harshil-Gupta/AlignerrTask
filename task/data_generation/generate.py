"""Lorenz-96 forcing-inversion data generator (repository-private; not shipped in the image).

Data-generating process
-----------------------
Latent dynamics: the Lorenz-96 ring with N = 40 nodes, driven by a spatially
smooth, time-constant forcing field F and by an UNRESOLVED SUB-GRID TENDENCY
U that the public task statement discloses only as a family:

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F_i + U_i(t),   indices mod N
    U_i(t)  = p(x_i(t)) + eta_i(t)
    p(x)    = c0 + c1 x + c2 x^2 + c3 x^3        fixed for ALL trajectories
    eta_i   <- phi * eta_i + s * sqrt(1 - phi^2) * N(0, 1)   AR(1) red noise,
              updated once per RK4 sub-step, phi = exp(-h / tau)

The polynomial coefficients (POLY), the red-noise stationary std (ETA_STD) and
its correlation time (ETA_TAU) are HIDDEN constants that live only in this
file. The public prompt states that a state-dependent, temporally correlated
sub-grid tendency of minority magnitude exists, is identical in distribution
across every training and test trajectory, and is never disclosed in form or
value; the training labels define the forcing convention (the target is F,
not F plus the time-mean of U).

Forcing prior: each trajectory draws base_i ~ N(8, 2^2) i.i.d. over the 40
nodes and smooths it with the CIRCULAR 3-tap kernel [0.2, 0.6, 0.2] (np.roll),
so the forcing distribution is translation invariant on the ring.

Initial condition x(0) ~ U[0, 1)^N, integrated through a burn-in of 10 time
units before 200 samples are recorded at t = linspace(0, 5, 200).

Integration: batched fixed-step classical RK4 in numpy with SUBSTEPS sub-steps
per output interval (h = (5/199)/SUBSTEPS). The red noise is held constant
within a sub-step. The scheme is part of the hidden process definition (the
public prompt discloses neither the step nor the noise update).

Observation model: each trajectory carries two hidden nuisance parameters,
    gamma ~ U[0.45, 0.80]   (observation-map exponent)
    sigma ~ U[1.6, 3.4]     (observation noise standard deviation)
drawn independently of F and of each other. The recorded observation is
    y = sign(x) (|x| + 1)^gamma + eps,      eps ~ N(0, sigma^2) i.i.d.
gamma and sigma are never published; the test-split values are written to a
private scorer-side file for documentation only.

Reproducibility: every trajectory owns an independent RNG spawned from one
np.random.SeedSequence(master_seed) child. All of a trajectory's draws (F
base, x0, gamma, sigma, initial eta, the red-noise stream, the observation
noise) come from that child in a fixed order, and the integration is
vectorised over trajectories, so the dataset is identical regardless of batch
size or scheduling.

Developer flag: L96_DEV_SUBGRID (default "1"). Setting it to "0" integrates the
DISCLOSED idealised equation (U = 0) with the same seeds; this exists only for
author-side mismatch diagnostics and prints a loud warning. Never generate
committed data with it off.

CLI contract: the public training split uses seed 42. The test split needs
LBX_PRIVATE_CHALLENGE_SEED in the environment (only its sha256 digest prefix is
printed). Output files:
    data/train_dynamics.npy          float32 (2000, 200, 40)
    data/train_forcing.npy           float32 (2000, 40)
    data/test_dynamics.npy           float32 (500, 200, 40)
    scorer/data/test_forcing.npy     float32 (500, 40)     (hidden truth)
    scorer/data/test_nuisance.npy    float32 (500, 2)      [gamma, sigma], private
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

N_NODES = 40
T_STEPS = 200
T_WINDOW = 5.0
BURN_IN = 10.0
FORCING_MEAN = 8.0
FORCING_STD = 2.0
FORCING_KERNEL = (0.2, 0.6, 0.2)
GAMMA_RANGE = (0.45, 0.80)
SIGMA_RANGE = (1.6, 3.4)
N_TRAIN = 2000
N_TEST = 500
PUBLIC_TRAIN_SEED = 42
SUBSTEPS = 5                     # h = (5/199)/5 ~= 0.005025

# ---- hidden sub-grid process (fixed across all trajectories; repo-private) ----
POLY = (0.18, -0.072, 0.012, -0.0021)   # c0..c3 of p(x); std(p(x)) ~ 0.43 on the attractor
ETA_STD = 0.30                          # stationary std of the AR(1) red noise
ETA_TAU = 0.20                          # correlation time (time units); phi = exp(-h / tau)

SUBGRID_ENABLED = os.environ.get("L96_DEV_SUBGRID", "1") != "0"


def circular_smooth(base: np.ndarray, kernel=FORCING_KERNEL) -> np.ndarray:
    """Periodic 3-tap smoothing along the last axis: F_i = k0*base_{i-1} + k1*base_i + k2*base_{i+1}."""
    k0, k1, k2 = kernel
    return k0 * np.roll(base, 1, axis=-1) + k1 * base + k2 * np.roll(base, -1, axis=-1)


def obs_transform(x: np.ndarray, gamma) -> np.ndarray:
    return np.sign(x) * (np.abs(x) + 1.0) ** gamma


def subgrid_poly(x: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3 = POLY
    return c0 + c1 * x + c2 * x * x + c3 * x * x * x


def rhs(x: np.ndarray, F: np.ndarray, U) -> np.ndarray:
    """Vectorised Lorenz-96 right-hand side on the periodic ring plus the sub-grid tendency."""
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + F + U


def rk4_step(x, F, U, h):
    k1 = rhs(x, F, U)
    k2 = rhs(x + 0.5 * h * k1, F, U)
    k3 = rhs(x + 0.5 * h * k2, F, U)
    k4 = rhs(x + h * k3, F, U)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_batch(x0, F, rngs, n_out, dt_out, burn_steps, record_U=False):
    """x0, F: (B, N). rngs: B per-trajectory Generators. Returns x (B, n_out, N) [, U (B, n_out, N)]."""
    B, N = x0.shape
    h = dt_out / SUBSTEPS
    phi = np.exp(-h / ETA_TAU)
    s = ETA_STD * np.sqrt(1.0 - phi * phi)
    x = x0.astype(np.float64).copy()
    total = burn_steps + (n_out - 1) * SUBSTEPS
    if SUBGRID_ENABLED:
        eta = np.stack([r.normal(0.0, ETA_STD, N) for r in rngs])
        # per-trajectory bulk draws keep each noise stream independent of B and of scheduling
        noise = np.stack([r.normal(0.0, 1.0, (total, N)).astype(np.float32) for r in rngs])
    else:
        eta = np.zeros((B, N))
        noise = None
    ctr = [0]

    def step(x, eta):
        if SUBGRID_ENABLED:
            eta = phi * eta + s * noise[:, ctr[0]]
            ctr[0] += 1
            U = subgrid_poly(x) + eta
        else:
            U = 0.0
        return rk4_step(x, F, U, h), eta, U

    for _ in range(burn_steps):
        x, eta, _ = step(x, eta)
    xs = np.empty((B, n_out, N))
    Us = np.empty((B, n_out, N)) if record_U else None
    xs[:, 0] = x
    if record_U:
        Us[:, 0] = (subgrid_poly(x) + eta) if SUBGRID_ENABLED else 0.0
    for t in range(1, n_out):
        for _ in range(SUBSTEPS):
            x, eta, U = step(x, eta)
        xs[:, t] = x
        if record_U:
            Us[:, t] = U if SUBGRID_ENABLED else 0.0
    return (xs, Us) if record_U else xs


def generate_dataset_full(num_samples: int, seed: int, return_x: bool = True):
    """Full diagnostic generator: (Y, F, nuis[, x_latent, U])."""
    master = np.random.SeedSequence(seed)
    children = master.spawn(num_samples)
    rngs = [np.random.default_rng(c) for c in children]
    base = np.stack([r.normal(FORCING_MEAN, FORCING_STD, N_NODES) for r in rngs])
    F = circular_smooth(base)
    x0 = np.stack([r.random(N_NODES) for r in rngs])
    gamma = np.array([r.uniform(*GAMMA_RANGE) for r in rngs])
    sigma = np.array([r.uniform(*SIGMA_RANGE) for r in rngs])
    dt_out = T_WINDOW / (T_STEPS - 1)
    burn_steps = int(round(BURN_IN / (dt_out / SUBSTEPS)))
    x, U = integrate_batch(x0, F, rngs, T_STEPS, dt_out, burn_steps, record_U=True)
    noise = np.stack([r.normal(0.0, 1.0, (T_STEPS, N_NODES)) for r in rngs])
    y = obs_transform(x, gamma[:, None, None]) + sigma[:, None, None] * noise
    nuis = np.stack([gamma, sigma], 1).astype(np.float32)
    out = (y.astype(np.float32), F.astype(np.float32), nuis)
    if return_x:
        out = out + (x.astype(np.float32), U.astype(np.float32))
    return out


def generate_dataset(num_samples: int, seed: int, workers: int | None = None):
    """Return (Y float32 (n, T, N), F float32 (n, N), nuisance float32 (n, 2)). `workers` is accepted for
    CLI compatibility; the integration is vectorised and needs no process pool."""
    y, F, nuis = generate_dataset_full(num_samples, seed, return_x=False)
    return y, F, nuis


if __name__ == "__main__":
    if not SUBGRID_ENABLED:
        print("WARNING: L96_DEV_SUBGRID=0 -> integrating the DISCLOSED idealised equation only. "
              "Developer diagnostics only; never commit data generated this way.", file=sys.stderr)
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    scorer_data_dir = base_dir / "scorer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scorer_data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating train set...")
    t0 = time.time()
    Y_train, F_train, _nuis_train = generate_dataset(N_TRAIN, seed=PUBLIC_TRAIN_SEED)
    np.save(data_dir / "train_dynamics.npy", Y_train)
    np.save(data_dir / "train_forcing.npy", F_train)
    print(f"Train set done in {time.time() - t0:.1f}s")
    # Training nuisance values are deliberately NOT written anywhere.

    seed_str = os.environ.get("LBX_PRIVATE_CHALLENGE_SEED")
    if not seed_str:
        print("ERROR: LBX_PRIVATE_CHALLENGE_SEED environment variable must be set to generate test sets.", file=sys.stderr)
        sys.exit(1)
    private_seed = int(seed_str)
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16]
    print(f"Generating hidden test set (seed digest {digest}...).")
    t0 = time.time()
    Y_test, F_test, nuis_test = generate_dataset(N_TEST, seed=private_seed)
    np.save(data_dir / "test_dynamics.npy", Y_test)
    np.save(scorer_data_dir / "test_forcing.npy", F_test)
    np.save(scorer_data_dir / "test_nuisance.npy", nuis_test)
    print(f"Test set done in {time.time() - t0:.1f}s")
    print("Data generation complete.")
