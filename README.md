# Stochastic superconvergence in diffusion samplers

Code and stored results for the experiments on the cancellation churn of Gaussian diffusion samplers:
the point where the leading discretisation coefficient changes sign and the terminal KL convergence
order jumps from `N^-2` to `N^-4`. Everything needed to rerun the studies and regenerate the figures
is here; nothing else.

## Layout

- `src/ssc/` — the core library. The VP forward process and its marginal variance `V(t)`, the
  churn-parameterised reverse Euler–Maruyama recursion, the closed-form discretisation coefficients
  `C(λ)` and `c2(λ)`, the cancellation churn `λ*` (root finder, extended precision), KL and
  Wasserstein metrics, and the anisotropic / learned-score variants.
- `experiments/` — one script per study (`e01`…`e70`). The Gaussian-target studies are exact
  recursions at 40–120-digit precision (`mpmath`), so the predicted and measured curves agree to many
  figures with no fitting. The trained-score studies (`e32`, `e36`, `e39`, `e66`) train small networks
  in PyTorch. `make_figures.py` and `gallery.py` rebuild every figure from the stored results.
- `results/` — the JSON output of each run, plus the trained MNIST checkpoint. The figures
  regenerate from these without rerunning anything.
- `tests/` — a small identity/consistency suite for the exact arm (the recursion, the closed forms,
  the metric degree rule).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The exact-arm experiments need only `numpy` and `mpmath`; figures need `matplotlib`; the trained-score
experiments need `torch` and `torchvision` (and download MNIST on first run).

## Reproduce

```bash
# one experiment -> writes results/<name>.json
python experiments/e01_headline.py

# rebuild all figures from the stored results -> writes to figures/
python experiments/make_figures.py
python experiments/gallery.py

# the consistency tests (either runner works)
python tests/test_core.py    # standalone: prints each check, exits 0/1
pytest tests/                # or under pytest
```

Results are written under `results/` and figures under `figures/` (both created automatically). The
exact-arm runs finish in seconds to a few minutes on a CPU; the trained-score runs are the slow ones.
