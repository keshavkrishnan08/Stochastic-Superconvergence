# Stochastic Superconvergence

**The noise a diffusion sampler injects is an accuracy dial, not just a diversity dial. At one
specific churn value the sampler's leading discretization error cancels, and terminal KL falls as
`N^-4` instead of `N^-2`.**

![Terminal KL against step count on log-log axes for a Gaussian target: the probability-flow ODE, the reverse SDE and an over-churned sampler all fall as N to the minus 2, while the cancellation churn lambda* = 1.2705 falls as N to the minus 4 and reaches 7e-18 at 65,536 steps](docs/headline_kl.png)

Stochastic diffusion samplers sit on a one-parameter family between the deterministic
probability-flow ODE (churn `lambda = 0`) and the reverse SDE (`lambda = 1`). EDM exposes the same
knob as `S_churn`, usually set by sweeping sample quality. The view here is different. For an
Euler-Maruyama sampler the first-order error coefficient `C(lambda)` is a smooth function of the
churn that changes sign. At its root `lambda*` the first-order error vanishes and the convergence
order jumps. For a Gaussian target the score is linear and every iterate stays Gaussian, so the
sampler's terminal variance can be computed exactly by recursion at up to 120 significant digits,
with no sampling at all. Theory and measurement then agree to three or four decimals without
fitting anything.

```
linear forward SDE (VP, VE, EDM-style schedules), Gaussian target N(0, s^2)
    ->  churn-lambda reverse Euler-Maruyama:  v_{k+1} = (1 - A_k dt)^2 v_k + lambda^2 b_k^2 dt
    ->  closed-form leading coefficient C(lambda) = lim N (v_N - s^2)
    ->  lambda* = root of C:  the first-order error cancels
    ->  measured KL order 2 -> 4, then stress tests off the ideal case
```

Cancel the error instead of shrinking the step.

## The core idea

1. **The error has a known leading term.** For a linear schedule, the terminal variance error is
   `v_N - s^2 = C(lambda)/N + c2/N^2 + ...`, with `C` given in closed form as an integral against
   an integrating factor (`src/ssc/coefficient.py`). KL is quadratic in that error, so generically
   `KL ~ C^2 / (4 s^4 N^2)`.
2. **The coefficient changes sign.** On the VP schedule with `s^2 > 1`, the ODE over-contracts
   (`C(0) < 0`) and more churn pushes `C` up, so there is exactly one positive root `lambda*`. There
   the variance error drops to `O(N^-2)` and KL to `O(N^-4)`. For `s^2 <= 1` there is no root.
3. **The order jump follows the metric's degree.** Distances that are linear in the variance error
   (total variation, W1) improve from order 1 to 2. Quadratic ones (KL, reverse KL, W2^2, squared
   Hellinger, chi^2) improve from 2 to 4.
4. **Then push on every assumption.** Anisotropic covariances, other schedules, non-uniform grids,
   higher-order integrators, miscalibrated and trained scores, Gaussian mixtures, the literal EDM
   sampler, and MNIST. The Honesty section lists where the effect does not survive.

## What it does (measured, mostly on the exact Gaussian recursion)

| Question | Setup | Result | Result file |
|---|---|---|---|
| Does the order double at `lambda*`? | VP, s^2 = 2, B = 4, T = 5, N = 16 to 65,536 | tail KL order **4.000** at `lambda*` = 1.2705, vs 2.000 at lambda = 0, 1 and 2 | `e01_headline` |
| How big is the win? | KL at N = 1,024 | **1.2e-10** at `lambda*`, vs 8.7e-7 for the reverse SDE and 8.4e-6 for the ODE | `e01_headline` |
| Is the root unique, and only for s^2 > 1? | sign-change scan over 180 configs with s^2 > 1 and 60 with s^2 <= 1 | one root in every s^2 > 1 config, none otherwise; the bound behind uniqueness holds in **756 of 756** configs | `e34`, `e74` |
| How does `lambda*` scale? | lambda* / sqrt(s^2) as s^2 grows | approaches kappa of about **1.20** (1.2072 at s^2 = 4,096) | `e04`, `e16` |
| Is this just Richardson extrapolation? | total Euler steps to reach KL = 1e-10, s^2 = 4 | `lambda*` alone: about 1,439 steps. Richardson on the ODE: about 4,798. Both together: about 409 | `e54` |
| Does it survive other schedules? | VP ramp, VE with sigma^2 ~ t^2 (EDM-like), power-law step grids (rho = 1.5, 2, 3) | KL order 4.00 at each schedule's own root | `e08`, `e15`, `e45` |
| What if the score is slightly wrong? | miscalibrated score -x/(V + delta), s^2 = 2 | KL floor about **282x** lower at `lambda*`; steps to come within 3x of the ODE's floor scale as delta^-1.02 for the ODE vs delta^-0.47 at `lambda*` (3,852 vs 158 steps at delta = 3e-5) | `e64` |
| Does it stack with a better integrator? | two-stage RK2 midpoint rule (alpha = 0.5) | its own churn root (1.035) lifts KL order from 4.05 to **5.98** | `e63_integrator_root` |
| What about anisotropic targets? | per-eigenmode churn, d = 2 to 64 | order 4.02 to 4.03, vs 1.97 to 1.99 with one global churn | `e70` |
| Does it predict the literal EDM sampler? | EDM Algorithm 2 (Heun + S_churn) with an exact Gaussian denoiser, s^2 = 1, 4, 16 | predicted optimal S_churn 2.98, 4.24, 6.37 vs grid optimum 3.0, 4.25, 6.25 | `e35` |

Every number comes from a JSON in `results/`. The exact-arm runs carry no sampling noise, which is
why the measured orders land on 2.000 and 4.000.

## How it works

**Schedules.** `src/ssc/diffusion.py` defines linear forward SDEs `dx = a(t) x dt + b(t) dW` (VP with
constant or ramped beta, VE linear, VE quadratic). Each has Gaussian marginals `N(0, V(t))` with
`V' = 2aV + b^2` and an exact score `-x / V(t)`.

**Exact recursion.** With a Gaussian target and a linear score, the reverse Euler-Maruyama iterate
stays Gaussian, so `src/ssc/recursion.py` propagates its variance exactly in `mpmath` at 20 to 120
digits. KL, W2, TV and the other metrics follow in closed form (`src/ssc/metrics.py`). Euler-Maruyama,
an exact per-step map and a Heun variant are all available.

**Coefficients and roots.** `src/ssc/coefficient.py` gives `C(lambda)` in closed form, its
second-order partner `c2`, the score-error floor coefficient `D`, and Richardson estimates of each as
cross-checks. `src/ssc/lambda_star.py` finds `lambda*` by bisection in `u = lambda^2`.

**Beyond the exact arm.** `src/ssc/anisotropic.py` handles diagonal and rotated covariances.
`src/ssc/learned_score.py` adds a miscalibrated linear score (still exact) and small PyTorch score
networks trained by denoising score matching. `src/ssc/gmm.py` samples Gaussian mixtures with their
exact nonlinear score by Monte Carlo. `e35` runs the literal EDM sampler with an exact Gaussian
denoiser; `e36`, `e39` and `e60` to `e62` run it on trained denoisers (2D mixtures, MNIST) and score
samples by sliced Wasserstein distance.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python experiments/e01_headline.py      # one study -> results/e01_headline.json
python experiments/make_figures.py      # rebuild figures from stored results -> figures/
python experiments/gallery.py
python tests/test_core.py               # identity checks, prints each, exits 0 or 1
pytest tests/                           # same checks under pytest
```

Most experiment scripts skip when their result file already exists, so delete the JSON to
recompute. The exact-arm studies need only `numpy` and `mpmath` and finish in seconds to minutes on a
CPU. The trained-score studies need `torch` and `torchvision` (MNIST downloads on first run) and are
the slow ones.

## Layout

```
src/ssc/
  diffusion.py        linear schedules, marginal variance V(t), reverse drift
  recursion.py        exact terminal-variance recursion (EM, exact map, Heun)
  coefficient.py      closed-form C(lambda), c2, D; Richardson cross-checks
  lambda_star.py      the cancellation churn and its dependence on (s^2, B, T)
  metrics.py          Gaussian KL, W2, TV and friends; empirical W2; local order
  anisotropic.py      per-mode coefficients, compromise churn, per-mode churn
  learned_score.py    miscalibrated linear score; DSM-trained score networks
  gmm.py              Gaussian-mixture reverse sampler (Monte Carlo)
experiments/          e01 to e75 study scripts, multi-study suites, figure generators
results/              66 result JSONs and the trained MNIST U-Net (e39_unet.pt)
tests/test_core.py    schedule, recursion, coefficient, root and headline-order identities
```

## Honesty

* The clean result is for Gaussian targets with exact linear scores. That is what makes the
  recursion exact, and it is also the main limitation.
* Off the Gaussian the effect weakens or disappears. On a two-mode mixture with its exact score,
  sampled by Monte Carlo, the W2 order stays near 1.9 at every churn tested (`e10`). With exact
  density propagation on four mixture shapes, `lambda*` is still the best churn on the grid and cuts
  the variance error 8x to 43x at N = 64, but fitted orders range from 2.0 to 3.95 and the script
  itself flags them as unreliable off the Gaussian (`e72`).
* Trained scores mostly do not show it yet. On a trained 2D ring with the EDM Heun sampler, churn 0
  was best at 7 of 8 coarse step counts (`e62`). On MNIST the best churn lowers sliced Wasserstein
  from 0.0675 to 0.0650, inside the seed spread (`e39`). On trained 1D scores `lambda*` reached the
  ODE's target 2.4x to 6.9x faster on five of six networks, but the saving did not grow as the
  score improved, although the theory says it should (fitted exponent 0.02 against score RMSE,
  `e66`). Predicted vs measured optimal churn across trained 2D targets gives R^2 = 0.33 against the
  identity line, from 5 usable targets of 8 (`e60`).
* The `delta^-1/2` step law holds at s^2 = 1.5 and 2 but flattens at larger scales on the tested
  range: fitted exponents -0.32, -0.18 and -0.04 at s^2 = 4, 8 and 16 (`e70`).
* There is no root in some settings: VE with linear variance growth (`e08`), Heun with the standard
  weights (`e09`), and any single global churn on a non-degenerate anisotropic spectrum (`e06`).
* The closed-form coefficient matches Richardson extrapolation to 3.4e-5 relative error across 480
  configs, which misses the script's own 1e-5 bar (`e02` stores `pass: false`). The elementary
  anchors match to 1e-11.
* Not in the repo: the manuscript, the derivations that `src/ssc` docstrings cite
  (`theory/derivations.md`), rendered figures and MNIST sample grids, and stored results for `e50`
  and `e75`.

MIT licensed.
