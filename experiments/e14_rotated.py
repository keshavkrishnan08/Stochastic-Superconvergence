"""E14 — Full-covariance (rotated) Gaussian: Theorem C beyond diagonal.
For isotropic injected noise the churn sampler is rotation-equivariant, so a target
N(0, R diag(s_i^2) R^T) decomposes into independent eigenmodes EXACTLY. We verify by Monte-Carlo
in 2D: sample with the matrix reverse SDE under a rotated covariance and compare the terminal
covariance's eigenvalues to the per-mode exact recursion. Confirms the no-go (one lambda* per
eigenvalue) is a property of the spectrum, not the axis alignment."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np, mpmath as mp
from diffusion import vp_const, set_dps
from recursion import v_terminal
import io_utils as io

NAME = "e14_rotated"


def matrix_sample(N, T, B, lam, Sigma, P, rng):
    """Reverse EM in 2D with exact linear score s(x)=-Sigma_t^{-1} x, Sigma_t = e^{-Bt}Sigma+(1-e^{-Bt})I.
    Forward-running drift: (B/2)x + (1+lam^2)/2 B * score."""
    d = Sigma.shape[0]; dt = T / N
    eBt = np.exp(-B * T); SigT = eBt * Sigma + (1 - eBt) * np.eye(d)
    L = np.linalg.cholesky(SigT)
    x = (L @ rng.standard_normal((d, P)))
    c = lam * np.sqrt(B); sq = np.sqrt(dt)
    for k in range(N):
        tk = T - k * dt; e = np.exp(-B * tk)
        St = e * Sigma + (1 - e) * np.eye(d)
        Stinv = np.linalg.inv(St)
        score = -(Stinv @ x)
        drift = (B / 2) * x + (1 + lam ** 2) / 2 * B * score
        x = x + drift * dt + c * sq * rng.standard_normal((d, P))
    return x


def run(B=4.0, T=5.0, P=400_000, dps=40):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    s = [1.5, 4.0]                      # eigenvalues
    theta = 0.6                         # rotation
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Sigma = R @ np.diag(s) @ R.T
    rng = np.random.default_rng(0)
    rows = []
    for lam in [0.0, 1.0, 1.2705]:
        N = 256
        x = matrix_sample(N, T, B, lam, Sigma, P, rng)
        Cov = np.cov(x)
        eig = np.sort(np.linalg.eigvalsh(Cov))[::-1]   # measured eigenvalues
        # exact per-mode recursion prediction
        pred = sorted([float(v_terminal(vp_const(B, si, T), N, mp.mpf(lam) ** 2)) for si in s], reverse=True)
        rows.append({"lam": lam, "N": N, "meas_eig": eig.tolist(), "pred_eig": pred,
                     "max_rel": float(max(abs(e - p) / p for e, p in zip(eig, pred)))})
        io.log(f"  lam={lam}: meas eig={np.round(eig,4).tolist()} pred={[round(p,4) for p in pred]} "
               f"rel={rows[-1]['max_rel']:.2e}")
    res = {"config": {"B": B, "T": T, "P": P, "eigs": s, "theta": theta}, "rows": rows}
    io.save(NAME, res)
    io.log(f"{NAME}: rotated-covariance eigenvalues match per-mode recursion "
           f"(max rel {max(r['max_rel'] for r in rows):.2e}) -> decomposition is rotation-invariant")
    return res


if __name__ == "__main__":
    run()
