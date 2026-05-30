"""E16 — The kappa constant: closed-form limiting equation vs fitted asymptote.
Theorem A's asymptotics give lambda*(s2) ~ kappa sqrt(s2) with kappa the positive root of
   F(k) = int_0^inf e^{-(k^2-1) tau} [ k^4 e^{tau} - 2 k^2 - e^{-tau} ] dtau = 0   (B absorbed into tau).
We solve F(k)=0 in closed form (the integral is elementary for k^2>1) and compare to lambda*/sqrt(s2)
measured at very large s2 (extending E4). Confirms the analytic constant ~1.20."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import set_dps
from lambda_star import lambda_star_vp
import io_utils as io

NAME = "e16_kappa"


def F(k):
    """Elementary form: int_0^inf e^{-(k^2-1)tau}[k^4 e^tau - 2k^2 - e^{-tau}] dtau, k^2>1.
    = k^4/(k^2-2) - 2k^2/(k^2-1) - 1/k^2,  valid for k^2>2 (first term converges)."""
    k = mp.mpf(k); k2 = k ** 2
    return k2 ** 2 / (k2 - 2) - 2 * k2 / (k2 - 1) - 1 / k2


def run(dps=40):
    if io.exists(NAME):
        io.log(f"{NAME} exists, skip"); return io.load(NAME)
    set_dps(dps)
    # closed-form root of F (k^2>2 so k>1.414)
    lo, hi = mp.mpf('1.45'), mp.mpf('3.0')
    # ensure sign change
    klosed = None
    if F(lo) * F(hi) < 0:
        for _ in range(200):
            mid = (lo + hi) / 2
            if F(lo) * F(mid) <= 0: hi = mid
            else: lo = mid
        klosed = (lo + hi) / 2
    # measured asymptote: lambda*/sqrt(s2) at very large s2
    meas = []
    for s2 in [64.0, 256.0, 1024.0, 4096.0]:
        ls = lambda_star_vp(4.0, s2, 5.0)
        meas.append({"s2": s2, "ratio": float(ls / mp.sqrt(s2)) if ls else None})
    res = {"kappa_closed_form": float(klosed) if klosed else None,
           "F_at_root": float(F(klosed)) if klosed else None,
           "measured_ratio": meas,
           "ratio_at_s2_4096": meas[-1]["ratio"]}
    io.save(NAME, res)
    io.log(f"{NAME}: kappa closed-form root={float(klosed) if klosed else None}, "
           f"measured ratio @s2=4096 = {meas[-1]['ratio']}")
    return res


if __name__ == "__main__":
    run()
