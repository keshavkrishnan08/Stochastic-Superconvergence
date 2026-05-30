"""E46 - the metric-degree law across the whole configuration grid (not just one point).

E38 showed at one config that the variance-error order improves 1->2 at lambda*, so KL and W2^2 (quadratic)
go 2->4 while TV (linear) goes 1->2. Here we confirm that decomposition holds across a grid of (B,T,s2):
for every configuration the measured order at lambda* is 4 in KL and W2^2 and 2 in TV, and 2/2/1 just off
it. Exact deterministic recursion, no sampling.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
import io_utils as io
from diffusion import vp_const, set_dps
from lambda_star import lambda_star_vp
from recursion import v_terminal_closed
from metrics import kl_gauss, w2sq_gauss


def tv_gauss(v, s2):
    v = mp.mpf(v); s2 = mp.mpf(s2)
    if v == s2:
        return mp.mpf(0)
    a2, b2 = (v, s2) if v < s2 else (s2, v)
    a, b = mp.sqrt(a2), mp.sqrt(b2)
    c = mp.sqrt(2 * a2 * b2 / (b2 - a2) * mp.log(b / a))
    Phi = lambda z: (1 + mp.erf(z / mp.sqrt(2))) / 2
    return abs((2 * Phi(c / a) - 1) - (2 * Phi(c / b) - 1))


def run(dps=55):
    set_dps(dps)
    Bs = [2.0, 4.0, 8.0]; Ts = [4.0, 8.0]; s2s = [1.5, 2.0, 4.0]
    Ns = [512, 1024, 2048, 4096, 8192]
    def order(seq):
        return float(mp.log(seq[-3] / seq[-1]) / mp.log(mp.mpf(Ns[-1]) / Ns[-3]))
    rows = []
    for B in Bs:
        for T in Ts:
            for s2 in s2s:
                ls = lambda_star_vp(B, s2, T)
                if ls is None:
                    continue
                sched = vp_const(B, s2, T)
                def orders(u):
                    vs = [v_terminal_closed(sched, N, u) for N in Ns]
                    return (order([kl_gauss(v, s2) for v in vs]),
                            order([w2sq_gauss(v, s2) for v in vs]),
                            order([tv_gauss(v, s2) for v in vs]))
                kl_s, w2_s, tv_s = orders(ls ** 2)
                kl_o, w2_o, tv_o = orders((ls * mp.mpf("1.3")) ** 2)
                rows.append({"B": B, "T": T, "s2": s2, "KL_star": kl_s, "W2_star": w2_s, "TV_star": tv_s,
                             "KL_off": kl_o, "W2_off": w2_o, "TV_off": tv_o})
                io.log(f"  e46 B={B} T={T} s2={s2}: at* KL/W2/TV={kl_s:.2f}/{w2_s:.2f}/{tv_s:.2f} "
                       f"off={kl_o:.2f}/{w2_o:.2f}/{tv_o:.2f}")
    star = lambda k: (min(r[k] for r in rows), max(r[k] for r in rows))
    summ = {k: star(k) for k in ["KL_star", "W2_star", "TV_star", "KL_off", "W2_off", "TV_off"]}
    io.save("e46_metric_grid", {"config": {"Bs": Bs, "Ts": Ts, "s2s": s2s, "Ns": Ns},
                                "rows": rows, "summary_minmax": summ, "n": len(rows)})
    io.log(f"e46_metric_grid DONE: {len(rows)} configs; at* KL{summ['KL_star']} W2{summ['W2_star']} TV{summ['TV_star']}")


if __name__ == "__main__":
    run()
