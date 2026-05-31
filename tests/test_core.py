"""Core identity tests. Must all PASS before any experiment launch.

Run as a script:  python3 tests/test_core.py    (prints each check, exits 0/1)
Or under pytest:  pytest tests/                  (collects test_core_identities)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import mpmath as mp
from diffusion import vp_const, vp_ramp, ve_linear, ve_quad, set_dps
from recursion import v_terminal, v_terminal_closed
from coefficient import C_closed, C_closed_vp, C0_vp, C1_vp, C_richardson
from lambda_star import lambda_star_vp, u_star_general, u_star_richardson
from metrics import kl_gauss, local_order

set_dps(60)


def _checks():
    """Run every core identity check, printing each; raise AssertionError on the first failure.
    Returns the number of checks run, so the script runner can report the total."""
    n = [0]

    def check(name, cond, detail=""):
        n[0] += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        assert cond, f"{name}  {detail}"

    s2, B, T = mp.mpf(2), mp.mpf(4), mp.mpf(5)
    sched = vp_const(B, s2, T)

    print("== 1. schedule consistency V' = 2aV+b2, and V'' ==")
    for sc in [vp_const(4, 2, 5), vp_ramp(2, 1, 2, 5), ve_linear(1.5, 2, 5), ve_quad(0.5, 2, 5)]:
        t = mp.mpf('1.3')
        e1 = abs(sc.Vp(t) - (2 * sc.a(t) * sc.V(t) + sc.b2(t)))
        h = mp.mpf('1e-20'); fd = (sc.Vp(t + h) - sc.Vp(t - h)) / (2 * h)
        e2 = abs(sc.Vpp(t) - fd)
        check(f"{sc.name} consistency", e1 < mp.mpf('1e-40') and e2 < mp.mpf('1e-12'),
              f"V'err={mp.nstr(e1,2)} V''err={mp.nstr(e2,2)}")

    print("== 2. recursion EM == closed-form product/sum ==")
    for lam in [0.0, 1.0, 2.0]:
        for N in [64, 512]:
            a = v_terminal(sched, N, lam ** 2, integrator="EM")
            b = v_terminal_closed(sched, N, lam ** 2)
            check(f"EM==closed lam={lam} N={N}", abs(a - b) / abs(b) < mp.mpf('1e-45'),
                  f"reldiff={mp.nstr(abs(a-b)/abs(b),2)}")

    print("== 3. C_closed_vp == C_closed(general,ODE) == C_richardson, and anchors ==")
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        Cv = C_closed_vp(B, lam, s2, T); Cr, _ = C_richardson(sched, lam ** 2)
        check(f"C_vp==C_rich lam={lam}", abs(Cv - Cr) / abs(Cv) < mp.mpf('1e-6'), f"rd={mp.nstr(abs(Cv-Cr)/abs(Cv),2)}")
    Cv = C_closed_vp(B, 1.0, s2, T); Cg = C_closed(sched, 1.0)   # one general-ODE cross-check (~3.5s)
    check("C_vp==C_gen(ODE) lam=1.0", abs(Cv - Cg) / abs(Cv) < mp.mpf('1e-8'), f"rd={mp.nstr(abs(Cv-Cg)/abs(Cv),2)}")
    check("anchor C(0)", abs(C0_vp(B, s2, T) - C_closed_vp(B, 0, s2, T)) < mp.mpf('1e-12'),
          f"elem={mp.nstr(C0_vp(B,s2,T),10)}")
    check("anchor C(1)", abs(C1_vp(B, s2, T) - C_closed_vp(B, 1, s2, T)) < mp.mpf('1e-12'),
          f"elem={mp.nstr(C1_vp(B,s2,T),10)}")
    check("C(0) < 0 (ODE over-contracts for s2>1)", C0_vp(B, s2, T) < 0, f"C0={mp.nstr(C0_vp(B,s2,T),6)}")

    print("== 4. lambda* root: present for s2>1, absent for s2<=1 ==")
    ls = lambda_star_vp(B, s2, T)
    check("lam* exists s2=2", ls is not None and abs(ls - mp.mpf('1.27050')) < mp.mpf('1e-3'),
          f"lam*={mp.nstr(ls,8) if ls else None}")
    check("no root s2=0.5", lambda_star_vp(B, 0.5, T) is None)
    check("no root s2=1.0", lambda_star_vp(B, 1.0, T) is None)
    check("general-schedule root (VP via Richardson)", u_star_richardson(sched) is not None)

    print("== 5. HEADLINE: KL order ~2 generic, ~4 at lambda* ==")
    set_dps(90)
    sched2 = vp_const(B, s2, T)
    lstar = lambda_star_vp(B, s2, T)
    Ns = [256, 1024, 4096]

    def kl_at(lam, N):
        return kl_gauss(v_terminal(sched2, N, mp.mpf(lam) ** 2, integrator="EM"), s2)
    kl1 = [kl_at(1.0, N) for N in Ns]
    kls = [kl_at(lstar, N) for N in Ns]
    p1 = local_order(kl1[0], kl1[2], Ns[0], Ns[2])
    ps = local_order(kls[0], kls[2], Ns[0], Ns[2])
    check("order ~2 at lam=1", abs(p1 - 2.0) < 0.03, f"p={p1:.4f}")
    check("order ~4 at lam*", abs(ps - 4.0) < 0.05, f"p={ps:.4f}  (KL at N=4096: {mp.nstr(kls[2],3)})")

    return n[0]


def test_core_identities():
    """pytest entry point: every core identity must hold (raises AssertionError otherwise)."""
    _checks()


if __name__ == "__main__":
    try:
        total = _checks()
        print(f"\n=== {total} passed, 0 failed ===")
        sys.exit(0)
    except AssertionError as exc:
        print(f"\n=== FAILED: {exc} ===")
        sys.exit(1)
