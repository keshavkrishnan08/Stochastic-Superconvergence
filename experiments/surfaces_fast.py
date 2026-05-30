"""Fast float64 surface grids for the gallery (figures need ~3 sig figs, not 60 digits).
C(lam;s2)=T*quad(Phi*sigma), recursion in float64. Canyon (E17) already saved in mpmath; skip."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ssc")))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import io_utils as io

def Vt(t,B,s2): return 1.0+(s2-1.0)*np.exp(-B*t)
def C_f(B,lam,s2,T):
    u=lam*lam
    def integ(t):
        V=Vt(t,B,s2); Phi=s2**(1+u)*np.exp(-u*B*t)/V**(1+u); sig=(B*B/4)*(-V+(1+u)**2/V-2*u)
        return Phi*sig
    val,_=quad(integ,0,T,limit=80); return T*val
def lam_star_f(B,s2,T):
    if s2<=1.0 or C_f(B,0.0,s2,T)>=0: return None
    hi=0.3
    while C_f(B,hi,s2,T)<0 and hi<50: hi*=1.6
    if C_f(B,hi,s2,T)<0: return None
    try: return brentq(lambda l:C_f(B,l,s2,T),0.0,hi,xtol=1e-6)
    except Exception: return None
def vN_f(N,T,B,lam,s2,eps2=0.0):
    dt=T/N; v=Vt(T,B,s2); u=lam*lam
    for k in range(int(N)):
        tk=T-k*dt; V=Vt(tk,B,s2); A=-B/2+(1+u)/2*B/V; v=(1-A*dt)**2*v+(u*B+eps2)*dt
    return v
def kl_f(v,s2): r=v/s2; return 0.5*(r-1-np.log(r))

def e18(B=4.0,T=5.0,nlam=70,ns2=70):
    lams=np.linspace(0,2.6,nlam); s2s=np.linspace(1.05,8.0,ns2)
    C=[[C_f(B,l,float(s),T) for l in lams] for s in s2s]
    lc=[lam_star_f(B,float(s),T) for s in s2s]
    io.save("e18_coeff_field",{"config":{"B":B,"T":T},"lams":lams.tolist(),"s2s":s2s.tolist(),"C":C,"lambda_star_curve":lc})
    io.log("e18_coeff_field fast done")
def e19(s2=2.0,B=4.0,T=5.0,nlam=90):
    lams=np.linspace(0.4,2.2,nlam); ls=lam_star_f(B,s2,T)
    Ns=[48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096]  # dense for a smooth 3D surface
    pairs=[(Ns[i],Ns[i+1]) for i in range(len(Ns)-1)]
    P=[[float(np.log(max(kl_f(vN_f(n1,T,B,l,s2),s2),1e-300)/max(kl_f(vN_f(n2,T,B,l,s2),s2),1e-300))/np.log(n2/n1)) for (n1,n2) in pairs] for l in lams]
    Nmid=[float((a*b)**0.5) for (a,b) in pairs]
    io.save("e19_order_field",{"config":{"s2":s2,"B":B,"T":T},"lambda_star":ls,"lams":lams.tolist(),
            "Npairs":[list(p) for p in pairs],"Nmid":Nmid,"order":P})
    io.log("e19_order_field fast done")
def e20(s2=2.0,B=4.0,T=5.0,N=512,nlam=70,neps=40):
    lams=np.linspace(0,2.0,nlam); epss=np.linspace(0,0.3,neps); ls=lam_star_f(B,s2,T)
    Z=[]; valley=[]
    for e in epss:
        row=[float(np.log10(max(kl_f(vN_f(N,T,B,l,s2,e*e),s2),1e-30))) for l in lams]
        Z.append(row); valley.append(float(lams[int(np.argmin(row))]))
    io.save("e20_goldilocks_surf",{"config":{"s2":s2,"B":B,"T":T,"N":N},"lambda_star":ls,"lams":lams.tolist(),"epss":epss.tolist(),"logKL":Z,"valley_lambda":valley})
    io.log("e20_goldilocks_surf fast done")
def e21(B=4.0,T=5.0,ns2=90,nlam=90):
    s2s=np.linspace(0.4,8.0,ns2); lams=np.linspace(0,3.0,nlam)
    S=[[float(np.sign(C_f(B,l,float(s),T))) for l in lams] for s in s2s]
    lc=[lam_star_f(B,float(s),T) for s in s2s]
    io.save("e21_phase_diagram",{"config":{"B":B,"T":T},"s2s":s2s.tolist(),"lams":lams.tolist(),"signC":S,"lambda_star_curve":lc})
    io.log("e21_phase_diagram fast done")
def e22(B=4.0,T=5.0,nlam=120):
    spec=[1.3,2.0,3.2,5.0]; lams=np.linspace(0,3.0,nlam)
    Ci={f"{s}":[C_f(B,l,s,T) for l in lams] for s in spec}
    per=[lam_star_f(B,s,T) for s in spec]
    K=[float(sum(C_f(B,l,s,T)**2/s**4 for s in spec)) for l in lams]
    io.save("e22_aniso_spaghetti",{"config":{"B":B,"T":T},"spectrum":spec,"lams":lams.tolist(),"C_i":Ci,"per_mode_root":per,"K":K,"lambda_dagger":float(lams[int(np.argmin(K))])})
    io.log("e22_aniso_spaghetti fast done")

if __name__=="__main__":
    import time
    for fn in [e18,e19,e20,e21,e22]:
        t=time.time()
        try: fn(); io.log(f"[{fn.__name__}] {time.time()-t:.1f}s")
        except Exception as ex:
            import traceback; io.log(f"[{fn.__name__}] ERR {ex}\n{traceback.format_exc()}")
    io.log("surfaces_fast DONE")
