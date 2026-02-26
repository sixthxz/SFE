# -*- coding: utf-8 -*-
"""
SFE-05.7 — Structural Injection + Detection Boundary Mapping
=============================================================
Stochastic Field Engine, revision 05.7

CONTEXT:
    SFE-05.6 proved the detector does not hallucinate coherence in noise.
    core_write_rate = 0.470 (target range), zero coherence events, zero
    regime shifts — correct behavior for free diffusion. The analytical
    floor for Surprise/Volatility under free diffusion is ~1.79.

    SFE-05.7 introduces minimal field structure and measures the detection
    boundary: the minimum drift magnitude at which the calibrated detector
    produces its first honest coherence event.

WHAT THIS VERSION DOES:

    1. DRIFT SWEEP
       Add weak drift mu to the position process each Langevin step:
           dx = (F_flux + F_attract)/gamma * dt_eff + mu * dt_eff + noise
       mu swept over: [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
       Full SFE-05.6 architecture runs unchanged for each mu.
       Record per mu: Surprise/Volatility, n_coherence_events,
           first_event_cycle, core_write_rate, anomaly_rate, n_regime_shifts.

    2. DETECTION BOUNDARY
       mu_threshold = min(mu : coherence_events > 0)
       This is the anticipation threshold — minimum structure the architecture
       can detect without hallucinating.

    3. PERIODIC FORCING
       After drift sweep, one additional run with sinusoidal forcing:
           dx += A * sin(omega * cycle_count) * dt_eff
       A = 0.05, omega = 0.01
       Compare: does periodic force produce coherence at lower amplitude?
       Does Kalman converge faster?

ARCHITECTURE (unchanged from SFE-05.6):
    Dual buffer (core N=512, anomaly N=256)
    Adaptive gate threshold = std(surprise[-window:]) * multiplier
    gate_multiplier=1.0, floor=field_volatility
    Coherence condition: Surprise/Volatility < 1.0 AND hull_variance < 0.15
    AND core_write_rate >= 0.20
    Coherence centroid pull (alpha=0.15), rehearsal (k=5), speed modulation
    Adaptive lambda tension dynamics

PARAMETERS:
    mu_values     = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    A_periodic    = 0.05
    omega_periodic = 0.01
"""

import numpy as np
try:
    import torch; TORCH = True
except ImportError:
    TORCH = False
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import uniform_filter1d
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)
import os; FIG_DIR = '/tmp/sfe057_figs'; os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.7  —  Structural Injection + Detection Boundary Mapping")
print("=" * 70)
print(f"  Backend: {'PyTorch ' + torch.__version__ if TORCH else 'NumPy'}")
print()

# ─── Units ────────────────────────────────────────────────────────────────────
kBT = 1.0; gamma = 1.0; D_diff = kBT / gamma; Landauer = np.log(2)

# ─── Simulation ───────────────────────────────────────────────────────────────
dt = 0.01; N = 20000; tau_meas = 10; N_cycles = N // tau_meas

# ─── Field grid ───────────────────────────────────────────────────────────────
x_min, x_max = -8.0, 8.0; Nx = 400
x_grid = np.linspace(x_min, x_max, Nx); dx = x_grid[1] - x_grid[0]

# ─── Inherited parameters (unchanged from SFE-05.6) ──────────────────────────
lambda_coup_init = 0.30; lambda_min = 0.05; lambda_max = 0.80
sigma_m = 0.90; sigma_memory = 1.20; coherence_var_th = 0.15; window_scale = 1.0
alpha_attract = 0.15; min_events_before_pull = 10
n_candidates = 5; rehearsal_on = True
dt_coherent = dt; dt_surprised = dt * 0.5
BUFFER_N = 512; COHERENCE_WIN = 40
gate_multiplier = 1.0; gate_window = 50
anomaly_buffer_size = 256; regime_shift_multiplier = 2.0; min_core_rate = 0.20

# ─── SFE-05.7 sweep parameters ────────────────────────────────────────────────
mu_values      = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
A_periodic     = 0.05
omega_periodic = 0.01

# ─── Field volatility ─────────────────────────────────────────────────────────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

print(f"  mu sweep: {mu_values}")
print(f"  periodic: A={A_periodic}  omega={omega_periodic}")
print(f"  N={N}  tau_meas={tau_meas}  COHERENCE_WIN={COHERENCE_WIN}")
print(f"  gate_multiplier={gate_multiplier}  floor=field_volatility={field_volatility:.4f}")
print(f"  coherence condition: surp/vol < 1.0  AND  hull_var < {coherence_var_th}")
print(f"                       AND core_rate >= {min_core_rate}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# ALL CLASSES (identical to SFE-05.6)
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    def __init__(self, N, cols=3, name='buf'):
        self.N=N; self.cols=cols; self.name=name
        self.buf = torch.zeros(N,cols,dtype=torch.float32) if TORCH \
                   else np.zeros((N,cols),dtype=np.float32)
        self.filled=self.writes=0
    def push(self, row):
        self.writes+=1
        if TORCH: self.buf=torch.roll(self.buf,shifts=1,dims=0); self.buf[0]=torch.tensor(row,dtype=torch.float32)
        else: self.buf=np.roll(self.buf,shift=1,axis=0); self.buf[0]=np.array(row,dtype=np.float32)
        self.filled=min(self.filled+1,self.N)
    def get_numpy(self): return self.buf[:self.filled].numpy() if TORCH else self.buf[:self.filled]
    def normalize(self):
        data=self.get_numpy()
        if len(data)<4: return data,np.zeros(self.cols),np.ones(self.cols)
        if TORCH:
            T=self.buf[:self.filled]; mu=T.mean(dim=0); sig=T.std(dim=0)+1e-8
            return ((T-mu)/sig).numpy(),mu.numpy(),sig.numpy()
        mu=data.mean(axis=0); sig=data.std(axis=0)+1e-8
        return (data-mu)/sig,mu,sig

class AdaptiveGate:
    def __init__(self, multiplier, window, floor):
        self.mult=multiplier; self.window=window; self.floor=floor
        self._hist=[]; self.threshold=floor
    def update(self, surprise):
        self._hist.append(surprise)
        if len(self._hist)>self.window: self._hist.pop(0)
        if len(self._hist)>=4: self.threshold=max(float(np.std(self._hist))*self.mult,self.floor)
        return self.threshold
    def core_gate(self, s):    return s <= self.threshold
    def anomaly_gate(self, s): return s >  self.threshold

class RegimeShiftDetector:
    def __init__(self, multiplier, window):
        self.mult=multiplier; self.window=window; self._flags=[]; self._rates=[]; self.events=[]
    def update(self, anom_written, cycle, centroid_pos):
        self._flags.append(int(anom_written))
        if len(self._flags)>self.window: self._flags.pop(0)
        if len(self._flags)<10: return False
        rate=float(np.mean(self._flags)); self._rates.append(rate)
        if len(self._rates)<10: return False
        baseline=float(np.mean(self._rates))+1e-8
        if rate>baseline*self.mult:
            self.events.append({'cycle':cycle,'rate':rate,'baseline':baseline,
                                'centroid':centroid_pos.copy() if centroid_pos is not None else np.zeros(3)})
            return True
        return False
    @property
    def n(self): return len(self.events)
    def rolling_rate(self): return float(np.mean(self._flags)) if self._flags else 0.0

def gaussian_rho(mu, sigma):
    rho=np.exp(-0.5*((x_grid-mu)/sigma)**2); return rho/np.trapezoid(rho,x_grid)

def fp_flux(rho, F_arr):
    return (F_arr/gamma)*rho - D_diff*np.gradient(rho,x_grid)

def fp_step(rho, F_arr):
    N_=len(rho); v=F_arr/gamma; df=np.zeros(N_+1); ff=np.zeros(N_+1)
    for i in range(1,N_):
        vf=0.5*(v[i-1]+v[i]); df[i]=vf*rho[i-1] if vf>=0 else vf*rho[i]
        ff[i]=D_diff*(rho[i]-rho[i-1])/dx
    rho_new=np.maximum(rho-(dt/dx)*np.diff(df-ff),0.0)
    norm=np.trapezoid(rho_new,x_grid); return rho_new/norm if norm>1e-12 else rho_new

def compute_alignment(x_pos, rho_true, rho_perc):
    g_t=float(np.interp(x_pos,x_grid,np.gradient(rho_true,x_grid)))
    g_p=float(np.interp(x_pos,x_grid,np.gradient(rho_perc,x_grid)))
    if abs(g_t)<1e-10 or abs(g_p)<1e-10: return 0.0
    return float(np.clip(g_t*g_p/(abs(g_t)*abs(g_p)),-1.0,1.0))

class KalmanOD:
    def __init__(self): self.x_hat=0.0; self.P=2*kBT/gamma*tau_meas*dt; self.Q=2*kBT/gamma*dt
    def predict_n(self,n): self.P+=n*self.Q
    def update(self,z):
        P_prior=self.P; K=self.P/(self.P+sigma_m**2); innov=z-self.x_hat
        self.x_hat+=K*innov; self.P*=(1-K); return P_prior,abs(innov)
    def I_gain(self,P): return max(0.5*np.log2(1+P/sigma_m**2),0.0)
    def reset(self): self.x_hat=0.0; self.P=2*kBT/gamma*tau_meas*dt

class PerceivedField:
    def __init__(self,max_n=300): self.samples=[]; self.weights=[]; self.max_n=max_n
    def add(self,xp,w=1.0):
        self.samples.append(xp); self.weights.append(w)
        if len(self.samples)>self.max_n: self.samples.pop(0); self.weights.pop(0)
    def get_rho(self):
        if len(self.samples)<2: return gaussian_rho(0.0,1.0)
        rho=np.zeros(Nx); w_tot=sum(self.weights)
        for xp,w in zip(self.samples,self.weights):
            rho+=(w/w_tot)*np.exp(-0.5*((x_grid-xp)/sigma_memory)**2)
        norm=np.trapezoid(rho,x_grid); return rho/norm if norm>1e-12 else rho

def convex_hull_3d(pts):
    if len(pts)<5: return 0.0,0.0,None
    try: h=ConvexHull(pts); return h.volume,h.area,h
    except QhullError: return 0.0,0.0,None

def pca_axes(pts):
    if len(pts)<4: return np.array([1.,0.,0.]),np.eye(3)
    c=pts-pts.mean(axis=0); cov=np.cov(c.T)
    if cov.ndim<2: return np.array([1.,0.,0.]),np.eye(3)
    vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]
    vals=np.maximum(vals[order],0.); vecs=vecs[:,order]
    return vals/(vals.sum()+1e-12),vecs.T

class CoherenceCentroid:
    def __init__(self): self.centroid=np.zeros(3); self.n=0
    def update(self,pos):
        self.n+=1; self.centroid=((self.n-1)*self.centroid+pos)/self.n
        return self.centroid.copy()
    def attraction_force_x(self,campo):
        return alpha_attract*(self.centroid[0]-campo) if self.n>0 else 0.0
    @property
    def is_ready(self): return self.n>=min_events_before_pull
    @property
    def position(self): return self.centroid.copy()

class CoherenceRecorder:
    def __init__(self,var_th,field_vol,window,centroid):
        self.var_th=var_th; self.fvol=field_vol; self.window=max(window,4)
        self.centroid=centroid; self.events=[]; self._vols=[]
    def check(self,vol,mean_surp,pts,mean_state,lam,cycle,core_rate):
        self._vols.append(vol)
        if len(self._vols)>self.window: self._vols.pop(0)
        if len(self._vols)<4: return False,0.0,0.0
        mean_v=np.mean(self._vols)+1e-8; rel_var=float(np.var(self._vols))/(mean_v**2)
        surp_rat=mean_surp/(self.fvol+1e-10)
        is_event=(rel_var<self.var_th)and(surp_rat<1.0)and(core_rate>=min_core_rate)
        if is_event and len(pts)>=5:
            vol_e,area_e,hull_e=convex_hull_3d(pts); ratio_e,axes_e=pca_axes(pts)
            self.centroid.update(mean_state)
            self.events.append({'cycle':cycle,'vol':vol_e,'area':area_e,'hull':hull_e,
                                'pts':pts.copy(),'pca_ratio':ratio_e,'pca_axes':axes_e,
                                'rel_var':rel_var,'surp_ratio':surp_rat,'lambda':lam,
                                'centroid':self.centroid.position,'core_rate':core_rate})
        return is_event,rel_var,surp_rat
    @property
    def n(self): return len(self.events)
    def mean_vol(self): return float(np.mean([e['vol'] for e in self.events])) if self.events else 0.0
    def mean_pca(self):
        if not self.events: return np.ones(3)/3,np.eye(3)
        return (np.array([e['pca_ratio'] for e in self.events]).mean(axis=0),
                np.array([e['pca_axes']  for e in self.events]).mean(axis=0))

def base_langevin(x, rho_true, F_free, F_extra, rng, lam, dt_eff):
    J_arr=fp_flux(rho_true,F_free); J_at=float(np.interp(x,x_grid,J_arr))
    F_flux=lam*J_at/(abs(J_at)+1e-10)
    xi=np.sqrt(2*kBT*gamma)*rng.standard_normal()
    dx_=((F_flux+F_extra)/gamma)*dt_eff+xi*np.sqrt(dt_eff)/gamma
    return float(np.clip(x+dx_,x_min+0.1,x_max-0.1)),abs(J_at)

def active_inference_step(x,rho_true,F_free,rng,lam,x_hat,centroid_obj,campo,sr):
    dt_eff=dt_coherent if sr<1.0 else dt_surprised
    F_att=centroid_obj.attraction_force_x(campo) if centroid_obj.is_ready else 0.0
    x_s,c_s=base_langevin(x,rho_true,F_free,F_att,rng,lam,dt_eff); s_s=abs(x_s-x_hat)
    if not rehearsal_on or n_candidates<=1: return x_s,c_s,False
    bx=x_s; bs=s_s; bc=c_s; acc=False
    for _ in range(n_candidates-1):
        xc,cc=base_langevin(x,rho_true,F_free,F_att,rng,lam,dt_eff); sc=abs(xc-x_hat)
        if sc<bs: bx=xc; bs=sc; bc=cc; acc=True
    return bx,bc,acc


# ═════════════════════════════════════════════════════════════════════════════
# CORE RUN FUNCTION — parameterized by mu (drift) and optional periodic forcing
# ═════════════════════════════════════════════════════════════════════════════

def run_one(seed=42, mu=0.0, A_sin=0.0, omega_sin=0.0, verbose=False):
    """
    Run the full SFE-05.6 architecture with added structural forcing.
    mu:      constant drift added per Langevin step (scaled by dt_eff)
    A_sin:   amplitude of sinusoidal forcing
    omega_sin: angular frequency of sinusoidal forcing (per cycle)

    The forcing term is: F_struct = mu + A_sin * sin(omega_sin * cycle)
    Applied inside base_langevin as F_extra.
    """
    rng=np.random.default_rng(seed)
    kf=KalmanOD(); pf=PerceivedField()
    core_buf=CircularBuffer(BUFFER_N,cols=3,name='core')
    anomaly_buf=CircularBuffer(anomaly_buffer_size,cols=3,name='anomaly')
    gate=AdaptiveGate(gate_multiplier,gate_window,floor=field_volatility)
    centroid=CoherenceCentroid()
    window_len=max(int(N_cycles*0.5*window_scale),8)
    recorder=CoherenceRecorder(coherence_var_th,field_volatility,window_len,centroid)
    regime=RegimeShiftDetector(regime_shift_multiplier,max(gate_window,30))

    rho_true=gaussian_rho(0.0,2.0); F_free=np.zeros(Nx)
    x=0.0; pf.add(x); lam=lambda_coup_init

    sr_log=[]; gate_log=[]; cr_log=[]; ar_log=[]; vol_log=[]
    rv_log=[]; surp_log=[]; campo_log=[]; lam_log=[]
    event_cycles=[]; centroid_hist=[]; regime_cycles=[]; rehearsal_log=[]
    snap_at={N//8:'early',N//2:'mid',7*N//8:'late'}; snap_va={}
    vol_win=[]; surp_win=[]; total=0; cw=0; aw=0; cycle=0
    n_reset=max(N_cycles//8,1); cur_sr=2.0

    for i in range(N):
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas=x+sigma_m*rng.standard_normal()
            P_prior,surp=kf.update(x_meas)
            rho_perc=pf.get_rho()
            alignment=compute_alignment(x,rho_true,rho_perc)
            J_arr=fp_flux(rho_true,F_free)
            campo=float(abs(np.interp(x,x_grid,J_arr)))
            cur_sr=surp/(field_volatility+1e-10)

            g_th=gate.update(surp); gate_log.append(g_th); total+=1
            wc=gate.core_gate(surp); wa=gate.anomaly_gate(surp)
            if wc: core_buf.push([campo,alignment,surp]); cw+=1
            anom_written=False
            if wa: anomaly_buf.push([campo,alignment,surp]); aw+=1; anom_written=True

            sr_log.append(cur_sr); surp_log.append(surp); campo_log.append(campo)
            lam_log.append(lam)
            surp_win.append(surp)
            if len(surp_win)>COHERENCE_WIN: surp_win.pop(0)

            regime.update(anom_written,cycle,centroid.position if centroid.n>0 else None)
            if regime.n>len(regime_cycles): regime_cycles.append(cycle)

            cycle+=1

            if cycle%COHERENCE_WIN==0 and core_buf.filled>=5:
                T_hat,_,_=core_buf.normalize()
                vol,area,_=convex_hull_3d(T_hat)
                vol_log.append(vol); vol_win.append(vol)
                if len(vol_win)>window_len: vol_win.pop(0)
                core_rate=cw/max(total,1); anom_rate=aw/max(total,1)
                cr_log.append(core_rate); ar_log.append(anom_rate)
                if len(vol_win)>=3:
                    vm=np.mean(vol_win); vs=np.std(vol_win)+1e-8; z=(vol-vm)/vs
                    if z>1.0:    lam=min(lam*1.05,lambda_max)
                    elif z<-0.5: lam=max(lam*0.98,lambda_min)
                ms=float(np.mean(surp_win)); ms_raw=core_buf.get_numpy().mean(axis=0)
                is_ev,rel_var,surp_rat=recorder.check(vol,ms,T_hat,ms_raw,lam,cycle,core_rate)
                rv_log.append(rel_var)
                if is_ev:
                    event_cycles.append(cycle); centroid_hist.append(centroid.position.copy())
                    if verbose:
                        print(f"      ★ cycle={cycle:4d}  surp/vol={surp_rat:.3f}  "
                              f"core={core_rate:.3f}  cen={centroid.position.round(3)}")

        if (i+1)%(n_reset*tau_meas)==0 and i>0: kf.reset()

        # ── Structural forcing ────────────────────────────────────────────
        dt_eff=dt_coherent if cur_sr<1.0 else dt_surprised
        F_struct=(mu + A_sin*np.sin(omega_sin*cycle))*gamma   # convert to force units

        F_att=centroid.attraction_force_x(campo_log[-1] if campo_log else 0.0) \
              if centroid.is_ready else 0.0
        F_extra=F_att+F_struct

        # Counterfactual selection
        x_s,c_s=base_langevin(x,rho_true,F_free,F_extra,rng,lam,dt_eff)
        s_s=abs(x_s-kf.x_hat); bx=x_s; bs=s_s; bc=c_s; acc=False
        if rehearsal_on and n_candidates>1:
            for _ in range(n_candidates-1):
                xc,cc=base_langevin(x,rho_true,F_free,F_extra,rng,lam,dt_eff)
                sc=abs(xc-kf.x_hat)
                if sc<bs: bx=xc; bs=sc; bc=cc; acc=True
        rehearsal_log.append(int(acc))

        rho_true=fp_step(rho_true,F_free); pf.add(bx,w=1./(sigma_m+0.1)); x=bx

        if i in snap_at:
            T_hat,_,_=core_buf.normalize(); lbl=snap_at[i]
            vol,area,hull=convex_hull_3d(T_hat); snap_va[lbl]=(vol,area,hull,T_hat.copy())

    T_hat_f,_,_=core_buf.normalize(); vol_f,area_f,hull_f=convex_hull_3d(T_hat_f)
    snap_va['final']=(vol_f,area_f,hull_f,T_hat_f.copy())

    mean_sr=float(np.mean(sr_log)); crf=cw/max(total,1); arf=aw/max(total,1)
    first_event_cycle=event_cycles[0] if event_cycles else None

    return dict(
        mean_sr=mean_sr, n_events=recorder.n, first_event_cycle=first_event_cycle,
        core_rate=crf, anom_rate=arf, n_regime=regime.n, regime_cycles=regime_cycles,
        event_cycles=event_cycles, centroid_final=centroid.position,
        centroid_n=centroid.n, centroid_hist=centroid_hist,
        events=recorder.events, mean_vol_events=recorder.mean_vol(),
        mean_pca=recorder.mean_pca(), lam_final=lam,
        rehearsal_rate=float(np.mean(rehearsal_log)) if rehearsal_log else 0.0,
        snap_va=snap_va, buf_final=T_hat_f,
        vol_log=np.array(vol_log), rv_log=np.array(rv_log),
        sr_log=np.array(sr_log), gate_log=np.array(gate_log),
        cr_log=np.array(cr_log), ar_log=np.array(ar_log),
    )


# ═════════════════════════════════════════════════════════════════════════════
# DRIFT SWEEP
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("DRIFT SWEEP")
print("─" * 70)

sweep_results = []
for mu in mu_values:
    print(f"  mu={mu:.3f}  ", end='', flush=True)
    r = run_one(seed=42, mu=mu, A_sin=0.0, omega_sin=0.0, verbose=(mu > 0.01))
    sweep_results.append({'mu': mu, **r})
    status = '★ COH' if r['n_events'] > 0 else '—'
    print(f"surp/vol={r['mean_sr']:.4f}  events={r['n_events']}  "
          f"core={r['core_rate']:.3f}  anom={r['anom_rate']:.3f}  "
          f"regime={r['n_regime']}  {status}")

# Detection boundary
mu_threshold    = None
threshold_result = None
for sr in sweep_results:
    if sr['n_events'] > 0:
        mu_threshold = sr['mu']
        threshold_result = sr
        break

print()
if mu_threshold is not None:
    tr = threshold_result
    print(f"  ★ Detection boundary: mu_threshold = {mu_threshold}")
    print(f"    mean surp/vol at threshold = {tr['mean_sr']:.4f}")
    print(f"    first event at cycle       = {tr['first_event_cycle']}")
    print(f"    core_write_rate            = {tr['core_rate']:.3f}")
else:
    print(f"  No coherence events at any mu. Threshold not found in this sweep range.")


# ═════════════════════════════════════════════════════════════════════════════
# PERIODIC FORCING TEST
# ═════════════════════════════════════════════════════════════════════════════
print()
print("─" * 70)
print(f"PERIODIC FORCING  A={A_periodic}  omega={omega_periodic}")
print("─" * 70)
r_per = run_one(seed=42, mu=0.0, A_sin=A_periodic, omega_sin=omega_periodic, verbose=True)
print(f"  surp/vol={r_per['mean_sr']:.4f}  events={r_per['n_events']}  "
      f"core={r_per['core_rate']:.3f}  anom={r_per['anom_rate']:.3f}  "
      f"regime={r_per['n_regime']}")


# ═════════════════════════════════════════════════════════════════════════════
# SWEEP VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering sweep figure...", end='', flush=True)

BG='#07080f'; FG='#dde1ec'; GOLD='#f5c842'; TEAL='#3dd6c8'; VIOLET='#b87aff'
ROSE='#ff5f7e'; GREEN='#4ade80'; AMBER='#fb923c'; COH='#fde68a'
REGIME='#f97316'; DIM='#1e2235'; WH='#ffffff'

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':DIM,
    'text.color':FG,'axes.labelcolor':FG,'xtick.color':'#555870','ytick.color':'#555870'})

mus       = np.array([s['mu']        for s in sweep_results])
mean_srs  = np.array([s['mean_sr']   for s in sweep_results])
n_evs     = np.array([s['n_events']  for s in sweep_results])
fe_cycles = [s['first_event_cycle']  for s in sweep_results]
core_rs   = np.array([s['core_rate'] for s in sweep_results])
anom_rs   = np.array([s['anom_rate'] for s in sweep_results])
n_regimes = np.array([s['n_regime']  for s in sweep_results])

mu_plot = np.where(mus == 0, 5e-5, mus)   # for log scale; 0 → tiny offset

fig, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor=BG)
fig.suptitle("SFE-05.7  ·  Structural Injection — Detection Boundary Mapping\n"
             "Drift sweep: mu ∈ [0, 0.20]  |  Periodic test: A=0.05  ω=0.01",
             fontsize=13, color=GOLD, y=1.002, fontweight='bold')

for ax in axes.flat:
    ax.set_facecolor(BG); ax.grid(True, alpha=0.12); ax.tick_params(labelsize=8)

def vline_mu(ax, color=COH, ls='--'):
    if mu_threshold is not None:
        mup = mu_threshold if mu_threshold > 0 else 5e-5
        ax.axvline(mup, color=color, lw=1.5, ls=ls, alpha=0.80, label=f'μ_threshold={mu_threshold}')

# Panel 0: Surprise/Volatility vs mu
ax0 = axes[0, 0]
ax0.semilogx(mu_plot, mean_srs, 'o-', color=ROSE, lw=2, ms=8, markeredgewidth=0)
ax0.axhline(1.0, color=GREEN, lw=1.3, ls='--', label='Target = 1.0')
ax0.axhline(mean_srs[0], color=TEAL, lw=0.8, ls=':', alpha=0.6,
            label=f'Free diffusion = {mean_srs[0]:.3f}')
vline_mu(ax0)
# Mark periodic result
ax0.axhline(r_per['mean_sr'], color=VIOLET, lw=1.2, ls=':',
            label=f'Periodic (A={A_periodic}) = {r_per["mean_sr"]:.3f}')
ax0.set_xlabel("μ (drift, log scale)", fontsize=9); ax0.set_ylabel("Mean Surprise/Volatility", fontsize=9)
ax0.set_title("Surprise / Volatility vs Drift", color=FG, fontsize=10)
ax0.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 1: Coherence events vs mu
ax1 = axes[0, 1]
ax1.semilogx(mu_plot, n_evs, 's-', color=COH, lw=2, ms=8, markeredgewidth=0)
ax1.scatter([5e-5], [r_per['n_events']], color=VIOLET, s=100, marker='D', zorder=5,
            label=f'Periodic = {r_per["n_events"]} events')
vline_mu(ax1)
ax1.set_xlabel("μ (drift, log scale)", fontsize=9); ax1.set_ylabel("Coherence Events", fontsize=9)
ax1.set_title("Coherence Events vs Drift", color=FG, fontsize=10)
ax1.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 2: First event cycle vs mu
ax2 = axes[0, 2]
fe_vals = [c if c is not None else np.nan for c in fe_cycles]
valid_mu = [mu_plot[i] for i, v in enumerate(fe_vals) if not np.isnan(v)]
valid_fe = [v for v in fe_vals if not np.isnan(v)]
if valid_mu:
    ax2.semilogx(valid_mu, valid_fe, '^-', color=TEAL, lw=2, ms=9, markeredgewidth=0,
                 label='First event cycle')
if r_per['first_event_cycle'] is not None:
    ax2.axhline(r_per['first_event_cycle'], color=VIOLET, lw=1.2, ls=':',
                label=f'Periodic first event = {r_per["first_event_cycle"]}')
vline_mu(ax2)
ax2.set_xlabel("μ (drift, log scale)", fontsize=9); ax2.set_ylabel("First Coherence Event (cycle)", fontsize=9)
ax2.set_title("Detection Speed vs Drift", color=FG, fontsize=10)
ax2.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 3: core_write_rate vs mu (should stay in 0.3–0.7)
ax3 = axes[1, 0]
ax3.semilogx(mu_plot, core_rs, 'o-', color=GREEN, lw=2, ms=8, label='Core rate (drift)')
ax3.semilogx(mu_plot, anom_rs, 'o--', color=REGIME, lw=2, ms=8, label='Anomaly rate (drift)', alpha=0.8)
ax3.axhline(r_per['core_rate'], color=GREEN, lw=1.2, ls=':', alpha=0.6,
            label=f'Periodic core = {r_per["core_rate"]:.3f}')
ax3.axhspan(0.30, 0.70, alpha=0.07, color=GREEN)
vline_mu(ax3)
ax3.set_xlabel("μ (drift, log scale)", fontsize=9); ax3.set_ylabel("Write Rate", fontsize=9)
ax3.set_title("Dual Buffer Write Rates vs Drift\nGreen band = functional range", color=FG, fontsize=10)
ax3.set_ylim(0, 1.1); ax3.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 4: Regime shifts vs mu
ax4 = axes[1, 1]
ax4.semilogx(mu_plot, n_regimes, 'D-', color=REGIME, lw=2, ms=8, label='Regime shifts (drift)')
ax4.axhline(r_per['n_regime'], color=VIOLET, lw=1.2, ls=':', label=f'Periodic = {r_per["n_regime"]}')
vline_mu(ax4)
ax4.set_xlabel("μ (drift, log scale)", fontsize=9); ax4.set_ylabel("Regime Shift Events", fontsize=9)
ax4.set_title("Regime Shifts vs Drift\n(detector is observational only)", color=FG, fontsize=10)
ax4.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 5: Summary table
ax5 = axes[1, 2]; ax5.axis('off')
threshold_str = f"μ_threshold = {mu_threshold}" if mu_threshold is not None else "Not found"
lines = [
    "SFE-05.7  DETECTION BOUNDARY",
    "─" * 30,
    "",
    f"Sweep: mu ∈ {mu_values}",
    "",
    f"BASELINE (mu=0.0)",
    f"  surp/vol   = {mean_srs[0]:.4f}",
    f"  core_rate  = {core_rs[0]:.3f}",
    f"  events     = {n_evs[0]}",
    "",
    f"DETECTION BOUNDARY",
    f"  {threshold_str}",
]
if mu_threshold is not None:
    tr = threshold_result
    lines += [
        f"  surp/vol   = {tr['mean_sr']:.4f}",
        f"  first evt  = cycle {tr['first_event_cycle']}",
        f"  core_rate  = {tr['core_rate']:.3f}",
    ]
lines += [
    "",
    f"PERIODIC FORCING",
    f"  A={A_periodic}  ω={omega_periodic}",
    f"  surp/vol   = {r_per['mean_sr']:.4f}",
    f"  events     = {r_per['n_events']}",
    f"  core_rate  = {r_per['core_rate']:.3f}",
]
if mu_threshold is not None and r_per['n_events'] > 0:
    lines += ["", "  Periodic detects at lower"]
    lines += [f"  amplitude than drift: TRUE"]
elif r_per['n_events'] > 0:
    lines += ["", "  Periodic detected even"]
    lines += [f"  though drift did not"]
else:
    lines += ["", "  Periodic: no detection"]

ax5.text(0.04, 0.97, "\n".join(lines),
         transform=ax5.transAxes, fontsize=7.5, fontfamily='monospace', color=FG, va='top',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0c14', edgecolor=GOLD, lw=1.2))

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(FIG_DIR,'sfe057_sweep.png'), dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# MANIFOLD WORLD — at mu_threshold (or highest mu if none found)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering manifold world...", end='', flush=True)

target_result = threshold_result if threshold_result is not None else sweep_results[-1]
target_mu     = mu_threshold     if mu_threshold     is not None else mu_values[-1]
target_label  = f"μ={target_mu}"

def draw_hull(ax, pts, hull, fc, ec, af=0.10, ae=0.28):
    if hull is None or pts is None: return
    step=max(len(hull.simplices)//300,1)
    poly=Poly3DCollection([pts[s] for s in hull.simplices[::step]],alpha=af,linewidth=0.3)
    poly.set_facecolor(fc); poly.set_edgecolor(ec); ax.add_collection3d(poly)

def style_3d(ax,xl,yl,zl,title):
    ax.set_xlabel(xl,fontsize=8,labelpad=2); ax.set_ylabel(yl,fontsize=8,labelpad=2)
    ax.set_zlabel(zl,fontsize=8,labelpad=2); ax.set_title(title,color=FG,fontsize=9,pad=4)
    ax.tick_params(labelsize=6)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False; p.set_edgecolor(DIM)
    ax.grid(True,alpha=0.10)

fig2 = plt.figure(figsize=(24, 18), facecolor=BG)
fig2.suptitle(
    f"SFE-05.7  ·  Detection Boundary  ·  {target_label}\n"
    f"First honest coherence geometry  |  mu_threshold={'FOUND: '+str(mu_threshold) if mu_threshold else 'NOT FOUND'}",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold')
gs = GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.34, top=0.966, bottom=0.06, left=0.04, right=0.97)

r_t = target_result
snaps = r_t['snap_va']

# ── Panel 0: core buffer point cloud ─────────────────────────────────────────
ax0 = fig2.add_subplot(gs[0,:2], projection='3d'); ax0.set_facecolor(BG)
pts_f=r_t['buf_final']; _,_,hull_f=convex_hull_3d(pts_f)
sc=ax0.scatter(pts_f[:,0],pts_f[:,1],pts_f[:,2],c=np.linspace(0,1,len(pts_f)),
               cmap='plasma',s=7,alpha=0.70,linewidths=0)
draw_hull(ax0,pts_f,hull_f,TEAL,TEAL,0.07,0.18)
for ev in r_t['events'][:2]: draw_hull(ax0,ev['pts'],ev['hull'],COH,COH,0.20,0.55)
cen=r_t['centroid_final']
if r_t['centroid_n']>=min_events_before_pull:
    ax0.scatter(*cen,color=WH,s=200,marker='*',zorder=15)
    cm=pts_f.mean(axis=0); ax0.quiver(*cm,*(cen-cm),color=GOLD,lw=2.5,alpha=0.85,arrow_length_ratio=0.20)
try:
    Xp,Yp=np.meshgrid([-2.5,2.5],[-2.5,2.5])
    ax0.plot_surface(Xp,Yp,np.zeros_like(Xp),alpha=0.05,color=GOLD,linewidth=0)
    ax0.text(2.2,2.2,0.08,"S-U",color=GOLD,fontsize=7,alpha=0.5)
except: pass
plt.colorbar(sc,ax=ax0,shrink=0.45,pad=0.02).set_label('Time',color=FG,fontsize=7)
style_3d(ax0,"Campo  |J(x)|","Alignment","Surprise  |innov|",
         f"Core Buffer at {target_label}  [★=centroid]\n"
         f"events={r_t['n_events']}  core_rate={r_t['core_rate']:.3f}  "
         f"surp/vol={r_t['mean_sr']:.3f}  λ={r_t['lam_final']:.3f}")
ax0.view_init(elev=22,azim=-52)

# ── Panel 1: Hull evolution ───────────────────────────────────────────────────
ax1=fig2.add_subplot(gs[0,2],projection='3d'); ax1.set_facecolor(BG)
for lbl,col,af,ae in [('early',ROSE,0.12,0.35),('final',TEAL,0.07,0.20)]:
    e=snaps.get(lbl)
    if e:
        ax1.scatter(e[3][:,0],e[3][:,1],e[3][:,2],c=col,s=4,alpha=0.40,linewidths=0)
        draw_hull(ax1,e[3],e[2],col,col,af,ae)
if r_t['n_events']>0 and r_t['events'][0]['hull'] is not None:
    ev0=r_t['events'][0]; draw_hull(ax1,ev0['pts'],ev0['hull'],COH,COH,0.26,0.65)
if len(r_t['centroid_hist'])>=2:
    ch=np.array(r_t['centroid_hist']); ch_n=(ch-ch.mean(axis=0))/(ch.std(axis=0)+1e-8)
    ax1.plot(ch_n[:,0],ch_n[:,1],ch_n[:,2],color=WH,lw=1.5,alpha=0.65)
    ax1.scatter(ch_n[-1,0],ch_n[-1,1],ch_n[-1,2],color=WH,s=80,marker='*',zorder=12)
handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ROSE,ms=7,label=f"early V={snaps.get('early',(0,))[0]:.3f}"),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=TEAL,ms=7,label=f"final V={snaps.get('final',(0,))[0]:.3f}"),
    Line2D([0],[0],color=WH,lw=1.5,label='Centroid drift'),
]
if r_t['n_events']>0: handles.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COH,ms=10,label=f"coh. V={r_t['mean_vol_events']:.3f}"))
ax1.legend(handles=handles,fontsize=6.5,facecolor='#0d0f18',edgecolor='none')
style_3d(ax1,"Campo","Alignment","Surprise",f"Hull Evolution at {target_label}")
ax1.view_init(elev=25,azim=-40)

# ── Panel 2: Volume log ───────────────────────────────────────────────────────
ax2=fig2.add_subplot(gs[1,0]); ax2.set_facecolor(BG)
va=r_t['vol_log']; rv=r_t['rv_log']
if len(va)>0:
    tv=np.arange(len(va))*COHERENCE_WIN; tr2=np.arange(len(rv))*COHERENCE_WIN
    ax2.fill_between(tv,0,va,alpha=0.18,color=VIOLET); ax2.plot(tv,va,color=VIOLET,lw=1.8,label='Hull volume')
    ax2b=ax2.twinx(); ax2b.set_facecolor(BG)
    if len(rv)>0:
        ax2b.plot(tr2,rv,color=GOLD,lw=1.5,ls='--',label='Rel. variance')
        ax2b.axhline(coherence_var_th,color=GREEN,lw=1.2,ls=':',label=f'threshold={coherence_var_th}')
        ax2b.fill_between(tr2,0,rv,where=(rv<coherence_var_th),alpha=0.22,color=GREEN)
    ax2b.set_ylabel("Var(V)/mean(V)²",color=GOLD,fontsize=8); ax2b.tick_params(axis='y',labelcolor=GOLD,labelsize=7)
    for ec in r_t['event_cycles']: ax2.axvline(ec,color=COH,lw=0.8,alpha=0.5,ls='--')
    ax2.set_xlabel("Cycle",fontsize=9); ax2.set_ylabel("Hull Volume",color=VIOLET,fontsize=9)
    ax2.set_title(f"Volume + Variance at {target_label}",color=FG,fontsize=9)
    ax2.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
    ax2b.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper left')
    ax2.grid(True,alpha=0.12); ax2.tick_params(labelsize=7)

# ── Panel 3: Surprise ratio vs cycle ─────────────────────────────────────────
ax3=fig2.add_subplot(gs[1,1]); ax3.set_facecolor(BG)
sr=r_t['sr_log']; cyc=np.arange(len(sr))
if len(sr)>0:
    win=max(len(sr)//40,3); sr_s=uniform_filter1d(sr,size=win)
    ax3.fill_between(cyc,0,sr,alpha=0.09,color=ROSE)
    ax3.plot(cyc,sr_s,color=ROSE,lw=2.0,label=f'Surprise/Vol ({target_label})')
    # Overlay baseline (mu=0)
    sr0=sweep_results[0]['sr_log']; sr0_s=uniform_filter1d(sr0,size=win)
    ax3.plot(cyc[:len(sr0_s)],sr0_s,color=TEAL,lw=1.2,ls='--',alpha=0.7,label='Baseline (μ=0)')
    ax3.axhline(1.0,color=GREEN,lw=1.3,ls='--',label='Target = 1.0')
    for ec in r_t['event_cycles']: ax3.axvline(ec,color=COH,lw=0.8,alpha=0.5,ls='--')
    ax3.set_xlabel("Cycle",fontsize=9); ax3.set_ylabel("Surprise / Field Volatility",fontsize=9)
    ax3.set_title(f"Surprise Ratio: Baseline vs {target_label}",color=FG,fontsize=9)
    ax3.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none')
    ax3.grid(True,alpha=0.12); ax3.tick_params(labelsize=7); ax3.set_xlim(0,len(sr))

# ── Panel 4: State space projection ──────────────────────────────────────────
ax4=fig2.add_subplot(gs[1,2]); ax4.set_facecolor(BG)
pts2=r_t['buf_final']
if len(pts2)>0:
    sc4=ax4.scatter(pts2[:,0],pts2[:,1],c=pts2[:,2],cmap='inferno',s=6,alpha=0.75,linewidths=0)
    cb4=plt.colorbar(sc4,ax=ax4,shrink=0.82); cb4.set_label('Surprise (z-score)',color=FG,fontsize=7); cb4.ax.yaxis.set_tick_params(labelsize=6)
    ax4.axhline(0,color=GOLD,lw=1.0,ls='--',alpha=0.55,label='S-U plane')
    ax4.fill_between([-3.5,3.5],-3.5,0,alpha=0.04,color=ROSE); ax4.fill_between([-3.5,3.5],0,3.5,alpha=0.04,color=GREEN)
    if r_t['centroid_n']>=min_events_before_pull:
        ax4.scatter([0],[0],color=WH,s=220,marker='*',zorder=15,label='Centroid')
        mx=pts2[:,0].mean(); my=pts2[:,1].mean()
        ax4.annotate('',xy=(0,0),xytext=(mx,my),arrowprops=dict(arrowstyle='->',color=GOLD,lw=2.0))
    if r_t['n_events']>0:
        pca_r,pca_ax=r_t['mean_pca']
        for k,col in enumerate([ROSE,GREEN,AMBER]):
            vx=pca_ax[k,0]*pca_r[k]*2.0; vy=pca_ax[k,1]*pca_r[k]*2.0
            ax4.annotate('',xy=(vx,vy),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
    ax4.set_xlabel("Campo (z-score)",fontsize=9); ax4.set_ylabel("Alignment (z-score)",fontsize=9)
    ax4.set_title("Core Buffer State Space\n★=centroid  Arrows=PCA axes",color=FG,fontsize=9)
    ax4.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none',loc='lower right')
    ax4.grid(True,alpha=0.12); ax4.tick_params(labelsize=7); ax4.set_xlim(-3.5,3.5); ax4.set_ylim(-3.5,3.5)

# ── Summary ───────────────────────────────────────────────────────────────────
pca_r,_=r_t['mean_pca']
coh_state=('COHERENT' if r_t['n_events']>0 and r_t['mean_sr']<1.0 else
           'EVENTS DETECTED' if r_t['n_events']>0 else 'EXPLORING')
summary=[
    f"SFE-05.7  BOUNDARY MAP","─"*28,"",
    f"TARGET:  {target_label}",
    f"mu_threshold: {'FOUND at '+str(mu_threshold) if mu_threshold else 'NOT FOUND'}","",
    "─"*28,"SWEEP SUMMARY (mu → events)",
]
for sr in sweep_results:
    mark='★' if sr['n_events']>0 else ' '
    summary.append(f"{mark} μ={sr['mu']:.3f}  → {sr['n_events']} evts  sr={sr['mean_sr']:.3f}")
summary+=[
    "","─"*28,"PERIODIC",
    f"  A={A_periodic}  ω={omega_periodic}",
    f"  events  = {r_per['n_events']}",
    f"  surp/vol= {r_per['mean_sr']:.4f}","",
    "─"*28,f"STATE at boundary:",f"  {coh_state}",
]
if r_t['n_events']>0:
    summary+=[f"  PCA = [{pca_r[0]:.3f},{pca_r[1]:.3f},{pca_r[2]:.3f}]",
              f"  vol = {r_t['mean_vol_events']:.4f}",
              f"  cen = {r_t['centroid_final'].round(3)}"]

fig2.text(0.698,0.456,"\n".join(summary),fontsize=6.5,fontfamily='monospace',color=FG,va='top',
          bbox=dict(boxstyle='round,pad=0.75',facecolor='#0a0c14',edgecolor=GOLD,linewidth=1.3,alpha=0.97))

plt.savefig(os.path.join(FIG_DIR,'sfe057_manifold.png'),dpi=150,bbox_inches='tight',facecolor=BG)
plt.show(); print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("SFE-05.7  Final Summary — Detection Boundary Mapping"); print("="*70)
print()
print("  DRIFT SWEEP")
print(f"  {'μ':>8}  {'surp/vol':>9}  {'events':>6}  {'first_evt':>9}  {'core_rate':>9}  {'anom_rate':>9}  {'regime':>6}")
print(f"  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}")
for s in sweep_results:
    fe = str(s['first_event_cycle']) if s['first_event_cycle'] is not None else '—'
    mark = '★' if s['n_events'] > 0 else ' '
    print(f"  {mark}{s['mu']:>7.3f}  {s['mean_sr']:>9.4f}  {s['n_events']:>6}  "
          f"{fe:>9}  {s['core_rate']:>9.3f}  {s['anom_rate']:>9.3f}  {s['n_regime']:>6}")

print()
if mu_threshold is not None:
    tr=threshold_result
    print(f"  ★ DETECTION BOUNDARY FOUND")
    print(f"    mu_threshold          = {mu_threshold}")
    print(f"    Surprise/Vol at bound = {tr['mean_sr']:.4f}")
    print(f"    First event at cycle  = {tr['first_event_cycle']}")
    print(f"    Coherence events      = {tr['n_events']}")
    if tr['n_events']>0:
        pca_r,_=tr['mean_pca']
        print(f"    First geometry PCA    = {pca_r.round(3)}")
        print(f"    Centroid              = {tr['centroid_final'].round(4)}")
        print(f"    Hull vol at events    = {tr['mean_vol_events']:.4f}")
else:
    print(f"  Threshold not found within mu ∈ {mu_values}.")
    print(f"  The detector requires structure beyond mu=0.20 to cross coherence boundary.")

print()
print(f"  PERIODIC FORCING (A={A_periodic}, ω={omega_periodic})")
print(f"    Events detected         = {r_per['n_events']}")
print(f"    Surprise/Volatility     = {r_per['mean_sr']:.4f}")
print(f"    First event             = {r_per['first_event_cycle']}")
print(f"    Core write rate         = {r_per['core_rate']:.3f}")
if mu_threshold is not None and r_per['n_events'] > 0:
    pr = r_per['mean_sr']
    tr_sr = threshold_result['mean_sr']
    print(f"\n    Periodic vs drift at threshold:")
    print(f"      Periodic surp/vol = {pr:.4f}")
    print(f"      Drift surp/vol    = {tr_sr:.4f}")
    print(f"      Periodic detects at lower amplitude: {'YES' if r_per['n_events']>0 else 'NO'}")
print("="*70)
