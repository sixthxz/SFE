# -*- coding: utf-8 -*-
"""
SFE-05.8 — Harmonic Potential Well · Spring Constant Sweep
============================================================
Stochastic Field Engine, revision 05.8

CONTEXT:
    SFE-05.7 proved drift cannot reduce surprise — it is unmodeled bias.
    The Kalman filter must model the same dynamics as the field.
    A harmonic potential F=-k*(x-x0) creates an OU process where the
    particle returns to x0 with correlation time tau_corr=1/k.
    When the Kalman prediction step also models OU mean-reversion,
    innovations shrink as the filter accumulates evidence of the
    restoring force. This is the only field modification compatible
    with Surprise/Volatility < 1.0 under the current architecture.

FIELD MODIFICATION:
    F_total = F_tilt + (-k*(x-x0))          # Langevin
    x_hat_prior = x_hat * (1 - k*dt)        # Kalman predict (OU)
    P_prior = P*(1-k*dt)^2 + Q              # Kalman covariance propagate

SWEEP:
    k ∈ [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    k=0.0 → free diffusion (SFE-05.7 baseline, zero coherence)
    Predicted threshold: k ≈ 0.1

ARCHITECTURE: unchanged from SFE-05.6/05.7
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
import os; FIG_DIR = '/tmp/sfe058_figs'; os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.8  —  Harmonic Potential Well · Spring Constant Sweep")
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

# ─── Inherited parameters (SFE-05.6/05.7, unchanged) ─────────────────────────
lambda_coup_init = 0.30; lambda_min = 0.05; lambda_max = 0.80
sigma_m = 0.90; sigma_memory = 1.20; coherence_var_th = 0.15; window_scale = 1.0
alpha_attract = 0.15; min_events_before_pull = 10
n_candidates = 5; rehearsal_on = True
dt_coherent = dt; dt_surprised = dt * 0.5
BUFFER_N = 512; COHERENCE_WIN = 40
gate_multiplier = 1.0; gate_window = 50
anomaly_buffer_size = 256; regime_shift_multiplier = 2.0; min_core_rate = 0.20

# ─── Sweep parameters ─────────────────────────────────────────────────────────
k_values = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
x0       = 0.0   # potential well center

# ─── Field volatility (free diffusion, for reference) ─────────────────────────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

print(f"  k sweep: {k_values}")
print(f"  x0 = {x0}  (potential well center)")
print(f"  field_volatility (k=0) = {field_volatility:.4f}")
print(f"  Predicted k_threshold ≈ 0.1  (tau_corr = 10 × tau_meas*dt)")
print(f"  gate_multiplier={gate_multiplier}  floor=field_volatility")
print(f"  coherence: surp/vol < 1.0 AND hull_var < {coherence_var_th}")
print(f"             AND core_rate >= {min_core_rate}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS (OU-aware)
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5*((x_grid - mu)/sigma)**2)
    return rho / np.trapezoid(rho, x_grid)

def fp_flux(rho, F_arr):
    return (F_arr/gamma)*rho - D_diff*np.gradient(rho, x_grid)

def fp_step_ou(rho, F_arr, k_spring, x0_well):
    """Fokker-Planck step with harmonic restoring force included in F."""
    # F_arr already contains tilt force; add OU restoring on the grid
    F_total = F_arr - k_spring * (x_grid - x0_well)
    N_ = len(rho); v = F_total / gamma
    df = np.zeros(N_+1); ff = np.zeros(N_+1)
    for i in range(1, N_):
        vf = 0.5*(v[i-1]+v[i])
        df[i] = vf*rho[i-1] if vf >= 0 else vf*rho[i]
        ff[i] = D_diff*(rho[i]-rho[i-1])/dx
    rho_new = np.maximum(rho - (dt/dx)*np.diff(df - ff), 0.0)
    norm = np.trapezoid(rho_new, x_grid)
    return rho_new/norm if norm > 1e-12 else rho_new

def compute_alignment(x_pos, rho_true, rho_perc):
    g_t = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_p = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_t) < 1e-10 or abs(g_p) < 1e-10: return 0.0
    return float(np.clip(g_t*g_p/(abs(g_t)*abs(g_p)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN WITH OU PREDICTION (key addition for SFE-05.8)
# ═════════════════════════════════════════════════════════════════════════════

class KalmanOU:
    """
    Kalman filter that models OU mean reversion in its prediction step.

    Prediction for OU process x_{t+1} = x_t * exp(-k*dt) + 0*(1-exp(-k*dt)) + noise
    Linearized (small k*dt): x_hat_prior = x_hat * (1 - k*dt*n)
    Covariance:  P_prior = P*(1-k*dt*n)^2 + Q_eff

    For k=0, reduces exactly to free-diffusion Kalman.
    """
    def __init__(self, k_spring=0.0):
        self.k     = k_spring
        self.x_hat = 0.0
        self.P     = 2*kBT/gamma*tau_meas*dt
        self.Q     = 2*kBT/gamma*dt

    def predict_n(self, n):
        """Predict n steps ahead with OU dynamics."""
        if self.k == 0.0:
            self.P += n * self.Q
        else:
            # OU exact discrete: alpha = exp(-k*dt)^n = exp(-k*dt*n)
            alpha   = np.exp(-self.k * dt * n)
            sigma_sq_ou = kBT / self.k   # OU equilibrium variance
            self.x_hat  = self.x_hat * alpha   # mean reverts toward x0=0
            self.P      = self.P * alpha**2 + sigma_sq_ou*(1 - alpha**2)

    def update(self, z):
        P_prior = self.P
        K       = self.P / (self.P + sigma_m**2)
        innov   = z - self.x_hat
        self.x_hat += K * innov
        self.P     *= (1 - K)
        return P_prior, abs(innov)

    def I_gain(self, P): return max(0.5*np.log2(1 + P/sigma_m**2), 0.0)

    def reset(self):
        self.x_hat = 0.0
        self.P     = 2*kBT/gamma*tau_meas*dt


# ═════════════════════════════════════════════════════════════════════════════
# ALL OTHER CLASSES (unchanged from SFE-05.6/05.7)
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    def __init__(self, N, cols=3, name='buf'):
        self.N=N; self.cols=cols
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
        mu=data.mean(axis=0); sig=data.std(axis=0)+1e-8
        return (data-mu)/sig,mu,sig

class AdaptiveGate:
    def __init__(self, multiplier, window, floor):
        self.mult=multiplier; self.window=window; self.floor=floor
        self._hist=[]; self.threshold=floor
    def update(self, s):
        self._hist.append(s)
        if len(self._hist)>self.window: self._hist.pop(0)
        if len(self._hist)>=4: self.threshold=max(float(np.std(self._hist))*self.mult,self.floor)
        return self.threshold
    def core_gate(self, s): return s <= self.threshold
    def anomaly_gate(self, s): return s > self.threshold

class RegimeShiftDetector:
    def __init__(self, mult, window):
        self.mult=mult; self.window=window; self._flags=[]; self._rates=[]; self.events=[]
    def update(self, aw, cycle, cpos):
        self._flags.append(int(aw))
        if len(self._flags)>self.window: self._flags.pop(0)
        if len(self._flags)<10: return False
        rate=float(np.mean(self._flags)); self._rates.append(rate)
        if len(self._rates)<10: return False
        bl=float(np.mean(self._rates))+1e-8
        if rate>bl*self.mult:
            self.events.append({'cycle':cycle,'rate':rate,'baseline':bl,
                                'centroid':cpos.copy() if cpos is not None else np.zeros(3)})
            return True
        return False
    @property
    def n(self): return len(self.events)

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
        self.n+=1; self.centroid=((self.n-1)*self.centroid+pos)/self.n; return self.centroid.copy()
    def attraction_force_x(self,campo):
        return alpha_attract*(self.centroid[0]-campo) if self.n>0 else 0.0
    @property
    def is_ready(self): return self.n>=min_events_before_pull
    @property
    def position(self): return self.centroid.copy()

class CoherenceRecorder:
    def __init__(self,var_th,fvol,window,centroid):
        self.var_th=var_th; self.fvol=fvol; self.window=max(window,4)
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


# ═════════════════════════════════════════════════════════════════════════════
# CORE RUN FUNCTION — parameterized by k (spring constant)
# ═════════════════════════════════════════════════════════════════════════════

def run_one(seed=42, k=0.0, verbose=False):
    rng = np.random.default_rng(seed)
    kf  = KalmanOU(k_spring=k)   # OU-aware Kalman
    pf  = PerceivedField()

    # Adaptive field volatility: OU equilibrium std for gate floor
    # For k=0: sigma_eq = field_volatility; for k>0: sigma_eq = sqrt(kBT/k)
    # Use min(field_volatility, sigma_eq) as gate floor so gate tightens with k
    sigma_eq = np.sqrt(kBT/k) if k > 0 else field_volatility
    # Gate floor: use innovation std under OU steady state
    # sigma_innovation ≈ sqrt(P_ss + sigma_m^2) where P_ss ≈ kBT/k for small k
    # For simplicity: use field_volatility as fixed floor (matches SFE-05.6/07)
    gate_floor = field_volatility

    core_buf    = CircularBuffer(BUFFER_N, cols=3)
    anomaly_buf = CircularBuffer(anomaly_buffer_size, cols=3)
    gate        = AdaptiveGate(gate_multiplier, gate_window, floor=gate_floor)
    centroid    = CoherenceCentroid()
    window_len  = max(int(N_cycles*0.5*window_scale), 8)
    recorder    = CoherenceRecorder(coherence_var_th, field_volatility, window_len, centroid)
    regime      = RegimeShiftDetector(regime_shift_multiplier, max(gate_window,30))

    # OU equilibrium rho as initial condition
    if k > 0:
        sigma_ou = np.sqrt(kBT/k)
        rho_true = gaussian_rho(x0, sigma_ou)
    else:
        rho_true = gaussian_rho(0.0, 2.0)
    F_free = np.zeros(Nx)
    x = 0.0; pf.add(x); lam = lambda_coup_init

    sr_log=[]; surp_log=[]; campo_log=[]; gate_log=[]
    vol_log=[]; rv_log=[]; cr_log=[]; ar_log=[]
    event_cycles=[]; centroid_hist=[]; regime_cycles=[]; rehearsal_log=[]
    snap_at={N//8:'early',N//2:'mid',7*N//8:'late'}; snap_va={}
    vol_win=[]; surp_win=[]; total=0; cw=0; aw=0; cycle=0
    n_reset=max(N_cycles//8,1); cur_sr=2.0

    for i in range(N):
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas = x + sigma_m*rng.standard_normal()
            P_prior, surp = kf.update(x_meas)

            rho_perc  = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)
            J_arr     = fp_flux(rho_true, F_free)
            campo     = float(abs(np.interp(x, x_grid, J_arr)))
            cur_sr    = surp / (field_volatility + 1e-10)

            g_th = gate.update(surp); gate_log.append(g_th); total += 1
            wc = gate.core_gate(surp); wa = gate.anomaly_gate(surp)
            if wc: core_buf.push([campo, alignment, surp]); cw += 1
            aw_written = False
            if wa: anomaly_buf.push([campo, alignment, surp]); aw += 1; aw_written = True

            sr_log.append(cur_sr); surp_log.append(surp); campo_log.append(campo)
            surp_win.append(surp)
            if len(surp_win) > COHERENCE_WIN: surp_win.pop(0)

            regime.update(aw_written, cycle, centroid.position if centroid.n>0 else None)
            if regime.n > len(regime_cycles): regime_cycles.append(cycle)
            cycle += 1

            if cycle % COHERENCE_WIN == 0 and core_buf.filled >= 5:
                T_hat,_,_ = core_buf.normalize()
                vol,area,_ = convex_hull_3d(T_hat)
                vol_log.append(vol); vol_win.append(vol)
                if len(vol_win)>window_len: vol_win.pop(0)
                core_rate = cw/max(total,1); anom_rate = aw/max(total,1)
                cr_log.append(core_rate); ar_log.append(anom_rate)
                if len(vol_win)>=3:
                    vm=np.mean(vol_win); vs=np.std(vol_win)+1e-8; z=(vol-vm)/vs
                    if z>1.0:    lam=min(lam*1.05, lambda_max)
                    elif z<-0.5: lam=max(lam*0.98, lambda_min)
                ms=float(np.mean(surp_win)); ms_raw=core_buf.get_numpy().mean(axis=0)
                is_ev,rel_var,surp_rat=recorder.check(vol,ms,T_hat,ms_raw,lam,cycle,core_rate)
                rv_log.append(rel_var)
                if is_ev:
                    event_cycles.append(cycle); centroid_hist.append(centroid.position.copy())
                    if verbose:
                        print(f"      ★ cycle={cycle:4d}  surp/vol={surp_rat:.3f}  "
                              f"core={core_rate:.3f}  gate={g_th:.4f}  "
                              f"cen={centroid.position.round(3)}")

        if (i+1)%(n_reset*tau_meas)==0 and i>0: kf.reset()

        # ── OU Langevin step ──────────────────────────────────────────────
        dt_eff = dt_coherent if cur_sr < 1.0 else dt_surprised
        F_ou   = -k * (x - x0)          # restoring force
        J_arr  = fp_flux(rho_true, F_free)
        J_at   = float(np.interp(x, x_grid, J_arr))
        F_flux = lam * J_at / (abs(J_at)+1e-10)
        F_att  = centroid.attraction_force_x(campo_log[-1] if campo_log else 0.0) \
                 if centroid.is_ready else 0.0
        F_extra_full = F_att + F_ou

        # Counterfactual selection
        def step_candidate(F_ext):
            xi = np.sqrt(2*kBT*gamma)*rng.standard_normal()
            dx_ = ((F_flux + F_ext)/gamma)*dt_eff + xi*np.sqrt(dt_eff)/gamma
            return float(np.clip(x+dx_, x_min+0.1, x_max-0.1))

        x_s = step_candidate(F_extra_full); s_s = abs(x_s - kf.x_hat)
        bx=x_s; bs=s_s; acc=False
        if rehearsal_on and n_candidates > 1:
            for _ in range(n_candidates-1):
                xc = step_candidate(F_extra_full); sc = abs(xc - kf.x_hat)
                if sc < bs: bx=xc; bs=sc; acc=True
        rehearsal_log.append(int(acc))

        if k > 0:
            rho_true = fp_step_ou(rho_true, F_free, k, x0)
        else:
            N_ = len(rho_true); v = F_free/gamma
            df=np.zeros(N_+1); ff=np.zeros(N_+1)
            for ii in range(1,N_):
                vf=0.5*(v[ii-1]+v[ii]); df[ii]=vf*rho_true[ii-1] if vf>=0 else vf*rho_true[ii]
                ff[ii]=D_diff*(rho_true[ii]-rho_true[ii-1])/dx
            rho_new=np.maximum(rho_true-(dt/dx)*np.diff(df-ff),0.0)
            norm=np.trapezoid(rho_new,x_grid); rho_true=rho_new/norm if norm>1e-12 else rho_new

        pf.add(bx, w=1./(sigma_m+0.1)); x = bx

        if i in snap_at:
            T_hat,_,_=core_buf.normalize(); lbl=snap_at[i]
            vol,area,hull=convex_hull_3d(T_hat); snap_va[lbl]=(vol,area,hull,T_hat.copy())

    T_hat_f,_,_=core_buf.normalize(); vol_f,area_f,hull_f=convex_hull_3d(T_hat_f)
    snap_va['final']=(vol_f,area_f,hull_f,T_hat_f.copy())

    crf=cw/max(total,1); arf=aw/max(total,1)
    mean_sr=float(np.mean(sr_log))
    sigma_ou_eq = float(np.sqrt(kBT/k)) if k>0 else float('inf')

    return dict(
        k=k, mean_sr=mean_sr, n_events=recorder.n,
        first_event_cycle=event_cycles[0] if event_cycles else None,
        core_rate=crf, anom_rate=arf, n_regime=regime.n,
        sigma_ou_eq=sigma_ou_eq,
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
# SPRING CONSTANT SWEEP
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("SPRING CONSTANT SWEEP")
print("─" * 70)

sweep_results = []
for k in k_values:
    sigma_ou_str = f"σ_OU={np.sqrt(kBT/k):.3f}" if k > 0 else "σ_OU=∞"
    print(f"  k={k:.3f}  {sigma_ou_str:12s}  ", end='', flush=True)
    r = run_one(seed=42, k=k, verbose=(k >= 0.05))
    sweep_results.append(r)
    status = '★ COH' if r['n_events'] > 0 else '—'
    print(f"surp/vol={r['mean_sr']:.4f}  events={r['n_events']}  "
          f"core={r['core_rate']:.3f}  anom={r['anom_rate']:.3f}  "
          f"regime={r['n_regime']}  {status}")

# Detection boundary
k_threshold    = None
threshold_result = None
for sr in sweep_results:
    if sr['n_events'] > 0:
        k_threshold    = sr['k']
        threshold_result = sr
        break

print()
if k_threshold is not None:
    tr = threshold_result
    tau_corr = 1.0/k_threshold
    tau_meas_norm = tau_meas * dt
    ratio_tau = tau_corr / tau_meas_norm
    print(f"  ★ DETECTION BOUNDARY FOUND")
    print(f"    k_threshold           = {k_threshold}")
    print(f"    tau_corr              = 1/k = {tau_corr:.2f}  (OU correlation time)")
    print(f"    tau_meas_norm         = {tau_meas_norm:.3f}")
    print(f"    ratio_tau             = tau_corr / tau_meas_norm = {ratio_tau:.1f}")
    print(f"    Surprise/Vol          = {tr['mean_sr']:.4f}")
    print(f"    First event at cycle  = {tr['first_event_cycle']}")
    print(f"    OU equilibrium std    = {tr['sigma_ou_eq']:.4f}")
    if tr['n_events'] > 0:
        pca_r,_ = tr['mean_pca']
        print(f"    First geometry PCA    = {pca_r.round(3)}")
        print(f"    Hull volume at events = {tr['mean_vol_events']:.4f}")
        print(f"    Centroid              = {tr['centroid_final'].round(4)}")
else:
    print(f"  Threshold not found within k ∈ {k_values}.")
    print(f"  Higher spring constants required.")


# ═════════════════════════════════════════════════════════════════════════════
# SWEEP VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering sweep figure...", end='', flush=True)

BG='#07080f'; FG='#dde1ec'; GOLD='#f5c842'; TEAL='#3dd6c8'; VIOLET='#b87aff'
ROSE='#ff5f7e'; GREEN='#4ade80'; AMBER='#fb923c'; COH='#fde68a'
REGIME='#f97316'; DIM='#1e2235'; WH='#ffffff'

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':DIM,
    'text.color':FG,'axes.labelcolor':FG,'xtick.color':'#555870','ytick.color':'#555870'})

ks       = np.array([s['k']        for s in sweep_results])
mean_srs = np.array([s['mean_sr']  for s in sweep_results])
n_evs    = np.array([s['n_events'] for s in sweep_results])
fe_cycs  = [s['first_event_cycle'] for s in sweep_results]
core_rs  = np.array([s['core_rate']  for s in sweep_results])
anom_rs  = np.array([s['anom_rate']  for s in sweep_results])
sigma_ous = np.array([s['sigma_ou_eq'] for s in sweep_results])
n_regimes = np.array([s['n_regime']  for s in sweep_results])

k_plot = np.where(ks == 0, 5e-4, ks)   # for log scale

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig.suptitle("SFE-05.8  ·  Harmonic Potential Well — Spring Constant Sweep\n"
             "k ∈ [0.0, 1.0]  |  OU process  |  Kalman models mean reversion",
             fontsize=13, color=GOLD, y=1.001, fontweight='bold')
for ax in axes.flat:
    ax.set_facecolor(BG); ax.grid(True, alpha=0.12); ax.tick_params(labelsize=8)

def vline_k(ax, color=COH, ls='--'):
    if k_threshold is not None:
        kp = k_threshold if k_threshold > 0 else 5e-4
        ax.axvline(kp, color=color, lw=1.8, ls=ls, alpha=0.85,
                   label=f'k_threshold={k_threshold}')

# Panel 0: Surprise/Volatility vs k
ax0 = axes[0,0]
ax0.semilogx(k_plot, mean_srs, 'o-', color=ROSE, lw=2, ms=8, markeredgewidth=0,
             label='Surprise/Volatility')
ax0.axhline(1.0, color=GREEN, lw=1.5, ls='--', label='Coherence target = 1.0')
ax0.axhline(mean_srs[0], color=TEAL, lw=0.9, ls=':', alpha=0.65,
            label=f'Free diffusion = {mean_srs[0]:.3f}')
vline_k(ax0)
ax0.set_xlabel("k (spring constant, log scale)", fontsize=9)
ax0.set_ylabel("Mean Surprise / Volatility", fontsize=9)
ax0.set_title("Surprise Reduction vs Spring Constant", color=FG, fontsize=10)
ax0.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 1: Coherence events vs k
ax1 = axes[0,1]
ax1.semilogx(k_plot, n_evs, 's-', color=COH, lw=2, ms=9, markeredgewidth=0,
             label='Coherence events')
vline_k(ax1)
ax1.set_xlabel("k (log scale)", fontsize=9)
ax1.set_ylabel("Coherence Events", fontsize=9)
ax1.set_title("First Honest Coherence vs Spring Constant", color=FG, fontsize=10)
ax1.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')
# Annotate first detection
for i, (kp, ne) in enumerate(zip(k_plot, n_evs)):
    if ne > 0:
        ax1.annotate(f'k={ks[i]}', (kp, ne), textcoords='offset points',
                     xytext=(6,6), fontsize=7, color=COH)

# Panel 2: First event cycle vs k
ax2 = axes[0,2]
valid_k  = [k_plot[i] for i,v in enumerate(fe_cycs) if v is not None]
valid_fe = [v for v in fe_cycs if v is not None]
if valid_k:
    ax2.semilogx(valid_k, valid_fe, '^-', color=TEAL, lw=2, ms=9, markeredgewidth=0,
                 label='First coherence event')
vline_k(ax2)
ax2.set_xlabel("k (log scale)", fontsize=9)
ax2.set_ylabel("Cycle", fontsize=9)
ax2.set_title("Detection Speed vs Spring Constant\n(lower = faster detection)", color=FG, fontsize=10)
ax2.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 3: core/anomaly write rate vs k
ax3 = axes[1,0]
ax3.semilogx(k_plot, core_rs, 'o-', color=GREEN, lw=2, ms=8, label='Core write rate')
ax3.semilogx(k_plot, anom_rs, 'o--', color=REGIME, lw=2, ms=8, alpha=0.8, label='Anomaly rate')
ax3.axhspan(0.30, 0.70, alpha=0.07, color=GREEN)
vline_k(ax3)
ax3.set_xlabel("k (log scale)", fontsize=9); ax3.set_ylabel("Write Rate", fontsize=9)
ax3.set_title("Dual Buffer Write Rates vs k\nGreen band = target range", color=FG, fontsize=10)
ax3.set_ylim(0, 1.1); ax3.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 4: OU sigma_eq vs k (reference)
ax4 = axes[1,1]
k_ref  = k_values[1:]   # skip k=0
sig_ref = [np.sqrt(kBT/kk) for kk in k_ref]
ax4.loglog([k for k in k_plot if k > 5e-4], sig_ref, 'D-', color=VIOLET, lw=2, ms=8,
           label='σ_OU = √(kBT/k)')
ax4.axhline(field_volatility, color=TEAL, lw=1.2, ls='--',
            label=f'σ_field = {field_volatility:.4f}')
ax4.axhline(sigma_m, color=AMBER, lw=1.0, ls=':', label=f'σ_meas = {sigma_m}')
vline_k(ax4)
ax4.set_xlabel("k (log scale)", fontsize=9); ax4.set_ylabel("σ_OU (log scale)", fontsize=9)
ax4.set_title("OU Equilibrium Std vs k\nAt k_threshold: σ_OU vs σ_field", color=FG, fontsize=10)
ax4.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')

# Panel 5: Summary
ax5 = axes[1,2]; ax5.axis('off')
tau_info = ""
if k_threshold is not None:
    tau_corr = 1.0/k_threshold
    tau_mn = tau_meas * dt
    ratio_tau = tau_corr/tau_mn
    tau_info = (f"\n  tau_corr = 1/k = {tau_corr:.2f}"
                f"\n  tau_meas_norm = {tau_mn:.3f}"
                f"\n  ratio_tau = {ratio_tau:.1f} cycles")

lines = [
    "SFE-05.8  BOUNDARY MAP", "─"*30, "",
    f"k sweep: {k_values}", "",
    f"BASELINE (k=0.0)",
    f"  surp/vol  = {mean_srs[0]:.4f}",
    f"  core_rate = {core_rs[0]:.3f}",
    f"  events    = {n_evs[0]}", "",
    f"DETECTION BOUNDARY",
    f"  k_threshold = {'FOUND: '+str(k_threshold) if k_threshold else 'NOT FOUND'}",
]
if k_threshold is not None:
    tr = threshold_result
    pca_r,_ = tr['mean_pca']
    lines += [
        tau_info,
        f"\n  surp/vol  = {tr['mean_sr']:.4f}",
        f"  first evt = cycle {tr['first_event_cycle']}",
        f"  σ_OU_eq   = {tr['sigma_ou_eq']:.4f}",
        f"  core_rate = {tr['core_rate']:.3f}",
        f"  n_events  = {tr['n_events']}",
        f"  PCA ratio = [{pca_r[0]:.3f},{pca_r[1]:.3f},{pca_r[2]:.3f}]",
        f"  centroid  = {tr['centroid_final'].round(3)}",
    ]
lines += ["", "─"*30, "SWEEP SUMMARY (k → events  sr)"]
for s in sweep_results:
    mark='★' if s['n_events']>0 else ' '
    lines.append(f"{mark} k={s['k']:.2f}  {s['n_events']}evt  sr={s['mean_sr']:.3f}")

ax5.text(0.03, 0.98, "\n".join(lines),
         transform=ax5.transAxes, fontsize=7, fontfamily='monospace', color=FG, va='top',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0c14', edgecolor=GOLD, lw=1.2))

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(FIG_DIR,'sfe058_sweep.png'), dpi=150, bbox_inches='tight', facecolor=BG)
plt.show(); print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# MANIFOLD WORLD — at k_threshold
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering manifold world...", end='', flush=True)

r_t = threshold_result if threshold_result is not None else sweep_results[-1]
k_lbl = f"k={r_t['k']}"
snaps = r_t['snap_va']

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
    ax.grid(True, alpha=0.10)

fig2 = plt.figure(figsize=(24, 18), facecolor=BG)
tau_str = f"  τ_corr={1/r_t['k']:.1f}  τ_meas={tau_meas*dt:.3f}  ratio={1/(r_t['k']*tau_meas*dt):.1f}×" \
          if r_t['k'] > 0 else "  Free diffusion"
found_str = f"DETECTION BOUNDARY FOUND at {k_lbl}" if k_threshold else f"Highest k={k_lbl}"
fig2.suptitle(
    f"SFE-05.8  ·  {found_str}\n{tau_str}",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold')
gs = GridSpec(2,3,figure=fig2,hspace=0.42,wspace=0.34,top=0.966,bottom=0.06,left=0.04,right=0.97)

# ── Panel 0: Core buffer + hull + centroid ────────────────────────────────────
ax0=fig2.add_subplot(gs[0,:2],projection='3d'); ax0.set_facecolor(BG)
pts_f=r_t['buf_final']; _,_,hf=convex_hull_3d(pts_f)
sc=ax0.scatter(pts_f[:,0],pts_f[:,1],pts_f[:,2],c=np.linspace(0,1,len(pts_f)),
               cmap='plasma',s=7,alpha=0.70,linewidths=0)
draw_hull(ax0,pts_f,hf,TEAL,TEAL,0.07,0.18)
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
         f"Core Buffer at {k_lbl}  [★=centroid]\n"
         f"events={r_t['n_events']}  core_rate={r_t['core_rate']:.3f}  "
         f"surp/vol={r_t['mean_sr']:.3f}  λ={r_t['lam_final']:.3f}")
ax0.view_init(elev=22, azim=-52)

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
if r_t['n_events']>0:
    handles.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COH,ms=10,
                          label=f"1st coherence V={r_t['mean_vol_events']:.3f}"))
ax1.legend(handles=handles,fontsize=6.5,facecolor='#0d0f18',edgecolor='none')
style_3d(ax1,"Campo","Alignment","Surprise",f"Hull Evolution at {k_lbl}")
ax1.view_init(elev=25, azim=-40)

# ── Panel 2: Volume + variance ────────────────────────────────────────────────
ax2=fig2.add_subplot(gs[1,0]); ax2.set_facecolor(BG)
va=r_t['vol_log']; rv=r_t['rv_log']
if len(va)>0:
    tv=np.arange(len(va))*COHERENCE_WIN; tr2=np.arange(len(rv))*COHERENCE_WIN
    ax2.fill_between(tv,0,va,alpha=0.18,color=VIOLET); ax2.plot(tv,va,color=VIOLET,lw=1.8,label='Hull volume')
    ax2.axhline(np.mean(va),color=TEAL,lw=0.7,ls=':',alpha=0.5)
    ax2b=ax2.twinx(); ax2b.set_facecolor(BG)
    if len(rv)>0:
        ax2b.plot(tr2,rv,color=GOLD,lw=1.5,ls='--',label='Rel. variance')
        ax2b.axhline(coherence_var_th,color=GREEN,lw=1.2,ls=':',label=f'threshold={coherence_var_th}')
        ax2b.fill_between(tr2,0,rv,where=(rv<coherence_var_th),alpha=0.22,color=GREEN)
    ax2b.set_ylabel("Var(V)/mean(V)²",color=GOLD,fontsize=8); ax2b.tick_params(axis='y',labelcolor=GOLD,labelsize=7)
    for ec in r_t['event_cycles']: ax2.axvline(ec,color=COH,lw=0.9,alpha=0.6,ls='--')
    ax2.set_xlabel("Cycle",fontsize=9); ax2.set_ylabel("Hull Volume",color=VIOLET,fontsize=9)
    ax2.set_title(f"Volume + Variance at {k_lbl}\nGold dashed = coherence events",color=FG,fontsize=9)
    ax2.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
    ax2b.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper left')
    ax2.grid(True,alpha=0.12); ax2.tick_params(labelsize=7)

# ── Panel 3: Surprise ratio baseline vs k_threshold ──────────────────────────
ax3=fig2.add_subplot(gs[1,1]); ax3.set_facecolor(BG)
sr=r_t['sr_log']; cyc=np.arange(len(sr))
if len(sr)>0:
    win=max(len(sr)//40,3); sr_s=uniform_filter1d(sr,size=win)
    ax3.fill_between(cyc,0,sr,alpha=0.09,color=ROSE)
    ax3.plot(cyc,sr_s,color=ROSE,lw=2.0,label=f'Surprise/Vol ({k_lbl})')
    sr0=sweep_results[0]['sr_log']; sr0_s=uniform_filter1d(sr0,size=win)
    ax3.plot(cyc[:len(sr0_s)],sr0_s,color=TEAL,lw=1.3,ls='--',alpha=0.7,label='Baseline k=0')
    ax3.axhline(1.0,color=GREEN,lw=1.3,ls='--',label='Target = 1.0')
    for ec in r_t['event_cycles']: ax3.axvline(ec,color=COH,lw=0.8,alpha=0.55,ls='--')
    ax3.set_xlabel("Cycle",fontsize=9); ax3.set_ylabel("Surprise / Field Volatility",fontsize=9)
    ax3.set_title(f"Surprise: Baseline vs {k_lbl}\nGold dashed = coherence events",color=FG,fontsize=9)
    ax3.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none')
    ax3.grid(True,alpha=0.12); ax3.tick_params(labelsize=7); ax3.set_xlim(0,len(sr))

# ── Panel 4: Campo × Alignment + PCA ─────────────────────────────────────────
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
        for ki,col in enumerate([ROSE,GREEN,AMBER]):
            vx=pca_ax[ki,0]*pca_r[ki]*2.0; vy=pca_ax[ki,1]*pca_r[ki]*2.0
            ax4.annotate('',xy=(vx,vy),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
    ax4.set_xlabel("Campo (z-score)",fontsize=9); ax4.set_ylabel("Alignment (z-score)",fontsize=9)
    ax4.set_title("Core Buffer State Space\n★=centroid  Arrows=PCA axes of coherence sig.",color=FG,fontsize=9)
    ax4.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none',loc='lower right')
    ax4.grid(True,alpha=0.12); ax4.tick_params(labelsize=7); ax4.set_xlim(-3.5,3.5); ax4.set_ylim(-3.5,3.5)

# ── Summary ───────────────────────────────────────────────────────────────────
pca_r,_=r_t['mean_pca']
coh_state=('COHERENT' if r_t['n_events']>0 and r_t['mean_sr']<1.0 else
           'EVENTS DETECTED' if r_t['n_events']>0 else 'EXPLORING')
tau_summary = ""
if r_t['k'] > 0:
    tau_corr = 1.0/r_t['k']; ratio_tau = tau_corr/(tau_meas*dt)
    tau_summary = f"\n  tau_corr = {tau_corr:.2f}\n  tau_ratio = {ratio_tau:.1f}×"

summary=[
    f"SFE-05.8  {found_str}","─"*28,"",
    f"k_threshold: {'FOUND: '+str(k_threshold) if k_threshold else 'NOT FOUND'}",
    tau_summary,"",
    "─"*28, f"AT {k_lbl}:",
    f"  surp/vol  = {r_t['mean_sr']:.4f}",
    f"  σ_OU_eq   = {r_t['sigma_ou_eq']:.4f}",
    f"  n_events  = {r_t['n_events']}",
    f"  core_rate = {r_t['core_rate']:.3f}",
    f"  anom_rate = {r_t['anom_rate']:.3f}",
    f"  λ_final   = {r_t['lam_final']:.3f}",
    f"  rehearsal = {r_t['rehearsal_rate']:.3f}",
]
if r_t['n_events']>0:
    summary += [
        "",f"FIRST GEOMETRY",
        f"  vol  = {r_t['mean_vol_events']:.4f}",
        f"  PCA  = [{pca_r[0]:.3f},{pca_r[1]:.3f},{pca_r[2]:.3f}]",
        f"  cen  = {r_t['centroid_final'].round(3)}",
    ]
summary += ["","─"*28,f"STATE: {coh_state}"]

fig2.text(0.698,0.456,"\n".join(summary),fontsize=6.8,fontfamily='monospace',color=FG,va='top',
          bbox=dict(boxstyle='round,pad=0.75',facecolor='#0a0c14',edgecolor=GOLD,linewidth=1.3,alpha=0.97))

plt.savefig(os.path.join(FIG_DIR,'sfe058_manifold.png'),dpi=150,bbox_inches='tight',facecolor=BG)
plt.show(); print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("SFE-05.8  Final Summary — Harmonic Potential Well Sweep"); print("="*70)
print()
print(f"  {'k':>7}  {'σ_OU_eq':>8}  {'surp/vol':>9}  {'events':>6}  {'first_evt':>9}  {'core':>6}  {'anom':>6}")
print(f"  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*9}  {'─'*6}  {'─'*6}")
for s in sweep_results:
    sig = f"{s['sigma_ou_eq']:.3f}" if s['k']>0 else "   ∞"
    fe  = str(s['first_event_cycle']) if s['first_event_cycle'] else '—'
    mark = '★' if s['n_events']>0 else ' '
    print(f"  {mark}{s['k']:>6.3f}  {sig:>8}  {s['mean_sr']:>9.4f}  "
          f"{s['n_events']:>6}  {fe:>9}  {s['core_rate']:>6.3f}  {s['anom_rate']:>6.3f}")

print()
if k_threshold is not None:
    tr = threshold_result
    tau_corr = 1.0/k_threshold
    ratio_tau = tau_corr/(tau_meas*dt)
    pca_r,_ = tr['mean_pca']
    print(f"  ★ DETECTION BOUNDARY FOUND at k_threshold = {k_threshold}")
    print(f"    tau_corr = 1/k = {tau_corr:.2f}  (OU correlation time)")
    print(f"    ratio_tau = tau_corr / (tau_meas*dt) = {ratio_tau:.1f}  measurement cycles")
    print(f"    Surprise/Volatility    = {tr['mean_sr']:.4f}  (target < 1.0)")
    print(f"    First coherence event  = cycle {tr['first_event_cycle']}")
    print(f"    Coherence events total = {tr['n_events']}")
    print(f"    σ_OU equilibrium       = {tr['sigma_ou_eq']:.4f}")
    print(f"    Core write rate        = {tr['core_rate']:.3f}")
    print()
    print(f"  FIRST EMERGENT COHERENCE GEOMETRY:")
    print(f"    Hull volume at events  = {tr['mean_vol_events']:.4f}")
    print(f"    PCA variance ratio     = {pca_r.round(3)}")
    print(f"    Centroid               = {tr['centroid_final'].round(4)}")
    ev0 = tr['events'][0]
    print(f"    First event surp/vol   = {ev0['surp_ratio']:.4f}")
    print(f"    First event rel_var    = {ev0['rel_var']:.4f}")
    print(f"    First event lambda     = {ev0['lambda']:.4f}")
else:
    print(f"  Threshold not found within k ∈ {k_values}.")
    print(f"  Higher spring constants or longer runs required.")

print()
print("  EVOLUTION FROM SFE-05.7")
print("    Free diffusion    →  OU process with spring constant k")
print("    Blind Kalman      →  Kalman models OU mean reversion")
print("    Drift (unmodeled) →  Restoring force (modeled, correlated)")
print("    Zero coherence    →  Detection boundary mapped")
print("="*70)
