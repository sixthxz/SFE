# -*- coding: utf-8 -*-
"""
SFE-05.9 — Sensor Calibration Validation
==========================================
Stochastic Field Engine, revision 05.9

CONTEXT (from SFE-05.8):
    Theoretical innovation floor = sqrt(2/pi) * sigma_m
    With sigma_m=0.90: floor ratio = 1.606 — coherence unreachable
    With sigma_m<0.560: floor ratio < 1.0 — coherence theoretically reachable

    This is a validation run, not an exploration run.
    Single controlled change: sigma_m = 0.40 (was 0.90).

RUN STRUCTURE:
    Run 1: k=0.0  sigma_m=0.40  → must produce ZERO coherence (gate check)
    Run 2: k=1.0  sigma_m=0.40  → expected coherence events + ratio < 1.0
    Run 3: k=0.1  sigma_m=0.40  → predicted threshold from SFE-05.8 analysis

THEORETICAL PREDICTIONS:
    sigma_m=0.40 floor = sqrt(2/pi)*0.40 = 0.319
    field_volatility   = 0.447
    expected ratio floor = 0.714
    coherence target     = ratio < 1.0  → ACHIEVABLE

ALL ELSE UNCHANGED from SFE-05.8.
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
import os; FIG_DIR = '/tmp/sfe059_figs'; os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.9  —  Sensor Calibration Validation")
print("=" * 70)
print()

# ─── Units ────────────────────────────────────────────────────────────────────
kBT = 1.0; gamma = 1.0; D_diff = kBT / gamma; Landauer = np.log(2)

# ─── Simulation ───────────────────────────────────────────────────────────────
dt = 0.01; N = 20000; tau_meas = 10; N_cycles = N // tau_meas

# ─── Field grid ───────────────────────────────────────────────────────────────
x_min, x_max = -8.0, 8.0; Nx = 400
x_grid = np.linspace(x_min, x_max, Nx); dx = x_grid[1] - x_grid[0]

# ─── Inherited parameters (unchanged) ────────────────────────────────────────
lambda_coup_init = 0.30; lambda_min = 0.05; lambda_max = 0.80
sigma_memory = 1.20; coherence_var_th = 0.15; window_scale = 1.0
alpha_attract = 0.15; min_events_before_pull = 10
n_candidates = 5; rehearsal_on = True
dt_coherent = dt; dt_surprised = dt * 0.5
BUFFER_N = 512; COHERENCE_WIN = 40
gate_multiplier = 1.0; gate_window = 50
anomaly_buffer_size = 256; regime_shift_multiplier = 2.0; min_core_rate = 0.20
x0 = 0.0   # OU potential center

# ─── KEY CHANGE ──────────────────────────────────────────────────────────────
sigma_m = 0.40   # was 0.90 — now below derived threshold of 0.560

# ─── Field volatility ─────────────────────────────────────────────────────────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

# ─── Theoretical predictions ─────────────────────────────────────────────────
floor_old = np.sqrt(2/np.pi) * 0.90
floor_new = np.sqrt(2/np.pi) * sigma_m
ratio_floor_old = floor_old / field_volatility
ratio_floor_new = floor_new / field_volatility

print(f"  sigma_m (new)     = {sigma_m}  (was 0.90)")
print(f"  field_volatility  = {field_volatility:.4f}")
print(f"  Theoretical floor (sigma_m=0.90) = {floor_old:.4f}  ratio = {ratio_floor_old:.4f}")
print(f"  Theoretical floor (sigma_m={sigma_m})  = {floor_new:.4f}  ratio = {ratio_floor_new:.4f}")
print(f"  Coherence target  = ratio < 1.0  → {'ACHIEVABLE' if ratio_floor_new < 1.0 else 'UNREACHABLE'}")
print()
print("  Run 1: k=0.0, sigma_m=0.40  → expect ZERO coherence (gate check)")
print("  Run 2: k=1.0, sigma_m=0.40  → expect coherence events + ratio < 1.0")
print("  Run 3: k=0.1, sigma_m=0.40  → threshold confirmation")
print()


# ═════════════════════════════════════════════════════════════════════════════
# CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    def __init__(self, N, cols=3):
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
    def __init__(self, mult, window, floor):
        self.mult=mult; self.window=window; self.floor=floor
        self._hist=[]; self.threshold=floor
    def update(self, s):
        self._hist.append(s)
        if len(self._hist)>self.window: self._hist.pop(0)
        if len(self._hist)>=4: self.threshold=max(float(np.std(self._hist))*self.mult,self.floor)
        return self.threshold
    def core_gate(self, s): return s<=self.threshold
    def anomaly_gate(self, s): return s>self.threshold

class RegimeShiftDetector:
    def __init__(self,mult,window):
        self.mult=mult; self.window=window; self._flags=[]; self._rates=[]; self.events=[]
    def update(self,aw,cycle,cpos):
        self._flags.append(int(aw))
        if len(self._flags)>self.window: self._flags.pop(0)
        if len(self._flags)<10: return False
        rate=float(np.mean(self._flags)); self._rates.append(rate)
        if len(self._rates)<10: return False
        bl=float(np.mean(self._rates))+1e-8
        if rate>bl*self.mult:
            self.events.append({'cycle':cycle,'rate':rate}); return True
        return False
    @property
    def n(self): return len(self.events)

class KalmanOU:
    """OU-aware Kalman: prediction step uses exact OU discrete dynamics."""
    def __init__(self, k_spring=0.0):
        self.k=k_spring; self.x_hat=0.0
        self.P=2*kBT/gamma*tau_meas*dt; self.Q=2*kBT/gamma*dt
    def predict_n(self, n):
        if self.k==0.0:
            self.P+=n*self.Q
        else:
            alpha=np.exp(-self.k*dt*n); sigma_sq_ou=kBT/self.k
            self.x_hat=self.x_hat*alpha
            self.P=self.P*alpha**2+sigma_sq_ou*(1-alpha**2)
    def update(self, z):
        P_prior=self.P; K=self.P/(self.P+sigma_m**2)
        innov=z-self.x_hat; self.x_hat+=K*innov; self.P*=(1-K)
        return P_prior,abs(innov)
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

def gaussian_rho(mu,sigma):
    rho=np.exp(-0.5*((x_grid-mu)/sigma)**2); return rho/np.trapezoid(rho,x_grid)

def fp_flux(rho,F_arr): return (F_arr/gamma)*rho-D_diff*np.gradient(rho,x_grid)

def fp_step_ou(rho,F_arr,k_spring,x0_well):
    F_total=F_arr-k_spring*(x_grid-x0_well); N_=len(rho); v=F_total/gamma
    df=np.zeros(N_+1); ff=np.zeros(N_+1)
    for i in range(1,N_):
        vf=0.5*(v[i-1]+v[i]); df[i]=vf*rho[i-1] if vf>=0 else vf*rho[i]
        ff[i]=D_diff*(rho[i]-rho[i-1])/dx
    rho_new=np.maximum(rho-(dt/dx)*np.diff(df-ff),0.0)
    norm=np.trapezoid(rho_new,x_grid); return rho_new/norm if norm>1e-12 else rho_new

def compute_alignment(x_pos,rho_true,rho_perc):
    g_t=float(np.interp(x_pos,x_grid,np.gradient(rho_true,x_grid)))
    g_p=float(np.interp(x_pos,x_grid,np.gradient(rho_perc,x_grid)))
    if abs(g_t)<1e-10 or abs(g_p)<1e-10: return 0.0
    return float(np.clip(g_t*g_p/(abs(g_t)*abs(g_p)),-1.0,1.0))

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
# CORE RUN FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_one(seed=42, k=0.0, verbose=True):
    rng=np.random.default_rng(seed); kf=KalmanOU(k_spring=k); pf=PerceivedField()
    core_buf=CircularBuffer(BUFFER_N,cols=3); anomaly_buf=CircularBuffer(anomaly_buffer_size,cols=3)
    gate=AdaptiveGate(gate_multiplier,gate_window,floor=field_volatility)
    centroid=CoherenceCentroid()
    window_len=max(int(N_cycles*0.5*window_scale),8)
    recorder=CoherenceRecorder(coherence_var_th,field_volatility,window_len,centroid)
    regime=RegimeShiftDetector(regime_shift_multiplier,max(gate_window,30))

    rho_true=gaussian_rho(x0,np.sqrt(kBT/k)) if k>0 else gaussian_rho(0.0,2.0)
    F_free=np.zeros(Nx); x=0.0; pf.add(x); lam=lambda_coup_init

    sr_log=[]; surp_log=[]; campo_log=[]; gate_log=[]
    vol_log=[]; rv_log=[]; cr_log=[]; ar_log=[]
    event_cycles=[]; centroid_hist=[]
    snap_at={N//8:'early',N//2:'mid',7*N//8:'late'}; snap_va={}
    vol_win=[]; surp_win=[]; total=0; cw=0; aw=0; cycle=0
    n_reset=max(N_cycles//8,1); cur_sr=2.0

    for i in range(N):
        if i%tau_meas==0:
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
            aw_w=False
            if wa: anomaly_buf.push([campo,alignment,surp]); aw+=1; aw_w=True
            sr_log.append(cur_sr); surp_log.append(surp); campo_log.append(campo)
            surp_win.append(surp)
            if len(surp_win)>COHERENCE_WIN: surp_win.pop(0)
            regime.update(aw_w,cycle,centroid.position if centroid.n>0 else None)
            cycle+=1
            if cycle%COHERENCE_WIN==0 and core_buf.filled>=5:
                T_hat,_,_=core_buf.normalize()
                vol,area,_=convex_hull_3d(T_hat); vol_log.append(vol); vol_win.append(vol)
                if len(vol_win)>window_len: vol_win.pop(0)
                core_rate=cw/max(total,1); anom_rate=aw/max(total,1)
                cr_log.append(core_rate); ar_log.append(anom_rate)
                if len(vol_win)>=3:
                    vm=np.mean(vol_win); vs=np.std(vol_win)+1e-8; z=(vol-vm)/vs
                    if z>1.0:    lam=min(lam*1.05,lambda_max)
                    elif z<-0.5: lam=max(lam*0.98,lambda_min)
                ms=float(np.mean(surp_win)); ms_raw=core_buf.get_numpy().mean(axis=0)
                is_ev,rel_var,_=recorder.check(vol,ms,T_hat,ms_raw,lam,cycle,core_rate)
                rv_log.append(rel_var)
                if is_ev:
                    event_cycles.append(cycle); centroid_hist.append(centroid.position.copy())
                    if verbose:
                        sr_now=ms/(field_volatility+1e-10)
                        print(f"      ★ cycle={cycle:4d}  surp/vol={sr_now:.3f}  "
                              f"vol={vol:.3f}  core={core_rate:.3f}  "
                              f"cen={centroid.position.round(3)}")

        if (i+1)%(n_reset*tau_meas)==0 and i>0: kf.reset()

        dt_eff=dt_coherent if cur_sr<1.0 else dt_surprised
        F_ou=-k*(x-x0)
        J_arr=fp_flux(rho_true,F_free); J_at=float(np.interp(x,x_grid,J_arr))
        F_flux=lam*J_at/(abs(J_at)+1e-10)
        F_att=centroid.attraction_force_x(campo_log[-1] if campo_log else 0.0) \
              if centroid.is_ready else 0.0
        F_ext=F_att+F_ou

        def step(F_e):
            xi=np.sqrt(2*kBT*gamma)*rng.standard_normal()
            return float(np.clip(x+((F_flux+F_e)/gamma)*dt_eff+xi*np.sqrt(dt_eff)/gamma,x_min+0.1,x_max-0.1))

        x_s=step(F_ext); s_s=abs(x_s-kf.x_hat); bx=x_s; bs=s_s
        if rehearsal_on and n_candidates>1:
            for _ in range(n_candidates-1):
                xc=step(F_ext); sc=abs(xc-kf.x_hat)
                if sc<bs: bx=xc; bs=sc

        if k>0: rho_true=fp_step_ou(rho_true,F_free,k,x0)
        else:
            N_=len(rho_true); v=F_free/gamma; df=np.zeros(N_+1); ff=np.zeros(N_+1)
            for ii in range(1,N_):
                vf=0.5*(v[ii-1]+v[ii]); df[ii]=vf*rho_true[ii-1] if vf>=0 else vf*rho_true[ii]
                ff[ii]=D_diff*(rho_true[ii]-rho_true[ii-1])/dx
            rho_new=np.maximum(rho_true-(dt/dx)*np.diff(df-ff),0.0)
            norm=np.trapezoid(rho_new,x_grid); rho_true=rho_new/norm if norm>1e-12 else rho_new

        pf.add(bx,w=1./(sigma_m+0.1)); x=bx
        if i in snap_at:
            T_hat,_,_=core_buf.normalize(); lbl=snap_at[i]
            vol,area,hull=convex_hull_3d(T_hat); snap_va[lbl]=(vol,area,hull,T_hat.copy())

    T_hat_f,_,_=core_buf.normalize(); vol_f,area_f,hull_f=convex_hull_3d(T_hat_f)
    snap_va['final']=(vol_f,area_f,hull_f,T_hat_f.copy())
    crf=cw/max(total,1); arf=aw/max(total,1)

    return dict(
        k=k, sigma_m=sigma_m, mean_sr=float(np.mean(sr_log)),
        n_events=recorder.n,
        first_event_cycle=event_cycles[0] if event_cycles else None,
        core_rate=crf, anom_rate=arf, n_regime=regime.n,
        event_cycles=event_cycles, centroid_final=centroid.position,
        centroid_n=centroid.n, centroid_hist=centroid_hist,
        events=recorder.events, mean_vol_events=recorder.mean_vol(),
        mean_pca=recorder.mean_pca(), lam_final=lam,
        gate_thresh_final=gate.threshold,
        snap_va=snap_va, buf_final=T_hat_f,
        vol_log=np.array(vol_log), rv_log=np.array(rv_log),
        sr_log=np.array(sr_log), gate_log=np.array(gate_log),
        cr_log=np.array(cr_log), ar_log=np.array(ar_log),
    )


# ═════════════════════════════════════════════════════════════════════════════
# THREE RUNS
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print(f"RUN 1: k=0.0  sigma_m={sigma_m}  — free diffusion (gate integrity check)")
print("─" * 70)
r1 = run_one(seed=42, k=0.0, verbose=False)
gate_ok_1 = 0.30 <= r1['core_rate'] <= 0.70
print(f"  surp/vol={r1['mean_sr']:.4f}  events={r1['n_events']}  "
      f"core={r1['core_rate']:.3f}  anom={r1['anom_rate']:.3f}  "
      f"gate_th={r1['gate_thresh_final']:.4f}")
run1_status = 'PASS (zero events — gate intact)' if r1['n_events']==0 else 'FAIL (events in noise!)'
print(f"  STATUS: {run1_status}")

print()
print("─" * 70)
print(f"RUN 2: k=1.0  sigma_m={sigma_m}  — primary validation (expect coherence)")
print("─" * 70)
r2 = run_one(seed=42, k=1.0, verbose=True)
gate_ok_2 = 0.30 <= r2['core_rate'] <= 0.70
print(f"  surp/vol={r2['mean_sr']:.4f}  events={r2['n_events']}  "
      f"core={r2['core_rate']:.3f}  anom={r2['anom_rate']:.3f}  "
      f"gate_th={r2['gate_thresh_final']:.4f}")
run2_below_1 = r2['mean_sr'] < 1.0
run2_status  = ('PASS (events + ratio<1.0)' if r2['n_events']>0 and run2_below_1 else
                'PARTIAL (events, ratio>=1.0)' if r2['n_events']>0 else
                'FAIL (no events)')
print(f"  STATUS: {run2_status}")

print()
print("─" * 70)
print(f"RUN 3: k=0.1  sigma_m={sigma_m}  — threshold confirmation (predicted boundary)")
print("─" * 70)
r3 = run_one(seed=42, k=0.1, verbose=True)
gate_ok_3 = 0.30 <= r3['core_rate'] <= 0.70
print(f"  surp/vol={r3['mean_sr']:.4f}  events={r3['n_events']}  "
      f"core={r3['core_rate']:.3f}  anom={r3['anom_rate']:.3f}  "
      f"gate_th={r3['gate_thresh_final']:.4f}")
run3_status = ('PASS (events detected at predicted k)' if r3['n_events']>0 else
               'BELOW THRESHOLD (no events at k=0.1)')
print(f"  STATUS: {run3_status}")


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("VALIDATION TABLE")
print("=" * 70)
print(f"  {'Run':>3}  {'k':>5}  {'sigma_m':>7}  {'ratio':>7}  "
      f"{'events':>6}  {'1st_evt':>7}  {'core_rate':>9}  {'gate<1?':>7}")
print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*7}")
for idx,(r,lbl) in enumerate([(r1,'free'), (r2,'k=1.0'), (r3,'k=0.1')], 1):
    fe = str(r['first_event_cycle']) if r['first_event_cycle'] else '—'
    below = '✓' if r['mean_sr'] < 1.0 else '✗'
    print(f"  {idx:>3}  {r['k']:>5.2f}  {r['sigma_m']:>7.2f}  "
          f"{r['mean_sr']:>7.4f}  {r['n_events']:>6}  {fe:>7}  "
          f"{r['core_rate']:>9.3f}  {below:>7}")

print()
print(f"  Theoretical floor (sigma_m={sigma_m}): {ratio_floor_new:.4f}")
print(f"  Gate functional range 0.3–0.7: "
      f"{'ALL OK' if gate_ok_1 and gate_ok_2 and gate_ok_3 else 'CHECK RUNS'}")
print()
if r2['n_events'] > 0 and run2_below_1:
    ev0=r2['events'][0]; pca_r,_=r2['mean_pca']
    print("  ★ FIRST HONEST COHERENCE SIGNATURE")
    print(f"    k=1.0  sigma_m={sigma_m}")
    print(f"    Observed ratio         = {r2['mean_sr']:.4f}  (floor prediction: {ratio_floor_new:.4f})")
    print(f"    First event at cycle   = {r2['first_event_cycle']}")
    print(f"    Total coherence events = {r2['n_events']}")
    print(f"    Hull vol at first evt  = {ev0['vol']:.4f}")
    print(f"    PCA variance ratio     = {pca_r.round(3)}")
    print(f"    Centroid               = {r2['centroid_final'].round(4)}")
    print(f"    gate_threshold actual  = {r2['gate_thresh_final']:.4f}")
    print(f"    core_write_rate        = {r2['core_rate']:.3f}")
    print(f"    First evt surp/vol     = {ev0['surp_ratio']:.4f}")
    print(f"    First evt rel_var      = {ev0['rel_var']:.4f}")
    print(f"    First evt lambda       = {ev0['lambda']:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MANIFOLD WORLD — RUN 2 (primary validation)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering manifold world (Run 2)...", end='', flush=True)

BG='#07080f'; FG='#dde1ec'; GOLD='#f5c842'; TEAL='#3dd6c8'; VIOLET='#b87aff'
ROSE='#ff5f7e'; GREEN='#4ade80'; AMBER='#fb923c'; COH='#fde68a'
DIM='#1e2235'; WH='#ffffff'; REGIME='#f97316'

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':DIM,
    'text.color':FG,'axes.labelcolor':FG,'xtick.color':'#555870','ytick.color':'#555870'})

r = r2   # primary validation run
snaps = r['snap_va']

def draw_hull(ax,pts,hull,fc,ec,af=0.10,ae=0.28):
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

coh_state = ('COHERENT' if r['n_events']>0 and r['mean_sr']<1.0 else
             'EVENTS DETECTED' if r['n_events']>0 else 'EXPLORING')

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    f"SFE-05.9  ·  Sensor Calibration Validation  ·  Run 2: k=1.0  σ_m={sigma_m}\n"
    f"STATE: {coh_state}  |  events={r['n_events']}  |  ratio={r['mean_sr']:.4f}"
    f"  |  theory floor={ratio_floor_new:.4f}",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold')
gs = GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.34,top=0.966,bottom=0.06,left=0.04,right=0.97)

# ── Panel 0: Core buffer + hull + centroid ────────────────────────────────────
ax0=fig.add_subplot(gs[0,:2],projection='3d'); ax0.set_facecolor(BG)
pts_f=r['buf_final']; _,_,hf=convex_hull_3d(pts_f)
sc=ax0.scatter(pts_f[:,0],pts_f[:,1],pts_f[:,2],c=np.linspace(0,1,len(pts_f)),
               cmap='plasma',s=7,alpha=0.70,linewidths=0)
draw_hull(ax0,pts_f,hf,TEAL,TEAL,0.07,0.18)
for ev in r['events'][:3]: draw_hull(ax0,ev['pts'],ev['hull'],COH,COH,0.18,0.50)
cen=r['centroid_final']
if r['centroid_n']>=min_events_before_pull:
    ax0.scatter(*cen,color=WH,s=200,marker='*',zorder=15)
    cm=pts_f.mean(axis=0); ax0.quiver(*cm,*(cen-cm),color=GOLD,lw=2.5,alpha=0.85,arrow_length_ratio=0.20)
try:
    Xp,Yp=np.meshgrid([-2.5,2.5],[-2.5,2.5])
    ax0.plot_surface(Xp,Yp,np.zeros_like(Xp),alpha=0.05,color=GOLD,linewidth=0)
    ax0.text(2.2,2.2,0.08,"S-U",color=GOLD,fontsize=7,alpha=0.5)
except: pass
plt.colorbar(sc,ax=ax0,shrink=0.45,pad=0.02).set_label('Time',color=FG,fontsize=7)
style_3d(ax0,"Campo  |J(x)|","Alignment","Surprise  |innov|",
         f"Core Buffer  k=1.0  σ_m={sigma_m}  [★=centroid  gold=coherence hulls]\n"
         f"events={r['n_events']}  core_rate={r['core_rate']:.3f}  "
         f"surp/vol={r['mean_sr']:.3f}  λ={r['lam_final']:.3f}")
ax0.view_init(elev=22,azim=-52)

# ── Panel 1: Hull evolution ───────────────────────────────────────────────────
ax1=fig.add_subplot(gs[0,2],projection='3d'); ax1.set_facecolor(BG)
for lbl,col,af,ae in [('early',ROSE,0.12,0.35),('final',TEAL,0.07,0.20)]:
    e=snaps.get(lbl)
    if e:
        ax1.scatter(e[3][:,0],e[3][:,1],e[3][:,2],c=col,s=4,alpha=0.40,linewidths=0)
        draw_hull(ax1,e[3],e[2],col,col,af,ae)
if r['n_events']>0 and r['events'][0]['hull'] is not None:
    ev0=r['events'][0]; draw_hull(ax1,ev0['pts'],ev0['hull'],COH,COH,0.28,0.65)
if len(r['centroid_hist'])>=2:
    ch=np.array(r['centroid_hist']); ch_n=(ch-ch.mean(axis=0))/(ch.std(axis=0)+1e-8)
    ax1.plot(ch_n[:,0],ch_n[:,1],ch_n[:,2],color=WH,lw=1.8,alpha=0.70)
    ax1.scatter(ch_n[-1,0],ch_n[-1,1],ch_n[-1,2],color=WH,s=100,marker='*',zorder=12)
handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ROSE,ms=7,label=f"early V={snaps.get('early',(0,))[0]:.3f}"),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=TEAL,ms=7,label=f"final V={snaps.get('final',(0,))[0]:.3f}"),
    Line2D([0],[0],color=WH,lw=1.5,label='Centroid drift'),
]
if r['n_events']>0:
    handles.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COH,ms=10,
                          label=f"1st coherence V={r['mean_vol_events']:.3f}"))
ax1.legend(handles=handles,fontsize=6.5,facecolor='#0d0f18',edgecolor='none')
style_3d(ax1,"Campo","Alignment","Surprise",f"Hull Evolution  k=1.0  σ_m={sigma_m}")
ax1.view_init(elev=25,azim=-40)

# ── Panel 2: Volume + variance ────────────────────────────────────────────────
ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(BG)
va=r['vol_log']; rv=r['rv_log']
if len(va)>0:
    tv=np.arange(len(va))*COHERENCE_WIN; tr2=np.arange(len(rv))*COHERENCE_WIN
    ax2.fill_between(tv,0,va,alpha=0.18,color=VIOLET); ax2.plot(tv,va,color=VIOLET,lw=1.8,label='Hull volume')
    ax2.axhline(np.mean(va),color=TEAL,lw=0.7,ls=':',alpha=0.5)
    ax2b=ax2.twinx(); ax2b.set_facecolor(BG)
    if len(rv)>0:
        ax2b.plot(tr2,rv,color=GOLD,lw=1.5,ls='--',label='Rel. variance')
        ax2b.axhline(coherence_var_th,color=GREEN,lw=1.2,ls=':',label=f'coh_th={coherence_var_th}')
        ax2b.fill_between(tr2,0,rv,where=(rv<coherence_var_th),alpha=0.22,color=GREEN)
    ax2b.set_ylabel("Var(V)/mean(V)²",color=GOLD,fontsize=8); ax2b.tick_params(axis='y',labelcolor=GOLD,labelsize=7)
    for ec in r['event_cycles']: ax2.axvline(ec,color=COH,lw=1.0,alpha=0.65,ls='--')
    ax2.set_xlabel("Cycle",fontsize=9); ax2.set_ylabel("Hull Volume",color=VIOLET,fontsize=9)
    ax2.set_title(f"Volume + Variance  k=1.0  σ_m={sigma_m}\nGold dashed = coherence events",color=FG,fontsize=9)
    ax2.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
    ax2b.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper left')
    ax2.grid(True,alpha=0.12); ax2.tick_params(labelsize=7)

# ── Panel 3: Surprise ratio all 3 runs ───────────────────────────────────────
ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(BG)
cyc=np.arange(len(r2['sr_log']))
win=max(len(r2['sr_log'])//40,3)
for rr,col,lbl in [(r1,TEAL,f'Run1 k=0.0 (free)'),
                   (r2,ROSE,f'Run2 k=1.0 (primary)'),
                   (r3,AMBER,f'Run3 k=0.1 (thresh)')]:
    if len(rr['sr_log'])>0:
        sm=uniform_filter1d(rr['sr_log'],size=win)
        ax3.plot(cyc[:len(sm)],sm,lw=1.8,color=col,label=f"{lbl}  r={rr['mean_sr']:.3f}")
ax3.axhline(1.0,color=GREEN,lw=1.5,ls='--',label='Target = 1.0')
ax3.axhline(ratio_floor_new,color=WH,lw=1.0,ls=':',alpha=0.7,
            label=f'Theor. floor σ_m={sigma_m} = {ratio_floor_new:.3f}')
ax3.axhline(ratio_floor_old,color='#888',lw=0.8,ls=':',alpha=0.5,
            label=f'Old floor σ_m=0.90 = {ratio_floor_old:.3f}')
for ec in r2['event_cycles']: ax3.axvline(ec,color=COH,lw=0.7,alpha=0.45,ls='--')
ax3.set_xlabel("Cycle",fontsize=9); ax3.set_ylabel("Surprise / Field Volatility",fontsize=9)
ax3.set_title("Three Runs: Surprise Ratio Comparison\nGold dashed = coherence events (Run 2)",color=FG,fontsize=9)
ax3.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none')
ax3.grid(True,alpha=0.12); ax3.tick_params(labelsize=7); ax3.set_xlim(0,len(cyc))

# ── Panel 4: State space projection ──────────────────────────────────────────
ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(BG)
pts2=r['buf_final']
if len(pts2)>0:
    sc4=ax4.scatter(pts2[:,0],pts2[:,1],c=pts2[:,2],cmap='inferno',s=6,alpha=0.75,linewidths=0)
    cb4=plt.colorbar(sc4,ax=ax4,shrink=0.82); cb4.set_label('Surprise (z-score)',color=FG,fontsize=7); cb4.ax.yaxis.set_tick_params(labelsize=6)
    ax4.axhline(0,color=GOLD,lw=1.0,ls='--',alpha=0.55,label='S-U plane')
    ax4.fill_between([-3.5,3.5],-3.5,0,alpha=0.04,color=ROSE); ax4.fill_between([-3.5,3.5],0,3.5,alpha=0.04,color=GREEN)
    if r['centroid_n']>=min_events_before_pull:
        ax4.scatter([0],[0],color=WH,s=220,marker='*',zorder=15,label='Centroid')
        mx=pts2[:,0].mean(); my=pts2[:,1].mean()
        ax4.annotate('',xy=(0,0),xytext=(mx,my),arrowprops=dict(arrowstyle='->',color=GOLD,lw=2.0))
    if r['n_events']>0:
        pca_r,pca_ax=r['mean_pca']
        for ki,col in enumerate([ROSE,GREEN,AMBER]):
            vx=pca_ax[ki,0]*pca_r[ki]*2.0; vy=pca_ax[ki,1]*pca_r[ki]*2.0
            ax4.annotate('',xy=(vx,vy),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
    ax4.set_xlabel("Campo (z-score)",fontsize=9); ax4.set_ylabel("Alignment (z-score)",fontsize=9)
    ax4.set_title("Core Buffer State Space  (Run 2)\n★=centroid  Arrows=PCA axes",color=FG,fontsize=9)
    ax4.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none',loc='lower right')
    ax4.grid(True,alpha=0.12); ax4.tick_params(labelsize=7); ax4.set_xlim(-3.5,3.5); ax4.set_ylim(-3.5,3.5)

# ── Summary ───────────────────────────────────────────────────────────────────
pca_r,_=r2['mean_pca']
ev0_vol=r2['events'][0]['vol'] if r2['events'] else 0.0
ev0_sr =r2['events'][0]['surp_ratio'] if r2['events'] else 0.0
summary=[
    "SFE-05.9  VALIDATION","─"*30,"",
    f"KEY CHANGE: σ_m = {sigma_m} (was 0.90)","",
    "THEORETICAL FLOORS",
    f"  σ_m=0.90: ratio={ratio_floor_old:.4f}",
    f"  σ_m={sigma_m}:  ratio={ratio_floor_new:.4f}","",
    "─"*30,"VALIDATION TABLE",
    f"  Run  k    σ_m  ratio  evts gate_ok",
    f"  1  0.00  {sigma_m}  {r1['mean_sr']:.3f}  {r1['n_events']:>4}  {'OK' if gate_ok_1 else 'XX'}",
    f"  2  1.00  {sigma_m}  {r2['mean_sr']:.3f}  {r2['n_events']:>4}  {'OK' if gate_ok_2 else 'XX'}",
    f"  3  0.10  {sigma_m}  {r3['mean_sr']:.3f}  {r3['n_events']:>4}  {'OK' if gate_ok_3 else 'XX'}","",
    "─"*30,"RUN 2 STATUS",
    f"  {run2_status}","",
]
if r2['n_events']>0:
    summary += [
        "FIRST COHERENCE GEOMETRY",
        f"  1st cycle  = {r2['first_event_cycle']}",
        f"  hull vol   = {ev0_vol:.4f}",
        f"  surp_ratio = {ev0_sr:.4f}",
        f"  total evts = {r2['n_events']}",
        f"  centroid =",
        f"  [{r2['centroid_final'][0]:.4f},",
        f"   {r2['centroid_final'][1]:.4f},",
        f"   {r2['centroid_final'][2]:.4f}]",
        f"  PCA = [{pca_r[0]:.3f},",
        f"         {pca_r[1]:.3f},",
        f"         {pca_r[2]:.3f}]",
        f"  gate_th = {r2['gate_thresh_final']:.4f}",
        f"  core_rt = {r2['core_rate']:.3f}",
    ]
summary += ["","─"*30,f"STATE: {coh_state}"]

fig.text(0.698,0.456,"\n".join(summary),fontsize=6.8,fontfamily='monospace',color=FG,va='top',
         bbox=dict(boxstyle='round,pad=0.75',facecolor='#0a0c14',edgecolor=GOLD,linewidth=1.5,alpha=0.97))

plt.savefig(os.path.join(FIG_DIR,'sfe059_manifold.png'),dpi=150,bbox_inches='tight',facecolor=BG)
plt.show(); print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("SFE-05.9  Final Summary — Sensor Calibration Validation"); print("="*70)
print()
print("  THEORETICAL CALIBRATION")
print(f"    sigma_m (old)       = 0.90  →  floor ratio = {ratio_floor_old:.4f}")
print(f"    sigma_m (new)       = {sigma_m}  →  floor ratio = {ratio_floor_new:.4f}")
print(f"    field_volatility    = {field_volatility:.4f}")
print(f"    threshold sigma_m   = {np.sqrt(np.pi/2)*field_volatility:.4f}")
print()
print("  VALIDATION TABLE")
print(f"  {'Run':>3}  {'k':>5}  {'ratio':>7}  {'events':>6}  "
      f"{'first_evt':>9}  {'core_rate':>9}  {'gate_ok':>7}  {'status'}")
print("  " + "─"*70)
for idx,(rr,st) in enumerate([(r1,run1_status),(r2,run2_status),(r3,run3_status)], 1):
    fe=str(rr['first_event_cycle']) if rr['first_event_cycle'] else '—'
    go='OK' if 0.3<=rr['core_rate']<=0.7 else 'XX'
    print(f"  {idx:>3}  {rr['k']:>5.2f}  {rr['mean_sr']:>7.4f}  {rr['n_events']:>6}  "
          f"{fe:>9}  {rr['core_rate']:>9.3f}  {go:>7}  {st}")

print()
if r2['n_events']>0 and run2_below_1:
    ev0=r2['events'][0]; pca_r,_=r2['mean_pca']
    print("  ★ ★ ★  FIRST HONEST COHERENCE SIGNATURE  ★ ★ ★")
    print(f"    k=1.0  sigma_m={sigma_m}")
    print(f"    Observed ratio         = {r2['mean_sr']:.4f}")
    print(f"    Theory floor           = {ratio_floor_new:.4f}")
    print(f"    Margin above floor     = {r2['mean_sr'] - ratio_floor_new:+.4f}")
    print(f"    Total coherence events = {r2['n_events']}")
    print(f"    First event at cycle   = {r2['first_event_cycle']}")
    print(f"    Hull vol (first)       = {ev0['vol']:.4f}")
    print(f"    PCA variance ratio     = {pca_r.round(3)}")
    print(f"    Centroid               = {r2['centroid_final'].round(4)}")
    print(f"    gate_threshold actual  = {r2['gate_thresh_final']:.4f}")
    print(f"    core_write_rate        = {r2['core_rate']:.3f}")

print()
print("  LINEAGE (Surprise/Volatility)")
print("    05.3  free diffusion, no gating     = 2.054")
print("    05.4  gating + adaptive lambda      = 2.045")
print("    05.5  centroid pull + rehearsal     = 1.632")
print("    05.6  dual buffer + adaptive gate   = 1.635")
print("    05.7  drift sweep (unmodeled)       = 1.635 (flat)")
print("    05.8  OU sweep (floor hit)          = 1.607 (floor)")
print(f"    05.9  σ_m={sigma_m} + k=1.0 (validated)  = {r2['mean_sr']:.3f}")
print("="*70)
