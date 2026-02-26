# -*- coding: utf-8 -*-
"""
SFE-05.6 — Field Navigation System · Dual Buffer Architecture + Adaptive Gating
=================================================================================
Stochastic Field Engine, revision 05.6

EVOLUTION FROM SFE-05.5:
    SFE-05.5 achieved Surprise/Volatility = 1.632 via coherence centroid pull,
    mental rehearsal, and speed modulation. Dominant failure mode identified:
    gate_ratio = 0.786 (near-ungated). A single buffer accepting 78.6% of cycles
    cannot enforce epistemic selectivity — the agent learns from moments it
    barely survived, not only from moments it understood.

    SFE-05.6 separates learning from observation:

    1. DUAL BUFFER ARCHITECTURE
       Core buffer (N=512): writes when Surprise <= adaptive_threshold
         → The agent's learning memory. Drives centroid and coherence detection.
       Anomaly buffer (N=256): writes when Surprise > adaptive_threshold
         → Observation only. Never influences centroid or hull geometry.
         → Monitored for regime shift detection.

    2. ADAPTIVE GATE THRESHOLD
       threshold = std(surprise_history[-window:]) * gate_multiplier
       Based on rolling surprise std (same units as surprise itself).
       Floor = field_volatility (prevents over-gating in calm periods).
       Expected core_write_rate ≈ 0.34 (within target 0.3–0.7).

    3. REGIME SHIFT DETECTOR
       Monitors rolling anomaly write rate.
       Flags when rate > baseline * regime_shift_multiplier.
       Observational only — no behavioral changes.

    4. COHERENCE DETECTION GUARD
       Coherence events require core_write_rate >= min_core_rate (0.2).
       Surprise condition tightened to strict < 1.0.

STATE TENSOR: R^(N×3)  [unchanged]
    X — Campo / Y — Alignment / Z — Surprise

PARAMETERS (new in 05.6):
    gate_multiplier = 1.0  anomaly_buffer_size = 256
    regime_shift_multiplier = 2.0  min_core_rate = 0.20  gate_window = 50

ALL SFE-05.5 PARAMETERS UNCHANGED.
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
import os; FIG_DIR = '/tmp/sfe056_figs'; os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.6  —  Dual Buffer + Adaptive Gating + Regime Shift Detector")
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

# ─── Inherited parameters (SFE-05.5) ─────────────────────────────────────────
lambda_coup_init = 0.30; lambda_min = 0.05; lambda_max = 0.80
sigma_m = 0.90; sigma_memory = 1.20; coherence_var_th = 0.15; window_scale = 1.0
alpha_attract = 0.15; min_events_before_pull = 10
n_candidates = 5; rehearsal_on = True
dt_coherent = dt; dt_surprised = dt * 0.5
BUFFER_N = 512; COHERENCE_WIN = 40

# ─── New parameters (SFE-05.6) ───────────────────────────────────────────────
gate_multiplier         = 1.0
gate_window             = 50
anomaly_buffer_size     = 256
regime_shift_multiplier = 2.0
min_core_rate           = 0.20

# ─── Field volatility ─────────────────────────────────────────────────────────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

print(f"  [inherited]  lambda=[{lambda_min},{lambda_max}]  alpha_attract={alpha_attract}")
print(f"               n_candidates={n_candidates}  coh_var_th={coherence_var_th}")
print(f"\n  [new 05.6]   gate_multiplier={gate_multiplier}  gate_window={gate_window}")
print(f"               anomaly_buf={anomaly_buffer_size}  regime_mult={regime_shift_multiplier}")
print(f"               min_core_rate={min_core_rate}")
print(f"\n  Field volatility σ = {field_volatility:.4f}")
print(f"  Expected core_write_rate ≈ 0.34  (target 0.3–0.7)\n")


# ═════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER
# ═════════════════════════════════════════════════════════════════════════════
class CircularBuffer:
    def __init__(self, N, cols=3, name='buf'):
        self.N = N; self.cols = cols; self.name = name
        self.buf = torch.zeros(N, cols, dtype=torch.float32) if TORCH \
                   else np.zeros((N, cols), dtype=np.float32)
        self.filled = self.writes = 0

    def push(self, row):
        self.writes += 1
        if TORCH:
            self.buf = torch.roll(self.buf, shifts=1, dims=0)
            self.buf[0] = torch.tensor(row, dtype=torch.float32)
        else:
            self.buf = np.roll(self.buf, shift=1, axis=0)
            self.buf[0] = np.array(row, dtype=np.float32)
        self.filled = min(self.filled + 1, self.N)

    def get_numpy(self):
        return self.buf[:self.filled].numpy() if TORCH else self.buf[:self.filled]

    def normalize(self):
        data = self.get_numpy()
        if len(data) < 4:
            return data, np.zeros(self.cols), np.ones(self.cols)
        if TORCH:
            T = self.buf[:self.filled]; mu = T.mean(dim=0); sig = T.std(dim=0) + 1e-8
            return ((T - mu) / sig).numpy(), mu.numpy(), sig.numpy()
        mu = data.mean(axis=0); sig = data.std(axis=0) + 1e-8
        return (data - mu) / sig, mu, sig


# ═════════════════════════════════════════════════════════════════════════════
# ADAPTIVE GATE
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveGate:
    """
    threshold = max(std(surprise_history) * multiplier, field_volatility)
    core condition:    Surprise <= threshold
    anomaly condition: Surprise >  threshold
    """
    def __init__(self, multiplier, window, floor):
        self.mult = multiplier; self.window = window; self.floor = floor
        self._hist = []; self.threshold = floor

    def update(self, surprise):
        self._hist.append(surprise)
        if len(self._hist) > self.window: self._hist.pop(0)
        if len(self._hist) >= 4:
            self.threshold = max(float(np.std(self._hist)) * self.mult, self.floor)
        return self.threshold

    def core_gate(self, surprise):    return surprise <= self.threshold
    def anomaly_gate(self, surprise): return surprise >  self.threshold


# ═════════════════════════════════════════════════════════════════════════════
# REGIME SHIFT DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
class RegimeShiftDetector:
    def __init__(self, multiplier, window):
        self.mult = multiplier; self.window = window
        self._flags = []; self._rates = []; self.events = []

    def update(self, anomaly_written, cycle, centroid_pos):
        self._flags.append(int(anomaly_written))
        if len(self._flags) > self.window: self._flags.pop(0)
        if len(self._flags) < 10: return False
        rate = float(np.mean(self._flags)); self._rates.append(rate)
        if len(self._rates) < 10: return False
        baseline = float(np.mean(self._rates)) + 1e-8
        if rate > baseline * self.mult:
            self.events.append({'cycle': cycle, 'rate': rate, 'baseline': baseline,
                                'centroid': centroid_pos.copy() if centroid_pos is not None else np.zeros(3)})
            return True
        return False

    @property
    def n(self): return len(self.events)
    def rolling_rate(self): return float(np.mean(self._flags)) if self._flags else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS (inherited)
# ═════════════════════════════════════════════════════════════════════════════
def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma)**2)
    return rho / np.trapezoid(rho, x_grid)

def fp_flux(rho, F_arr):
    return (F_arr / gamma) * rho - D_diff * np.gradient(rho, x_grid)

def fp_step(rho, F_arr):
    N_ = len(rho); v = F_arr / gamma
    df = np.zeros(N_ + 1); ff = np.zeros(N_ + 1)
    for i in range(1, N_):
        vf = 0.5 * (v[i-1] + v[i])
        df[i] = vf * rho[i-1] if vf >= 0 else vf * rho[i]
        ff[i] = D_diff * (rho[i] - rho[i-1]) / dx
    rho_new = np.maximum(rho - (dt / dx) * np.diff(df - ff), 0.0)
    norm = np.trapezoid(rho_new, x_grid)
    return rho_new / norm if norm > 1e-12 else rho_new

def compute_alignment(x_pos, rho_true, rho_perc):
    g_t = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_p = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_t) < 1e-10 or abs(g_p) < 1e-10: return 0.0
    return float(np.clip(g_t * g_p / (abs(g_t) * abs(g_p)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN + PERCEIVED FIELD (inherited)
# ═════════════════════════════════════════════════════════════════════════════
class KalmanOD:
    def __init__(self):
        self.x_hat = 0.0; self.P = 2*kBT/gamma*tau_meas*dt; self.Q = 2*kBT/gamma*dt
    def predict_n(self, n): self.P += n * self.Q
    def update(self, z):
        P_prior = self.P; K = self.P / (self.P + sigma_m**2)
        self.x_hat += K * (z - self.x_hat); self.P *= (1 - K)
        return P_prior, abs(z - self.x_hat + K*(z-self.x_hat))   # |innov|
    def update(self, z):
        P_prior = self.P; K = self.P / (self.P + sigma_m**2)
        innov = z - self.x_hat
        self.x_hat += K * innov; self.P *= (1 - K)
        return P_prior, abs(innov)
    def I_gain(self, P): return max(0.5 * np.log2(1 + P / sigma_m**2), 0.0)
    def reset(self): self.x_hat = 0.0; self.P = 2*kBT/gamma*tau_meas*dt

class PerceivedField:
    def __init__(self, max_n=300): self.samples=[]; self.weights=[]; self.max_n=max_n
    def add(self, xp, w=1.0):
        self.samples.append(xp); self.weights.append(w)
        if len(self.samples) > self.max_n: self.samples.pop(0); self.weights.pop(0)
    def get_rho(self):
        if len(self.samples) < 2: return gaussian_rho(0.0, 1.0)
        rho = np.zeros(Nx); w_tot = sum(self.weights)
        for xp, w in zip(self.samples, self.weights):
            rho += (w/w_tot) * np.exp(-0.5*((x_grid-xp)/sigma_memory)**2)
        norm = np.trapezoid(rho, x_grid)
        return rho/norm if norm > 1e-12 else rho


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY (inherited)
# ═════════════════════════════════════════════════════════════════════════════
def convex_hull_3d(pts):
    if len(pts) < 5: return 0.0, 0.0, None
    try: h = ConvexHull(pts); return h.volume, h.area, h
    except QhullError: return 0.0, 0.0, None

def pca_axes(pts):
    if len(pts) < 4: return np.array([1.,0.,0.]), np.eye(3)
    c = pts - pts.mean(axis=0); cov = np.cov(c.T)
    if cov.ndim < 2: return np.array([1.,0.,0.]), np.eye(3)
    vals, vecs = np.linalg.eigh(cov); order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.); vecs = vecs[:, order]
    return vals / (vals.sum() + 1e-12), vecs.T


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE CENTROID (inherited)
# ═════════════════════════════════════════════════════════════════════════════
class CoherenceCentroid:
    def __init__(self): self.centroid = np.zeros(3); self.n = 0
    def update(self, pos):
        self.n += 1; self.centroid = ((self.n-1)*self.centroid + pos) / self.n
        return self.centroid.copy()
    def attraction_force_x(self, campo):
        return alpha_attract * (self.centroid[0] - campo) if self.n > 0 else 0.0
    @property
    def is_ready(self): return self.n >= min_events_before_pull
    @property
    def position(self): return self.centroid.copy()


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE RECORDER (updated: strict < 1.0 + core_rate guard)
# ═════════════════════════════════════════════════════════════════════════════
class CoherenceRecorder:
    def __init__(self, var_th, field_vol, window, centroid):
        self.var_th=var_th; self.fvol=field_vol; self.window=max(window,4)
        self.centroid=centroid; self.events=[]; self._vols=[]

    def check(self, vol, mean_surp, pts, mean_state, lam, cycle, core_rate):
        self._vols.append(vol)
        if len(self._vols) > self.window: self._vols.pop(0)
        if len(self._vols) < 4: return False, 0.0, 0.0
        mean_v = np.mean(self._vols) + 1e-8
        rel_var = float(np.var(self._vols)) / (mean_v**2)
        surp_rat = mean_surp / (self.fvol + 1e-10)

        is_event = (rel_var < self.var_th) and (surp_rat < 1.0) and (core_rate >= min_core_rate)
        if is_event and len(pts) >= 5:
            vol_e, area_e, hull_e = convex_hull_3d(pts)
            ratio_e, axes_e = pca_axes(pts)
            self.centroid.update(mean_state)
            self.events.append({'cycle':cycle,'vol':vol_e,'area':area_e,'hull':hull_e,
                                'pts':pts.copy(),'pca_ratio':ratio_e,'pca_axes':axes_e,
                                'rel_var':rel_var,'surp_ratio':surp_rat,'lambda':lam,
                                'centroid':self.centroid.position,'core_rate':core_rate})
        return is_event, rel_var, surp_rat

    @property
    def n(self): return len(self.events)
    def mean_vol(self): return float(np.mean([e['vol'] for e in self.events])) if self.events else 0.0
    def mean_pca(self):
        if not self.events: return np.ones(3)/3, np.eye(3)
        return (np.array([e['pca_ratio'] for e in self.events]).mean(axis=0),
                np.array([e['pca_axes']  for e in self.events]).mean(axis=0))


# ═════════════════════════════════════════════════════════════════════════════
# ACTIVE INFERENCE STEP (inherited from 05.5)
# ═════════════════════════════════════════════════════════════════════════════
def base_langevin(x, rho_true, F_free, F_extra, rng, lam, dt_eff):
    J_arr = fp_flux(rho_true, F_free); J_at = float(np.interp(x, x_grid, J_arr))
    F_flux = lam * J_at / (abs(J_at) + 1e-10)
    xi = np.sqrt(2*kBT*gamma) * rng.standard_normal()
    dx_ = ((F_flux + F_extra)/gamma)*dt_eff + xi*np.sqrt(dt_eff)/gamma
    return float(np.clip(x + dx_, x_min+0.1, x_max-0.1)), abs(J_at)

def active_inference_step(x, rho_true, F_free, rng, lam, x_hat, centroid_obj, campo, sr):
    dt_eff = dt_coherent if sr < 1.0 else dt_surprised
    F_att  = centroid_obj.attraction_force_x(campo) if centroid_obj.is_ready else 0.0
    x_s, c_s = base_langevin(x, rho_true, F_free, F_att, rng, lam, dt_eff)
    s_s = abs(x_s - x_hat)
    if not rehearsal_on or n_candidates <= 1:
        return x_s, c_s, False
    bx=x_s; bs=s_s; bc=c_s; acc=False
    for _ in range(n_candidates - 1):
        xc, cc = base_langevin(x, rho_true, F_free, F_att, rng, lam, dt_eff)
        sc = abs(xc - x_hat)
        if sc < bs: bx=xc; bs=sc; bc=cc; acc=True
    return bx, bc, acc


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Dual Buffer + Adaptive Gating + Regime Shift Detection")

def run_navigation(seed=42, N=N):
    rng = np.random.default_rng(seed)
    kf  = KalmanOD(); pf = PerceivedField()

    core_buf    = CircularBuffer(BUFFER_N,          cols=3, name='core')
    anomaly_buf = CircularBuffer(anomaly_buffer_size, cols=3, name='anomaly')
    gate        = AdaptiveGate(gate_multiplier, gate_window, floor=field_volatility)
    centroid    = CoherenceCentroid()
    window_len  = max(int(N_cycles * 0.5 * window_scale), 8)
    recorder    = CoherenceRecorder(coherence_var_th, field_volatility, window_len, centroid)
    regime      = RegimeShiftDetector(regime_shift_multiplier, max(gate_window, 30))

    rho_true = gaussian_rho(0.0, 2.0); F_free = np.zeros(Nx)
    x = 0.0; pf.add(x); lam = lambda_coup_init

    # Logs
    surprise_log=[]; campo_log=[]; lambda_log=[]; sr_log=[]; gate_log=[]
    vol_log=[]; area_log=[]; rv_log=[]; cr_log=[]; ar_log=[]
    event_cycles=[]; centroid_hist=[]; rehearsal_log=[]; regime_cycles=[]
    snap_at = {N//8:'early', N//2:'mid', 7*N//8:'late'}; snap_va={}
    vol_win=[]; surp_win=[]; total=0; cw=0; aw=0; cycle=0
    n_reset=max(N_cycles//8,1); I_cum=0.0; cur_sr=2.0

    for i in range(N):
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas = x + sigma_m * rng.standard_normal()
            P_prior, surp = kf.update(x_meas)
            I_cum += kBT * Landauer * kf.I_gain(P_prior)
            rho_perc = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)
            J_arr = fp_flux(rho_true, F_free)
            campo = float(abs(np.interp(x, x_grid, J_arr)))
            cur_sr = surp / (field_volatility + 1e-10)

            # Adaptive gate
            g_th = gate.update(surp); gate_log.append(g_th); total += 1

            # Dual buffer writes
            write_core = gate.core_gate(surp)
            write_anom = gate.anomaly_gate(surp)
            if write_core: core_buf.push([campo, alignment, surp]); cw += 1
            anom_written = False
            if write_anom: anomaly_buf.push([campo, alignment, surp]); aw += 1; anom_written=True

            surprise_log.append(surp); campo_log.append(campo)
            lambda_log.append(lam); sr_log.append(cur_sr)

            surp_win.append(surp)
            if len(surp_win) > COHERENCE_WIN: surp_win.pop(0)

            # Regime shift
            is_reg = regime.update(anom_written, cycle, centroid.position if centroid.n>0 else None)
            if is_reg:
                regime_cycles.append(cycle)
                print(f"    ⚠ REGIME_SHIFT  cycle={cycle}  anom_rate={regime.rolling_rate():.3f}")

            cycle += 1

            # Hull evaluation
            if cycle % COHERENCE_WIN == 0 and core_buf.filled >= 5:
                T_hat, _, _ = core_buf.normalize()
                vol, area, _ = convex_hull_3d(T_hat)
                vol_log.append(vol); area_log.append(area)
                vol_win.append(vol)
                if len(vol_win) > window_len: vol_win.pop(0)

                core_rate = cw / max(total, 1); anom_rate = aw / max(total, 1)
                cr_log.append(core_rate); ar_log.append(anom_rate)

                # Tension dynamics
                if len(vol_win) >= 3:
                    vm=np.mean(vol_win); vs=np.std(vol_win)+1e-8; z=(vol-vm)/vs
                    if z > 1.0:    lam = min(lam*1.05, lambda_max)
                    elif z < -0.5: lam = max(lam*0.98, lambda_min)

                # Coherence check
                ms_recent = float(np.mean(surp_win))
                ms_raw    = core_buf.get_numpy().mean(axis=0)
                is_ev, rel_var, surp_rat = recorder.check(
                    vol, ms_recent, T_hat, ms_raw, lam, cycle, core_rate)
                rv_log.append(rel_var)

                if is_ev:
                    event_cycles.append(cycle); centroid_hist.append(centroid.position.copy())
                    print(f"    ★ cycle={cycle:4d}  vol={vol:.3f}  surp/vol={surp_rat:.3f}  "
                          f"core={core_rate:.3f}  gate_th={g_th:.4f}  "
                          f"cen={centroid.position.round(3)}")

        if (i+1) % (n_reset*tau_meas) == 0 and i > 0: kf.reset()

        x_new, _, reh = active_inference_step(
            x, rho_true, F_free, rng, lam, kf.x_hat, centroid,
            campo_log[-1] if campo_log else 0.0, cur_sr)
        rehearsal_log.append(int(reh))
        rho_true = fp_step(rho_true, F_free); pf.add(x_new, w=1./(sigma_m+0.1)); x=x_new

        if i in snap_at:
            T_hat,_,_ = core_buf.normalize(); lbl=snap_at[i]
            vol,area,hull = convex_hull_3d(T_hat); snap_va[lbl]=(vol,area,hull,T_hat.copy())

    T_hat_f,_,_ = core_buf.normalize(); vol_f,area_f,hull_f=convex_hull_3d(T_hat_f)
    snap_va['final']=(vol_f,area_f,hull_f,T_hat_f.copy())
    crf=cw/max(total,1); arf=aw/max(total,1)

    return dict(
        buf_final=T_hat_f, snap_va=snap_va,
        vol_log=np.array(vol_log), area_log=np.array(area_log),
        rv_log=np.array(rv_log), lambda_log=np.array(lambda_log),
        surprise_log=np.array(surprise_log), campo_log=np.array(campo_log),
        sr_log=np.array(sr_log), gate_log=np.array(gate_log),
        cr_log=np.array(cr_log), ar_log=np.array(ar_log),
        events=recorder.events, n_events=recorder.n,
        mean_vol_events=recorder.mean_vol(), mean_pca=recorder.mean_pca(),
        core_rate_f=crf, anom_rate_f=arf,
        regime_events=regime.events, n_regime=regime.n, regime_cycles=regime_cycles,
        event_cycles=event_cycles, centroid_hist=centroid_hist,
        centroid_final=centroid.position, centroid_n=centroid.n,
        rehearsal_rate=float(np.mean(rehearsal_log)) if rehearsal_log else 0.0,
        window_len=window_len, lam_final=lam, n_cycles=cycle,
        gate_thresh_f=gate.threshold, vol_final=vol_f,
    )


print("  Running...", end='', flush=True)
r = run_navigation(seed=42)
print(f"\n  done. {r['n_cycles']} cycles.")

snaps     = r['snap_va']
mean_surp = float(np.mean(r['surprise_log']))
surp_rat  = mean_surp / field_volatility
core_ok   = 0.30 <= r['core_rate_f'] <= 0.70
improvement = 1.6304 - surp_rat

print(f"\n  Hull volumes (core buffer):")
for lbl in ['early','mid','late','final']:
    e = snaps.get(lbl)
    if e: print(f"    {lbl:8s}  vol={e[0]:.4f}  area={e[1]:.4f}")

print(f"\n  SURPRISE / VOLATILITY")
print(f"    SFE-05.5 baseline = 1.6304")
print(f"    SFE-05.6 result   = {surp_rat:.4f}  ({improvement:+.4f})")
print(f"    target            = < 1.0")
print(f"\n  DUAL BUFFER GATE")
print(f"    core_write_rate   = {r['core_rate_f']:.3f}  "
      f"({'OK' if core_ok else 'out of range'})")
print(f"    anomaly_rate      = {r['anom_rate_f']:.3f}")
print(f"    gate_thresh_final = {r['gate_thresh_f']:.4f}")
print(f"\n  COHERENCE EVENTS  = {r['n_events']}")
print(f"  REGIME SHIFTS     = {r['n_regime']}")
print(f"  CENTROID          = {r['centroid_final'].round(4)}")
print(f"  REHEARSAL RATE    = {r['rehearsal_rate']:.3f}")
print(f"  LAMBDA FINAL      = {r['lam_final']:.4f}")

coh_state = ('COHERENT'        if r['n_events'] > 0 and surp_rat < 1.0 else
             'EVENTS DETECTED' if r['n_events'] > 0 else 'EXPLORING')
print(f"\n  STATE: {coh_state}  |  Gate OK: {core_ok}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Manifold World
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering...", end='', flush=True)

BG='#07080f'; FG='#dde1ec'; GOLD='#f5c842'; TEAL='#3dd6c8'; VIOLET='#b87aff'
ROSE='#ff5f7e'; GREEN='#4ade80'; AMBER='#fb923c'; COH='#fde68a'
REGIME='#f97316'; DIM='#1e2235'; WH='#ffffff'

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':DIM,
    'text.color':FG,'axes.labelcolor':FG,'xtick.color':'#555870','ytick.color':'#555870'})

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle("SFE-05.6  ·  Dual Buffer Architecture + Adaptive Gating\n"
             "Core Buffer (learning)  ·  Anomaly Buffer (regime detection)  ·  Adaptive Gate",
             fontsize=13, color=GOLD, y=0.999, fontweight='bold')
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34, top=0.966, bottom=0.06, left=0.04, right=0.97)

def draw_hull(ax, pts, hull, fc, ec, af=0.10, ae=0.28):
    if hull is None or pts is None: return
    step = max(len(hull.simplices)//300, 1)
    poly = Poly3DCollection([pts[s] for s in hull.simplices[::step]], alpha=af, linewidth=0.3)
    poly.set_facecolor(fc); poly.set_edgecolor(ec); ax.add_collection3d(poly)

def style_3d(ax, xl, yl, zl, title):
    ax.set_xlabel(xl,fontsize=8,labelpad=2); ax.set_ylabel(yl,fontsize=8,labelpad=2)
    ax.set_zlabel(zl,fontsize=8,labelpad=2); ax.set_title(title,color=FG,fontsize=9,pad=4)
    ax.tick_params(labelsize=6)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False; p.set_edgecolor(DIM)
    ax.grid(True, alpha=0.10)


# ── PANEL 0: Core buffer point cloud + hull + centroid ───────────────────────
ax0 = fig.add_subplot(gs[0,:2], projection='3d'); ax0.set_facecolor(BG)
pts_f = r['buf_final']; _, _, hull_f = convex_hull_3d(pts_f)
sc = ax0.scatter(pts_f[:,0],pts_f[:,1],pts_f[:,2],
                 c=np.linspace(0,1,len(pts_f)),cmap='plasma',s=7,alpha=0.70,linewidths=0)
draw_hull(ax0, pts_f, hull_f, TEAL, TEAL, 0.07, 0.18)
for ev in r['events'][:2]: draw_hull(ax0, ev['pts'], ev['hull'], COH, COH, 0.20, 0.55)
cen = r['centroid_final']
if r['centroid_n'] >= min_events_before_pull:
    ax0.scatter(*cen, color=WH, s=200, marker='*', zorder=15)
    cm = pts_f.mean(axis=0)
    ax0.quiver(*cm, *(cen-cm), color=GOLD, lw=2.5, alpha=0.85, arrow_length_ratio=0.20)
for _ in r['regime_cycles'][:4]:
    ax0.scatter(0, 0, 2.5, color=REGIME, s=100, marker='^', zorder=14, alpha=0.80)
try:
    Xp,Yp = np.meshgrid([-2.5,2.5],[-2.5,2.5])
    ax0.plot_surface(Xp,Yp,np.zeros_like(Xp),alpha=0.05,color=GOLD,linewidth=0)
    ax0.text(2.2,2.2,0.08,"S-U",color=GOLD,fontsize=7,alpha=0.5)
except: pass
plt.colorbar(sc,ax=ax0,shrink=0.45,pad=0.02).set_label('Time',color=FG,fontsize=7)
style_3d(ax0,"Campo  |J(x)|","Alignment","Surprise  |innov|",
         f"Core Buffer  [★=centroid  ▲=regime shift]\n"
         f"core_rate={r['core_rate_f']:.3f}  gate_th={r['gate_thresh_f']:.4f}  "
         f"events={r['n_events']}  rehearsal={r['rehearsal_rate']:.3f}")
ax0.view_init(elev=22, azim=-52)


# ── PANEL 1: Hull evolution + centroid drift ──────────────────────────────────
ax1 = fig.add_subplot(gs[0,2], projection='3d'); ax1.set_facecolor(BG)
for lbl,col,af,ae in [('early',ROSE,0.12,0.35),('final',TEAL,0.07,0.20)]:
    e = snaps.get(lbl)
    if e:
        ax1.scatter(e[3][:,0],e[3][:,1],e[3][:,2],c=col,s=4,alpha=0.40,linewidths=0)
        draw_hull(ax1,e[3],e[2],col,col,af,ae)
if r['n_events'] > 0 and r['events'][0]['hull'] is not None:
    ev0=r['events'][0]; draw_hull(ax1,ev0['pts'],ev0['hull'],COH,COH,0.26,0.65)
if len(r['centroid_hist']) >= 2:
    ch=np.array(r['centroid_hist']); ch_n=(ch-ch.mean(axis=0))/(ch.std(axis=0)+1e-8)
    ax1.plot(ch_n[:,0],ch_n[:,1],ch_n[:,2],color=WH,lw=1.5,alpha=0.65)
    ax1.scatter(ch_n[-1,0],ch_n[-1,1],ch_n[-1,2],color=WH,s=80,marker='*',zorder=12)
handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ROSE,ms=7,label=f"early  V={snaps.get('early',(0,))[0]:.3f}"),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=TEAL,ms=7,label=f"final  V={snaps.get('final',(0,))[0]:.3f}"),
    Line2D([0],[0],color=WH,lw=1.5,label='Centroid drift'),
]
if r['n_events']>0: handles.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COH,ms=10,label=f"coh. V={r['mean_vol_events']:.3f}"))
if r['n_regime']>0: handles.append(Line2D([0],[0],marker='^',color='w',markerfacecolor=REGIME,ms=8,label='Regime shift'))
ax1.legend(handles=handles,fontsize=6.5,facecolor='#0d0f18',edgecolor='none')
style_3d(ax1,"Campo","Alignment","Surprise","Hull Evolution + Centroid Drift\nOrange ▲ = regime shift")
ax1.view_init(elev=25,azim=-40)


# ── PANEL 2: Volume + variance + event/regime markers ────────────────────────
ax2 = fig.add_subplot(gs[1,0]); ax2.set_facecolor(BG)
va=r['vol_log']; rv=r['rv_log']
if len(va)>0:
    tv=np.arange(len(va))*COHERENCE_WIN; tr=np.arange(len(rv))*COHERENCE_WIN
    ax2.fill_between(tv,0,va,alpha=0.18,color=VIOLET); ax2.plot(tv,va,color=VIOLET,lw=1.8,label='Hull volume')
    ax2.axhline(np.mean(va),color=TEAL,lw=0.7,ls=':',alpha=0.5)
    ax2b=ax2.twinx(); ax2b.set_facecolor(BG)
    if len(rv)>0:
        ax2b.plot(tr,rv,color=GOLD,lw=1.5,ls='--',label='Rel. variance')
        ax2b.axhline(coherence_var_th,color=GREEN,lw=1.2,ls=':',label=f'threshold={coherence_var_th}')
        ax2b.fill_between(tr,0,rv,where=(rv<coherence_var_th),alpha=0.22,color=GREEN)
    ax2b.set_ylabel("Var(V)/mean(V)²",color=GOLD,fontsize=8); ax2b.tick_params(axis='y',labelcolor=GOLD,labelsize=7)
    for ec in r['event_cycles']: ax2.axvline(ec,color=COH,lw=0.8,alpha=0.5,ls='--')
    for rc in r['regime_cycles']: ax2.axvline(rc,color=REGIME,lw=1.3,alpha=0.7,ls='-.')
    ax2.set_xlabel("Cycle",fontsize=9); ax2.set_ylabel("Hull Volume",color=VIOLET,fontsize=9)
    ax2.set_title("Core Buffer Volume + Variance\nGold=coh. events  Orange=regime shifts",color=FG,fontsize=8)
    ax2.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
    ax2b.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper left')
    ax2.grid(True,alpha=0.12); ax2.tick_params(labelsize=7)


# ── PANEL 3: Adaptive gate + write rates + surprise ratio ─────────────────────
ax3 = fig.add_subplot(gs[1,1]); ax3.set_facecolor(BG)
sr=r['sr_log']; gt=r['gate_log']; cr=r['cr_log']; ar=r['ar_log']; cyc=np.arange(len(sr))
if len(sr)>0:
    win=max(len(sr)//40,3); sr_s=uniform_filter1d(sr,size=win); gt_s=uniform_filter1d(gt,size=win)
    ax3.fill_between(cyc,0,sr,alpha=0.09,color=ROSE)
    ax3.plot(cyc,sr_s,color=ROSE,lw=2.0,label='Surprise/Volatility')
    ax3.axhline(1.6304,color=ROSE,lw=0.8,ls=':',alpha=0.5,label='05.5 baseline=1.630')
    ax3.axhline(1.0,color=GREEN,lw=1.3,ls='--',label='Target = 1.0')
    gt_norm=gt_s/(field_volatility+1e-8)
    ax3.plot(cyc[:len(gt_norm)],gt_norm,color=AMBER,lw=1.4,ls=':',alpha=0.85,label='Gate threshold / σ_field')
    ax3b=ax3.twinx(); ax3b.set_facecolor(BG)
    if len(cr)>0:
        t_r=np.arange(len(cr))*COHERENCE_WIN
        ax3b.plot(t_r,cr,color=GREEN,lw=1.8,label='Core rate (target 0.3–0.7)')
        ax3b.plot(t_r,ar,color=REGIME,lw=1.5,ls='--',label='Anomaly rate')
        ax3b.axhspan(0.30,0.70,alpha=0.06,color=GREEN)
        ax3b.axhline(min_core_rate,color=VIOLET,lw=0.8,ls=':',label=f'min_core={min_core_rate}')
        ax3b.set_ylim(0,1.1)
    ax3b.set_ylabel("Write Rate",fontsize=8); ax3b.tick_params(axis='y',labelsize=7)
    ax3.set_xlabel("Cycle",fontsize=9); ax3.set_ylabel("Surprise / Field Volatility",fontsize=9)
    ax3.set_title("Adaptive Gate + Dual Write Rates\nGreen band = target core_rate zone",color=FG,fontsize=9)
    ax3.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
    ax3b.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper left')
    ax3.grid(True,alpha=0.12); ax3.tick_params(labelsize=7); ax3.set_xlim(0,len(sr))


# ── PANEL 4: Campo × Alignment + PCA ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1,2]); ax4.set_facecolor(BG)
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
        for k,col in enumerate([ROSE,GREEN,AMBER]):
            vx=pca_ax[k,0]*pca_r[k]*2.0; vy=pca_ax[k,1]*pca_r[k]*2.0
            ax4.annotate('',xy=(vx,vy),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
    ax4.set_xlabel("Campo (z-score)",fontsize=9); ax4.set_ylabel("Alignment (z-score)",fontsize=9)
    ax4.set_title("Core Buffer State Space\n★=centroid  Arrows=PCA axes",color=FG,fontsize=9)
    ax4.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none',loc='lower right')
    ax4.grid(True,alpha=0.12); ax4.tick_params(labelsize=7); ax4.set_xlim(-3.5,3.5); ax4.set_ylim(-3.5,3.5)


# ── Summary ───────────────────────────────────────────────────────────────────
pca_r,_=r['mean_pca']; core_status='OK' if core_ok else ('OVER-GATED' if r['core_rate_f']<0.3 else 'UNDER-GATED')
summary=[
    "SFE-05.6  DUAL BUFFER","─"*30,"",
    "[1] Dual buffer (core/anomaly)",
    "[2] Adaptive gate (surp σ)","[3] Regime shift detector",
    "[4] Core rate guard on coherence","[5] Active inference inherited","","─"*30,
    "GATE",
    f"  multiplier  = {gate_multiplier}",f"  threshold   = {r['gate_thresh_f']:.4f}",
    f"  floor       = {field_volatility:.4f}  (σ_field)","",
    "DUAL BUFFER",
    f"  core_rate   = {r['core_rate_f']:.3f}  [{core_status}]",
    f"  anom_rate   = {r['anom_rate_f']:.3f}","",
    "COHERENCE",
    f"  N events    = {r['n_events']}",f"  Mean vol    = {r['mean_vol_events']:.4f}",
    f"  PCA         = [{pca_r[0]:.3f},{pca_r[1]:.3f},{pca_r[2]:.3f}]","",
    "CENTROID",
    f"  n used      = {r['centroid_n']}",
    f"  [{r['centroid_final'][0]:.4f},",f"   {r['centroid_final'][1]:.4f},",f"   {r['centroid_final'][2]:.4f}]","",
    f"REGIME SHIFTS = {r['n_regime']}","REHEARSAL",f"  rate = {r['rehearsal_rate']:.4f}","",
    "SURPRISE",
    f"  ratio = {surp_rat:.4f}",f"  vs 05.5 = {improvement:+.4f}",f"  target  = < 1.0","","─"*30,
    f"STATE: {coh_state}",
]
fig.text(0.698,0.456,"\n".join(summary),fontsize=6.8,fontfamily='monospace',color=FG,va='top',
         bbox=dict(boxstyle='round,pad=0.75',facecolor='#0a0c14',edgecolor=GOLD,linewidth=1.3,alpha=0.97))

plt.savefig(os.path.join(FIG_DIR,'sfe056_dual_buffer.png'),dpi=150,bbox_inches='tight',facecolor=BG)
plt.show(); print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("SFE-05.6  Final Summary"); print("="*70)
print(f"\n  DUAL BUFFER")
print(f"    core_write_rate  = {r['core_rate_f']:.3f}  (target 0.3–0.7 → {core_status})")
print(f"    anomaly_rate     = {r['anom_rate_f']:.3f}")
print(f"    gate_threshold   = {r['gate_thresh_f']:.4f}  (adaptive from surprise σ)")
print(f"\n  REGIME SHIFTS detected = {r['n_regime']}")
for ev in r['regime_events'][:3]: print(f"    cycle={ev['cycle']}  rate={ev['rate']:.3f}")
print(f"\n  SURPRISE / VOLATILITY")
print(f"    05.5 → 05.6  :  1.6304 → {surp_rat:.4f}  ({improvement:+.4f})")
print(f"    target       :  < 1.0")
print(f"\n  COHERENCE EVENTS = {r['n_events']}")
if r['n_events']>0:
    print(f"  CENTROID         = {r['centroid_final'].round(4)}")
    print(f"  PCA ratio        = {pca_r.round(3)}")
print(f"  LAMBDA FINAL     = {r['lam_final']:.4f}")
print(f"  REHEARSAL RATE   = {r['rehearsal_rate']:.4f}")
print(f"  STATE            = {coh_state}")
print(f"\n  EVOLUTION FROM SFE-05.5")
print(f"    Single buffer     →  Dual buffer (core / anomaly)")
print(f"    Static gate 2.5   →  Adaptive gate = std(surprise) × multiplier")
print(f"    No regime monitor →  RegimeShiftDetector (observational only)")
print(f"    Unchecked gating  →  core_rate guard on coherence detection")
print("="*70)
