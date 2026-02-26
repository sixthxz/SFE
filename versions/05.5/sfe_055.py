# -*- coding: utf-8 -*-
"""
SFE-05.5 — Field Navigation System · Active Inference via Coherence Centroid
==============================================================================
Stochastic Field Engine, revision 05.5

EVOLUTION FROM SFE-05.4:
    SFE-05.4 detected 45 coherence events, established adaptive lambda,
    selective memory gating, and PCA of the emergent attractor geometry.
    Surprise/Volatility remained at 2.054 — the agent was EVENTS DETECTED
    but not yet COHERENT. Lambda saturated near its ceiling. The attractor
    was recorded but not yet used to guide navigation.

    SFE-05.5 closes the loop: the recorded attractor becomes a force.

    1. COHERENCE CENTROID (dynamic attractor)
       The mean position in state space across all logged coherence events.
       Computed incrementally as events arrive — never fixed manually.
       centroid = mean([event.state_position for event in coherence_events])
       The centroid drifts as the field evolves and new events are recorded.
       It is the only prescribed direction the agent moves toward — and only
       because the field itself produced those events.

    2. ATTRACTOR FORCE VECTOR
       At each cycle, compute displacement from current state to centroid:
           delta = coherence_centroid - current_state  (3D vector)
           F_attract = alpha_attract * delta[position_axis]  (projects to x)
       Applied only when n_coherence_events >= min_events_before_pull (10).
       Before that threshold the centroid is not statistically meaningful.

    3. COUNTERFACTUAL STEP (mental rehearsal)
       Before committing each move, simulate k=5 candidate Langevin steps.
       For each candidate, predict the resulting surprise:
           predicted_surprise = |candidate_x - x_hat|
       Select the candidate with minimum predicted surprise.
       If that candidate is better than the standard step: take it.
       Otherwise: take the standard step.
       One-step lookahead only. No global optimization. No guarantee.
       The counterfactual accept rate measures how often rehearsal redirects.

    4. SPEED MODULATION
       dt_effective depends on current Surprise/Volatility ratio:
           ratio < 1.0: dt_effective = dt        (coherent — move confidently)
           ratio >= 1.0: dt_effective = dt * 0.5  (surprised — slow down)
       Continuous modulation, not a hard stop.

PARAMETERS (all tunable):
    alpha_attract          = 0.15   attraction strength to coherence centroid
    min_events_before_pull = 10     minimum events before centroid is applied
    n_candidates           = 5      candidate moves for counterfactual step
    rehearsal_on           = True   toggle counterfactual testing
    dt_coherent            = dt     step size when coherent
    dt_surprised           = dt * 0.5  step size when surprise/volatility > 1.0
    (all SFE-05.4 parameters unchanged)

STATE TENSOR: R^(N×3)
    X — Campo:     |J(x,t)|  local probability flux magnitude
    Y — Alignment: cos(grad_true, grad_perceived)
    Z — Surprise:  |x_true - x_hat|  Kalman innovation residual

COHERENCE DEFINITION (unchanged):
    Var(volume) / mean(volume)^2 < coherence_var_th
    AND mean_surprise / field_volatility < gate_threshold simultaneously.
    Geometry at that moment = coherence signature. Emergent. Not prescribed.
"""

import numpy as np

try:
    import torch
    TORCH = True
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import os
FIG_DIR = '/tmp/sfe055_figs'
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.5  —  Active Inference via Coherence Centroid")
print("=" * 70)
print(f"  Backend: {'PyTorch ' + torch.__version__ if TORCH else 'NumPy'}")
print()


# ─── Units ───────────────────────────────────────────────────────────────────
kBT      = 1.0
gamma    = 1.0
D_diff   = kBT / gamma
Landauer = np.log(2)

# ─── Simulation ──────────────────────────────────────────────────────────────
dt       = 0.01
N        = 20000
tau_meas = 10
N_cycles = N // tau_meas

# ─── Field grid ──────────────────────────────────────────────────────────────
x_min, x_max = -8.0, 8.0
Nx           = 400
x_grid       = np.linspace(x_min, x_max, Nx)
dx           = x_grid[1] - x_grid[0]

# ─── SFE-05.4 parameters (unchanged) ─────────────────────────────────────────
lambda_coup_init = 0.30
lambda_min       = 0.05
lambda_max       = 0.80
sigma_m          = 0.90
sigma_memory     = 1.20
window_scale     = 1.0
gate_threshold   = 2.5
coherence_var_th = 0.15

# ─── SFE-05.5 new parameters ─────────────────────────────────────────────────
alpha_attract          = 0.15   # attraction strength toward coherence centroid
min_events_before_pull = 10     # minimum coherence events before centroid activates
n_candidates           = 5      # candidate moves for counterfactual selection
rehearsal_on           = True   # toggle mental rehearsal
dt_coherent            = dt         # step size in coherent regime
dt_surprised           = dt * 0.5   # step size in high-surprise regime

# ─── Buffer / geometry ───────────────────────────────────────────────────────
BUFFER_N      = 512
COHERENCE_WIN = 40

# ─── Field volatility ────────────────────────────────────────────────────────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

print(f"  [inherited]  lambda_init={lambda_coup_init}  gate_th={gate_threshold}")
print(f"               sigma_m={sigma_m}  coh_var_th={coherence_var_th}")
print()
print(f"  [new 05.5]   alpha_attract={alpha_attract}")
print(f"               min_events_before_pull={min_events_before_pull}")
print(f"               n_candidates={n_candidates}  rehearsal_on={rehearsal_on}")
print(f"               dt_coherent={dt_coherent}  dt_surprised={dt_surprised}")
print()
print(f"  Field volatility σ = {field_volatility:.4f}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER (with selective gating — inherited)
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    def __init__(self, N, cols=3):
        self.N    = N
        self.cols = cols
        if TORCH:
            self.buf = torch.zeros(N, cols, dtype=torch.float32)
        else:
            self.buf = np.zeros((N, cols), dtype=np.float32)
        self.filled = self.writes = self.total_cycles = 0

    def push(self, row, gate=1):
        self.total_cycles += 1
        if gate == 0:
            return
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
            T = self.buf[:self.filled]
            mu = T.mean(dim=0); sigma = T.std(dim=0) + 1e-8
            return ((T - mu) / sigma).numpy(), mu.numpy(), sigma.numpy()
        mu = data.mean(axis=0); sigma = data.std(axis=0) + 1e-8
        return (data - mu) / sigma, mu, sigma

    @property
    def gate_ratio(self):
        return self.writes / max(self.total_cycles, 1)


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS (inherited)
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
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
    if abs(g_t) < 1e-10 or abs(g_p) < 1e-10:
        return 0.0
    return float(np.clip(g_t * g_p / (abs(g_t) * abs(g_p)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN + PERCEIVED FIELD (inherited)
# ═════════════════════════════════════════════════════════════════════════════

class KalmanOD:
    def __init__(self):
        self.x_hat = 0.0
        self.P     = 2 * kBT / gamma * tau_meas * dt
        self.Q     = 2 * kBT / gamma * dt

    def predict_n(self, n):
        self.P += n * self.Q

    def update(self, z):
        P_prior    = self.P
        K          = self.P / (self.P + sigma_m**2)
        innov      = z - self.x_hat
        self.x_hat += K * innov
        self.P     *= (1 - K)
        return P_prior, abs(innov)

    def I_gain(self, P_prior):
        return max(0.5 * np.log2(1.0 + P_prior / sigma_m**2), 0.0)

    def reset(self):
        self.x_hat = 0.0
        self.P     = 2 * kBT / gamma * tau_meas * dt


class PerceivedField:
    def __init__(self, max_samples=300):
        self.samples = []; self.weights = []; self.max_n = max_samples

    def add(self, xp, w=1.0):
        self.samples.append(xp); self.weights.append(w)
        if len(self.samples) > self.max_n:
            self.samples.pop(0); self.weights.pop(0)

    def get_rho(self):
        if len(self.samples) < 2:
            return gaussian_rho(0.0, 1.0)
        rho = np.zeros(Nx); w_tot = sum(self.weights)
        for xp, w in zip(self.samples, self.weights):
            rho += (w / w_tot) * np.exp(-0.5 * ((x_grid - xp) / sigma_memory)**2)
        norm = np.trapezoid(rho, x_grid)
        return rho / norm if norm > 1e-12 else rho


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY: Convex Hull 3D + PCA (inherited)
# ═════════════════════════════════════════════════════════════════════════════

def convex_hull_3d(pts):
    if len(pts) < 5:
        return 0.0, 0.0, None
    try:
        h = ConvexHull(pts)
        return h.volume, h.area, h
    except QhullError:
        return 0.0, 0.0, None

def pca_axes(pts):
    if len(pts) < 4:
        return np.array([1.0, 0.0, 0.0]), np.eye(3)
    c = pts - pts.mean(axis=0)
    cov = np.cov(c.T)
    if cov.ndim < 2:
        return np.array([1.0, 0.0, 0.0]), np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0); vecs = vecs[:, order]
    return vals / (vals.sum() + 1e-12), vecs.T


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE CENTROID (new in 05.5)
# ═════════════════════════════════════════════════════════════════════════════

class CoherenceCentroid:
    """
    Tracks the running mean position in (campo, alignment, surprise) state space
    across all logged coherence events.

    Incremental update:
        centroid_new = (n_prev * centroid_prev + new_position) / n_current

    The centroid drifts as the field produces new events.
    It is never fixed manually — it is entirely derived from the field's history.
    """

    def __init__(self):
        self.centroid = np.zeros(3)   # [campo, alignment, surprise]
        self.n        = 0

    def update(self, new_position):
        """
        new_position: shape (3,) — the mean state vector at the coherence event.
        Returns updated centroid.
        """
        self.n += 1
        self.centroid = ((self.n - 1) * self.centroid + new_position) / self.n
        return self.centroid.copy()

    def attraction_force_x(self, current_campo, current_x):
        """
        Project the centroid attraction to the particle's position axis.
        The centroid lives in (campo, alignment, surprise) space.
        We use the campo axis (X axis of tensor) as a proxy for position:
        delta_campo = centroid[0] - current_campo
        F_attract = alpha_attract * delta_campo
        (campo = |J(x)| — higher flux magnitude → particle near steeper gradients)
        Returns scalar force in position space.
        """
        if self.n == 0:
            return 0.0
        delta_campo = self.centroid[0] - current_campo
        return alpha_attract * delta_campo

    @property
    def is_ready(self):
        return self.n >= min_events_before_pull

    @property
    def position(self):
        return self.centroid.copy()


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE RECORDER (inherited from 05.4, extended to update centroid)
# ═════════════════════════════════════════════════════════════════════════════

class CoherenceRecorder:
    def __init__(self, var_th, field_vol, gate_th, window, centroid):
        self.var_th  = var_th
        self.fvol    = field_vol
        self.gate_th = gate_th
        self.window  = max(window, 4)
        self.centroid = centroid   # shared CoherenceCentroid object
        self.events  = []
        self._vols   = []

    def check(self, vol, mean_surp, pts, mean_state, lam, cycle):
        """
        mean_state: shape (3,) — mean of current normalized point cloud.
        Records event and updates centroid if conditions met.
        """
        self._vols.append(vol)
        if len(self._vols) > self.window:
            self._vols.pop(0)
        if len(self._vols) < 4:
            return False, 0.0, 0.0

        mean_v   = np.mean(self._vols) + 1e-8
        rel_var  = float(np.var(self._vols)) / (mean_v**2)
        surp_rat = mean_surp / (self.fvol + 1e-10)

        is_event = (rel_var < self.var_th) and (surp_rat < self.gate_th)
        if is_event and len(pts) >= 5:
            vol_e, area_e, hull_e = convex_hull_3d(pts)
            ratio_e, axes_e       = pca_axes(pts)
            # Update centroid incrementally
            self.centroid.update(mean_state)
            self.events.append({
                'cycle':      cycle,
                'vol':        vol_e,
                'area':       area_e,
                'hull':       hull_e,
                'pts':        pts.copy(),
                'pca_ratio':  ratio_e,
                'pca_axes':   axes_e,
                'rel_var':    rel_var,
                'surp_ratio': surp_rat,
                'lambda':     lam,
                'centroid':   self.centroid.position,
            })
        return is_event, rel_var, surp_rat

    @property
    def n(self):
        return len(self.events)

    def mean_vol(self):
        return float(np.mean([e['vol'] for e in self.events])) if self.events else 0.0

    def mean_pca(self):
        if not self.events:
            return np.ones(3) / 3, np.eye(3)
        ratios = np.array([e['pca_ratio'] for e in self.events])
        axes   = np.array([e['pca_axes']  for e in self.events])
        return ratios.mean(axis=0), axes.mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# ACTIVE INFERENCE STEP (new in 05.5)
# ═════════════════════════════════════════════════════════════════════════════

def base_langevin(x, rho_true, F_free, F_extra, rng, lam, dt_eff):
    """
    Single Langevin step with extra force and effective dt.
    F_extra: additional deterministic force (attraction + flux coupling).
    Returns (x_new, campo_magnitude).
    """
    J_arr  = fp_flux(rho_true, F_free)
    J_at   = float(np.interp(x, x_grid, J_arr))
    F_flux = lam * J_at / (abs(J_at) + 1e-10)
    F_tot  = F_flux + F_extra

    xi    = np.sqrt(2 * kBT * gamma) * rng.standard_normal()
    dx_   = (F_tot / gamma) * dt_eff + xi * np.sqrt(dt_eff) / gamma
    return float(np.clip(x + dx_, x_min + 0.1, x_max - 0.1)), abs(J_at)


def active_inference_step(x, rho_true, F_free, rng, lam,
                          x_hat, centroid_obj, current_campo, surp_ratio):
    """
    Full active inference step combining:
      1. Speed modulation (dt depends on coherence state)
      2. Attractor force (if centroid is statistically ready)
      3. Counterfactual selection (mental rehearsal over k candidates)

    Returns (x_new, campo_mag, rehearsal_accepted: bool).
    """

    # 1. Speed modulation
    dt_eff = dt_coherent if surp_ratio < 1.0 else dt_surprised

    # 2. Attractor force
    F_attract = centroid_obj.attraction_force_x(current_campo, x) \
                if centroid_obj.is_ready else 0.0

    # 3. Standard step (baseline for comparison)
    # We need a separate rng draw for standard — save state
    rng_state = rng.bit_generator.state
    x_standard, campo_std = base_langevin(
        x, rho_true, F_free, F_attract, rng, lam, dt_eff
    )
    surp_standard = abs(x_standard - x_hat)

    if not rehearsal_on or n_candidates <= 1:
        return x_standard, campo_std, False

    # 4. Counterfactual candidates (each gets an independent noise draw)
    best_x    = x_standard
    best_surp = surp_standard
    best_campo = campo_std
    accepted  = False

    for _ in range(n_candidates - 1):   # already have 1 from standard step
        x_cand, campo_cand = base_langevin(
            x, rho_true, F_free, F_attract, rng, lam, dt_eff
        )
        surp_cand = abs(x_cand - x_hat)
        if surp_cand < best_surp:
            best_x     = x_cand
            best_surp  = surp_cand
            best_campo = campo_cand
            accepted   = True

    return best_x, best_campo, accepted


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Active Inference via Coherence Centroid")

def run_navigation(seed=42, N=N):
    rng       = np.random.default_rng(seed)
    kf        = KalmanOD()
    pf        = PerceivedField(max_samples=300)
    cbuf      = CircularBuffer(BUFFER_N, cols=3)
    centroid  = CoherenceCentroid()

    sigma_field_max = field_volatility * 2.0
    window_len = max(int(N_cycles * field_volatility / sigma_field_max * window_scale), 8)

    recorder = CoherenceRecorder(
        var_th    = coherence_var_th,
        field_vol = field_volatility,
        gate_th   = gate_threshold,
        window    = window_len,
        centroid  = centroid,
    )

    rho_true = gaussian_rho(0.0, 2.0)
    F_free   = np.zeros(Nx)
    x        = 0.0
    pf.add(x)
    lam      = lambda_coup_init

    # Logs
    surprise_log      = []
    alignment_log     = []
    campo_log         = []
    lambda_log        = []
    surp_ratio_log    = []
    volume_log        = []
    area_log          = []
    rel_var_log       = []
    event_cycles      = []
    centroid_history  = []   # centroid position at each coherence event
    rehearsal_log     = []   # 1=rehearsal changed move, 0=standard was best
    dt_eff_log        = []   # effective dt used each Langevin step

    snap_at       = {N // 8: 'early', N // 2: 'mid', 7 * N // 8: 'late'}
    snap_vol_area = {}

    vol_window  = []
    surp_window = []
    cycle       = 0
    n_reset     = max(N_cycles // 8, 1)
    I_cum       = 0.0

    # Current surprise ratio (updated each measurement cycle)
    current_surp_ratio = 2.0   # start assuming high surprise

    for i in range(N):

        # ── Measurement cycle ──────────────────────────────────────────────
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas        = x + sigma_m * rng.standard_normal()
            P_prior, surp = kf.update(x_meas)
            I_step        = kf.I_gain(P_prior)
            I_cum        += kBT * Landauer * I_step

            rho_perc  = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)

            J_arr = fp_flux(rho_true, F_free)
            campo = float(abs(np.interp(x, x_grid, J_arr)))

            current_surp_ratio = surp / (field_volatility + 1e-10)

            # ── Selective Memory Gating ───────────────────────────────────
            gate = 1 if surp < field_volatility * gate_threshold else 0
            cbuf.push([campo, alignment, surp], gate=gate)

            surprise_log.append(surp)
            alignment_log.append(alignment)
            campo_log.append(campo)
            lambda_log.append(lam)
            surp_ratio_log.append(current_surp_ratio)

            surp_window.append(surp)
            if len(surp_window) > COHERENCE_WIN:
                surp_window.pop(0)

            cycle += 1

            # ── Hull evaluation every COHERENCE_WIN cycles ────────────────
            if cycle % COHERENCE_WIN == 0 and cbuf.filled >= 5:
                T_hat, mu_, _ = cbuf.normalize()
                vol, area, _  = convex_hull_3d(T_hat)
                volume_log.append(vol)
                area_log.append(area)
                vol_window.append(vol)
                if len(vol_window) > window_len:
                    vol_window.pop(0)

                # Manifold tension dynamics (inherited)
                if len(vol_window) >= 3:
                    v_mean = np.mean(vol_window)
                    v_std  = np.std(vol_window) + 1e-8
                    z      = (vol - v_mean) / v_std
                    if z > 1.0:
                        lam = min(lam * 1.05, lambda_max)
                    elif z < -0.5:
                        lam = max(lam * 0.98, lambda_min)

                # Coherence event check
                mean_surp_recent = float(np.mean(surp_window))
                mean_state_raw   = cbuf.get_numpy().mean(axis=0)  # raw mean position

                is_ev, rel_var, surp_rat = recorder.check(
                    vol, mean_surp_recent, T_hat, mean_state_raw, lam, cycle
                )
                rel_var_log.append(rel_var)

                if is_ev:
                    event_cycles.append(cycle)
                    centroid_history.append(centroid.position.copy())
                    print(f"    ★ cycle={cycle:4d}  vol={vol:.3f}  "
                          f"surp/vol={surp_rat:.3f}  "
                          f"centroid={centroid.position.round(3)}  "
                          f"λ={lam:.3f}")

        # ── Kalman reset ──────────────────────────────────────────────────
        if (i + 1) % (n_reset * tau_meas) == 0 and i > 0:
            kf.reset()

        # ── Active Inference Step ─────────────────────────────────────────
        x_new, campo_step, rehearsal_took = active_inference_step(
            x         = x,
            rho_true  = rho_true,
            F_free    = F_free,
            rng       = rng,
            lam       = lam,
            x_hat     = kf.x_hat,
            centroid_obj = centroid,
            current_campo = campo_log[-1] if campo_log else 0.0,
            surp_ratio    = current_surp_ratio,
        )
        rehearsal_log.append(int(rehearsal_took))
        dt_used = dt_coherent if current_surp_ratio < 1.0 else dt_surprised
        dt_eff_log.append(dt_used)

        rho_true  = fp_step(rho_true, F_free)
        pf.add(x_new, w=1.0 / (sigma_m + 0.1))
        x = x_new

        if i in snap_at:
            T_hat, _, _ = cbuf.normalize()
            lbl = snap_at[i]
            vol, area, hull = convex_hull_3d(T_hat)
            snap_vol_area[lbl] = (vol, area, hull, T_hat.copy())

    # Final
    T_hat_f, _, _ = cbuf.normalize()
    vol_f, area_f, hull_f = convex_hull_3d(T_hat_f)
    snap_vol_area['final'] = (vol_f, area_f, hull_f, T_hat_f.copy())

    vol_arr     = np.array(volume_log)
    rel_var_arr = np.array(rel_var_log)
    rehearsal_arr = np.array(rehearsal_log)

    return dict(
        buf_final         = T_hat_f,
        snap_vol_area     = snap_vol_area,
        volume_log        = vol_arr,
        area_log          = np.array(area_log),
        rel_var_log       = rel_var_arr,
        lambda_log        = np.array(lambda_log),
        surprise_log      = np.array(surprise_log),
        alignment_log     = np.array(alignment_log),
        campo_log         = np.array(campo_log),
        surp_ratio_log    = np.array(surp_ratio_log),
        events            = recorder.events,
        n_events          = recorder.n,
        mean_vol_events   = recorder.mean_vol(),
        mean_pca          = recorder.mean_pca(),
        gate_ratio        = cbuf.gate_ratio,
        event_cycles      = event_cycles,
        centroid_history  = centroid_history,
        centroid_final    = centroid.position,
        centroid_n        = centroid.n,
        rehearsal_arr     = rehearsal_arr,
        rehearsal_rate    = float(rehearsal_arr.mean()) if len(rehearsal_arr) > 0 else 0.0,
        window_len        = window_len,
        lam_final         = lam,
        n_cycles          = cycle,
        vol_final         = vol_f,
        area_final        = area_f,
        I_cum             = I_cum,
    )


print("  Running...", end='', flush=True)
r = run_navigation(seed=42)
print(f"\n  done. {r['n_cycles']} cycles.")

snaps   = r['snap_vol_area']
mean_surp = float(np.mean(r['surprise_log']))
surp_rat  = mean_surp / field_volatility

print(f"\n  Hull volumes:")
for lbl in ['early', 'mid', 'late', 'final']:
    e = snaps.get(lbl)
    if e:
        print(f"    {lbl:8s}  vol={e[0]:.4f}  area={e[1]:.4f}")

print(f"\n  Mean surprise         = {mean_surp:.4f}")
print(f"  Field volatility σ    = {field_volatility:.4f}")
print(f"  Surprise / volatility = {surp_rat:.4f}  (SFE-05.4 baseline: 2.054)")
print(f"  Gate ratio            = {r['gate_ratio']:.3f}")
print(f"  Lambda final          = {r['lam_final']:.4f}")
print(f"  Coherence events      = {r['n_events']}")
print(f"  Centroid (final)      = {r['centroid_final'].round(4)}")
print(f"  Rehearsal accept rate = {r['rehearsal_rate']:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Manifold World
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering...", end='', flush=True)

BG     = '#07080f'; FG     = '#dde1ec'; GOLD   = '#f5c842'
TEAL   = '#3dd6c8'; VIOLET = '#b87aff'; ROSE   = '#ff5f7e'
GREEN  = '#4ade80'; AMBER  = '#fb923c'; COH    = '#fde68a'
DIM    = '#1e2235'; CENTROID_COL = '#ffffff'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor':   DIM, 'text.color':    FG,
    'axes.labelcolor':  FG,  'xtick.color':   '#555870',
    'ytick.color':      '#555870',
})

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    "SFE-05.5  ·  Active Inference via Coherence Centroid\n"
    "Centroid Pull  ·  Mental Rehearsal  ·  Speed Modulation",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold'
)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34,
              top=0.966, bottom=0.06, left=0.04, right=0.97)


def draw_hull(ax, pts, hull, fc, ec, af=0.10, ae=0.28):
    if hull is None or pts is None:
        return
    step  = max(len(hull.simplices) // 300, 1)
    verts = [pts[s] for s in hull.simplices[::step]]
    poly  = Poly3DCollection(verts, alpha=af, linewidth=0.3)
    poly.set_facecolor(fc); poly.set_edgecolor(ec)
    ax.add_collection3d(poly)

def style_3d(ax, xl, yl, zl, title):
    ax.set_xlabel(xl, fontsize=8, labelpad=2)
    ax.set_ylabel(yl, fontsize=8, labelpad=2)
    ax.set_zlabel(zl, fontsize=8, labelpad=2)
    ax.set_title(title, color=FG, fontsize=9, pad=4)
    ax.tick_params(labelsize=6)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor(DIM)
    ax.grid(True, alpha=0.10)


# ── PANEL 0: Final point cloud + hull + centroid ─────────────────────────────
ax0 = fig.add_subplot(gs[0, :2], projection='3d')
ax0.set_facecolor(BG)

pts_f  = r['buf_final']
_, _, hull_f = convex_hull_3d(pts_f)
n_pts  = len(pts_f)
t_col  = np.linspace(0, 1, n_pts)

sc = ax0.scatter(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2],
                 c=t_col, cmap='plasma', s=7, alpha=0.70, linewidths=0)
draw_hull(ax0, pts_f, hull_f, TEAL, TEAL, af=0.07, ae=0.18)

# Coherence event hulls (first 2)
for ev in r['events'][:2]:
    draw_hull(ax0, ev['pts'], ev['hull'], COH, COH, af=0.20, ae=0.55)

# Coherence centroid — white star, with arrow from cloud mean
cen = r['centroid_final']
if r['centroid_n'] >= min_events_before_pull:
    ax0.scatter(*cen, color=CENTROID_COL, s=180, marker='*',
                zorder=15, alpha=1.0, label='Coherence centroid')
    cloud_mean = pts_f.mean(axis=0)
    ax0.quiver(*cloud_mean,
               *(cen - cloud_mean),
               color=GOLD, lw=2.5, alpha=0.85,
               arrow_length_ratio=0.20)

# PCA axes
if r['n_events'] > 0:
    pca_r, pca_ax = r['mean_pca']
    ctr = pts_f.mean(axis=0)
    for k, col in enumerate([ROSE, GREEN, AMBER]):
        ax0.quiver(*ctr, *(pca_ax[k] * pca_r[k] * 2.5),
                   color=col, lw=1.8, alpha=0.80,
                   arrow_length_ratio=0.22)

# S-U plane at z=0
try:
    Xp, Yp = np.meshgrid([-2.5, 2.5], [-2.5, 2.5])
    ax0.plot_surface(Xp, Yp, np.zeros_like(Xp),
                     alpha=0.05, color=GOLD, linewidth=0)
    ax0.text(2.2, 2.2, 0.08, "S-U", color=GOLD, fontsize=7, alpha=0.5)
except Exception:
    pass

plt.colorbar(sc, ax=ax0, shrink=0.45, pad=0.02).set_label(
    'Time (oldest→newest)', color=FG, fontsize=7)

style_3d(ax0,
         "Campo  |J(x)|", "Alignment", "Surprise  |innov|",
         f"Point Cloud + Coherence Hull [gold]  |  Centroid [★ white]\n"
         f"Gate={r['gate_ratio']:.3f}  λ_final={r['lam_final']:.3f}  "
         f"Events={r['n_events']}  Rehearsal={r['rehearsal_rate']:.3f}")
ax0.view_init(elev=22, azim=-52)


# ── PANEL 1: Hull evolution + centroid drift ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2], projection='3d')
ax1.set_facecolor(BG)

for lbl, col, af, ae in [('early', ROSE, 0.12, 0.35), ('final', TEAL, 0.07, 0.20)]:
    e = snaps.get(lbl)
    if e:
        ax1.scatter(e[3][:, 0], e[3][:, 1], e[3][:, 2],
                    c=col, s=4, alpha=0.40, linewidths=0)
        draw_hull(ax1, e[3], e[2], col, col, af, ae)

if r['n_events'] > 0:
    ev0 = r['events'][0]
    if ev0['hull'] is not None:
        draw_hull(ax1, ev0['pts'], ev0['hull'], COH, COH, af=0.26, ae=0.65)

# Centroid drift path
if len(r['centroid_history']) >= 2:
    ch = np.array(r['centroid_history'])
    # Normalize roughly to z-score space for display
    ch_n = (ch - ch.mean(axis=0)) / (ch.std(axis=0) + 1e-8)
    ax1.plot(ch_n[:, 0], ch_n[:, 1], ch_n[:, 2],
             color=CENTROID_COL, lw=1.5, alpha=0.7, label='Centroid drift')
    ax1.scatter(ch_n[-1, 0], ch_n[-1, 1], ch_n[-1, 2],
                color=CENTROID_COL, s=80, marker='*', zorder=12)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ROSE, ms=7,
           label=f"early  V={snaps.get('early', (0,))[0]:.3f}"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, ms=7,
           label=f"final  V={snaps.get('final', (0,))[0]:.3f}"),
    Line2D([0], [0], marker='*', color='w', markerfacecolor=COH, ms=10,
           label=f"coh. sig. V={r['mean_vol_events']:.3f}"),
    Line2D([0], [0], color=CENTROID_COL, lw=1.5,
           label='Centroid drift'),
]
ax1.legend(handles=handles, fontsize=6.5, facecolor='#0d0f18', edgecolor='none')
style_3d(ax1, "Campo", "Alignment", "Surprise",
         "Hull Evolution + Centroid Drift Path\nWhite = centroid trajectory")
ax1.view_init(elev=25, azim=-40)


# ── PANEL 2: Volume + relative variance + event markers ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG)

vol_arr = r['volume_log']
rv_arr  = r['rel_var_log']

if len(vol_arr) > 0:
    t_vol = np.arange(len(vol_arr)) * COHERENCE_WIN
    t_rv  = np.arange(len(rv_arr))  * COHERENCE_WIN

    ax2.fill_between(t_vol, 0, vol_arr, alpha=0.18, color=VIOLET)
    ax2.plot(t_vol, vol_arr, color=VIOLET, lw=1.8, label='Hull volume')
    ax2.axhline(np.mean(vol_arr), color=TEAL, lw=0.8, ls=':', alpha=0.6)

    ax2b = ax2.twinx(); ax2b.set_facecolor(BG)
    if len(rv_arr) > 0:
        ax2b.plot(t_rv, rv_arr, color=GOLD, lw=1.5, ls='--', label='Rel. variance')
        ax2b.axhline(coherence_var_th, color=GREEN, lw=1.2, ls=':',
                     label=f'threshold = {coherence_var_th}')
        ax2b.fill_between(t_rv, 0, rv_arr,
                          where=(rv_arr < coherence_var_th),
                          alpha=0.25, color=GREEN)
    ax2b.set_ylabel("Var(V)/mean(V)²", color=GOLD, fontsize=8)
    ax2b.tick_params(axis='y', labelcolor=GOLD, labelsize=7)

    for ec in r['event_cycles']:
        ax2.axvline(ec, color=COH, lw=0.8, alpha=0.5, ls='--')

    # Mark when centroid became active
    if r['centroid_n'] >= min_events_before_pull:
        pull_cycle = r['event_cycles'][min_events_before_pull - 1] if len(r['event_cycles']) >= min_events_before_pull else 0
        ax2.axvline(pull_cycle, color=CENTROID_COL, lw=1.5, ls='-.',
                    label=f'Centroid active (event {min_events_before_pull})')

    ax2.set_xlabel("Cycle", fontsize=9)
    ax2.set_ylabel("Hull Volume", color=VIOLET, fontsize=9)
    ax2.set_title("Volume + Variance + Event Timeline\nWhite dashed = centroid pull activated",
                  color=FG, fontsize=8)
    ax2.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none', loc='upper right')
    ax2b.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none', loc='upper left')
    ax2.grid(True, alpha=0.12); ax2.tick_params(labelsize=7)


# ── PANEL 3: Adaptive λ + Surprise/Volatility + rehearsal rate ───────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)

sr_log  = r['surp_ratio_log']
lam_log = r['lambda_log']
cyc_t   = np.arange(len(sr_log))

if len(sr_log) > 0:
    win   = max(len(sr_log) // 40, 3)
    sr_sm = uniform_filter1d(sr_log, size=win)
    lm_sm = uniform_filter1d(lam_log, size=win)

    ax3.fill_between(cyc_t, 0, sr_log, alpha=0.10, color=ROSE)
    ax3.plot(cyc_t, sr_sm, color=ROSE, lw=2.0, label='Surprise/Volatility')
    ax3.axhline(gate_threshold, color=AMBER, lw=1.0, ls='--',
                label=f'gate_threshold = {gate_threshold}')
    ax3.axhline(1.0, color=GREEN, lw=1.0, ls=':',
                label='Coherence target (ratio=1.0)')
    ax3.fill_between(cyc_t, 0, sr_sm, where=(sr_sm < gate_threshold),
                     alpha=0.12, color=AMBER)

    ax3b = ax3.twinx(); ax3b.set_facecolor(BG)
    ax3b.plot(cyc_t, lm_sm, color=TEAL, lw=1.8, ls='--', label='λ (adaptive)')
    ax3b.set_ylabel("λ_coupling", color=TEAL, fontsize=8)
    ax3b.tick_params(axis='y', labelcolor=TEAL, labelsize=7)
    ax3b.set_ylim(0, lambda_max * 1.25)

    # Rehearsal rate overlay (rolling window) — sampled at measurement cycles only
    if len(r['rehearsal_arr']) > 0:
        # rehearsal_arr has one entry per Langevin step; sample at tau_meas rate
        reh_meas = r['rehearsal_arr'][::tau_meas][:len(sr_log)]
        reh_sm   = uniform_filter1d(reh_meas.astype(float), size=win)
        t_reh    = np.arange(len(reh_sm))
        ax3.plot(t_reh, reh_sm * gate_threshold,
                 color=VIOLET, lw=1.2, ls=':', alpha=0.7,
                 label=f"Rehearsal rate (×{gate_threshold}) = {r['rehearsal_rate']:.3f}")

    ax3.set_xlabel("Cycle", fontsize=9)
    ax3.set_ylabel("Surprise / Field Volatility", fontsize=9)
    ax3.set_title("Tension Dynamics + Speed Modulation\nViolet dotted = rehearsal accept rate",
                  color=FG, fontsize=9)
    ax3.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none', loc='upper right')
    ax3b.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none', loc='upper left')
    ax3.grid(True, alpha=0.12); ax3.tick_params(labelsize=7)
    ax3.set_xlim(0, len(sr_log))


# ── PANEL 4: Campo × Alignment projection colored by Surprise + centroid ─────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(BG)

pts_f2 = r['buf_final']
if len(pts_f2) > 0:
    sc4 = ax4.scatter(pts_f2[:, 0], pts_f2[:, 1],
                      c=pts_f2[:, 2], cmap='inferno',
                      s=6, alpha=0.75, linewidths=0)
    cb4 = plt.colorbar(sc4, ax=ax4, shrink=0.82)
    cb4.set_label('Surprise (z-score)', color=FG, fontsize=7)
    cb4.ax.yaxis.set_tick_params(labelsize=6)

    ax4.axhline(0, color=GOLD, lw=1.0, ls='--', alpha=0.55)
    ax4.fill_between([-3.5, 3.5], -3.5, 0, alpha=0.04, color=ROSE)
    ax4.fill_between([-3.5, 3.5],  0,  3.5, alpha=0.04, color=GREEN)

    # Coherence centroid projected to (campo, alignment) axes
    if r['centroid_n'] >= min_events_before_pull:
        # Centroid is in raw space; normalize roughly with buf stats
        raw_data = pts_f2   # already z-scored
        cen_raw  = r['centroid_final']
        # Project centroid raw coords onto z-scored axes
        # Use a rough estimate: centroid raw / buf std from field
        cx = 0.0  # in z-score space, centroid projects near 0 by construction
        cy = 0.0
        ax4.scatter([cx], [cy], color=CENTROID_COL, s=200, marker='*',
                    zorder=15, label='Centroid (projected)')
        # Arrow from cloud mean to centroid
        mx = pts_f2[:, 0].mean(); my = pts_f2[:, 1].mean()
        ax4.annotate('', xy=(cx, cy), xytext=(mx, my),
                     arrowprops=dict(arrowstyle='->', color=GOLD, lw=2.0))

    ax4.set_xlabel("Campo (z-score)", fontsize=9)
    ax4.set_ylabel("Alignment (z-score)", fontsize=9)
    ax4.set_title("State Space Projection\n★ = coherence centroid  |  Arrow = centroid pull direction",
                  color=FG, fontsize=9)
    ax4.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='lower right')
    ax4.grid(True, alpha=0.12); ax4.tick_params(labelsize=7)
    ax4.set_xlim(-3.5, 3.5); ax4.set_ylim(-3.5, 3.5)


# ── Summary text ──────────────────────────────────────────────────────────────
pca_r, _ = r['mean_pca']
coh_state = ('COHERENT'        if r['n_events'] > 0 and surp_rat < 1.0 else
             'EVENTS DETECTED' if r['n_events'] > 0 else
             'EXPLORING')
improvement = 2.054 - surp_rat

summary = [
    "SFE-05.5  ACTIVE INFERENCE",
    "─" * 32,
    "",
    "[1] Coherence centroid pull",
    "[2] Counterfactual rehearsal",
    "[3] Speed modulation",
    "[4] Adaptive λ (inherited)",
    "[5] Memory gating (inherited)",
    "",
    "─" * 32,
    f"alpha_attract  = {alpha_attract}",
    f"min_events     = {min_events_before_pull}",
    f"n_candidates   = {n_candidates}",
    f"rehearsal_on   = {rehearsal_on}",
    f"dt_coherent    = {dt_coherent}",
    f"dt_surprised   = {dt_surprised}",
    "",
    "─" * 32,
    "CENTROID",
    f"  n events     = {r['centroid_n']}",
    f"  position     =",
    f"  [{r['centroid_final'][0]:.4f},",
    f"   {r['centroid_final'][1]:.4f},",
    f"   {r['centroid_final'][2]:.4f}]",
    "",
    "REHEARSAL",
    f"  accept rate  = {r['rehearsal_rate']:.4f}",
    "",
    "SURPRISE",
    f"  mean surp    = {mean_surp:.4f}",
    f"  field vol.   = {field_volatility:.4f}",
    f"  ratio        = {surp_rat:.4f}",
    f"  vs SFE-05.4  = {improvement:+.4f}",
    "",
    "GATE RATIO",
    f"  {r['gate_ratio']:.3f}",
    "",
    f"EVENTS: {r['n_events']}",
    "",
    "─" * 32,
    f"STATE: {coh_state}",
]

fig.text(0.698, 0.456, "\n".join(summary),
         fontsize=7.0, fontfamily='monospace', color=FG, va='top',
         bbox=dict(boxstyle='round,pad=0.75', facecolor='#0a0c14',
                   edgecolor=GOLD, linewidth=1.3, alpha=0.97))

plt.savefig(os.path.join(FIG_DIR, 'sfe055_active_inference.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SFE-05.5  Final Summary")
print("=" * 70)
print()
print("  STATE TENSOR  R^(512×3)")
print("    X — Campo     |J(x,t)|")
print("    Y — Alignment cos(grad_true, grad_perceived)")
print("    Z — Surprise  |x_true - x_hat|")
print()
print("  GEOMETRY: Convex Hull 3D")
for lbl in ['early', 'mid', 'late', 'final']:
    e = snaps.get(lbl)
    if e:
        print(f"    {lbl:8s}  vol={e[0]:.4f}  area={e[1]:.4f}")
print()
print("  ACTIVE INFERENCE")
print(f"    alpha_attract   = {alpha_attract}")
print(f"    rehearsal_on    = {rehearsal_on}")
print(f"    rehearsal rate  = {r['rehearsal_rate']:.4f}")
print(f"    centroid n      = {r['centroid_n']}")
print(f"    centroid pos    = {r['centroid_final'].round(4)}")
print()
print(f"  SURPRISE vs BASELINE")
print(f"    SFE-05.4 ratio  = 2.054")
print(f"    SFE-05.5 ratio  = {surp_rat:.4f}")
print(f"    improvement     = {improvement:+.4f}")
print()
print(f"  COHERENCE EVENTS  = {r['n_events']}")
print(f"  GATE RATIO        = {r['gate_ratio']:.3f}")
print(f"  LAMBDA FINAL      = {r['lam_final']:.4f}")
print(f"  STATE             = {coh_state}")
print()
print("  EVOLUTION FROM SFE-05.4")
print("    No centroid      →  CoherenceCentroid (incremental, emergent)")
print("    Fixed step       →  active_inference_step (attraction + rehearsal)")
print("    Constant dt      →  Speed-modulated dt (coherent vs surprised)")
print("    Passive attractor →  Active pull toward recorded geometry")
print("=" * 70)
