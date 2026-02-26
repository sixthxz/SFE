# -*- coding: utf-8 -*-
"""
SFE-05.4 — Field Navigation System · Attractor Definition
===========================================================
Stochastic Field Engine, revision 05.4

EVOLUTION FROM SFE-05.3:
    SFE-05.3 established the Manifold World: Convex Hull 3D, Surprise on
    the Z axis, S-U as a structural plane, hull volume stability as the
    coherence criterion. The agent was in EXPLORING state (Surprise/
    Volatility = 2.05, no coherence events detected, fixed lambda).

    SFE-05.4 adds three mechanisms that allow coherence to emerge:

    1. ATTRACTOR DEFINITION — Coherence Event Recorder
       Monitors two simultaneous conditions every COHERENCE_WIN cycles:
           Condition A: rolling Var(volume) / mean(volume)^2 < coherence_var_th
           Condition B: mean_surprise / field_volatility < gate_threshold
       When both hold, the current normalized point cloud is captured.
       PCA extracts principal axes. Hull volume and orientation are logged.
       This geometry IS the attractor — emergent, not prescribed.

    2. MANIFOLD TENSION DYNAMICS — Adaptive lambda
       At every hull evaluation:
           z = (V_current - V_mean) / V_std  over rolling window
           z > 1.0  → lambda *= 1.05  (structural stress, increase effort)
           z < -0.5 → lambda *= 0.98  (volume contracted, relax)
       lambda capped in [lambda_min, lambda_max].
       The shape of the manifold modulates the agent's coupling strength.

    3. SELECTIVE MEMORY GATING
       Buffer writes only when: Surprise < field_volatility * gate_threshold
       When gate=0: buffer unchanged, agent holds its confident memory.
       gate_ratio = fraction of cycles that wrote to buffer.
       Effect: point cloud compresses toward low-surprise regions.

STATE TENSOR: R^(N×3)
    X — Campo:     |J(x,t)|  local probability flux magnitude
    Y — Alignment: cos(grad_rho_true, grad_rho_perceived)
    Z — Surprise:  |x_true - x_hat|  Kalman innovation residual

PARAMETERS (all tunable):
    lambda_min       = 0.05   minimum coupling effort
    lambda_max       = 0.80   maximum coupling effort
    window_scale     = 1.0    multiplier on adaptive window length
    gate_threshold   = 2.5    Surprise/volatility ratio for gating
    coherence_var_th = 0.15   rolling relative variance threshold

COHERENCE DEFINITION (do not modify):
    Coherence is reached when simultaneously:
    1. Var(volume) / mean(volume)^2 < coherence_var_th  (sustained stability)
    2. mean_surprise / field_volatility < gate_threshold  (anticipating field)
    The geometry recorded at that moment is the coherence signature.
    Its shape is not prescribed. It is whatever emerges.
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
FIG_DIR = '/tmp/sfe054_figs'
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.4  —  Field Navigation System  ·  Attractor Definition")
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

# ─── Tunable parameters ──────────────────────────────────────────────────────
lambda_coup_init = 0.30
lambda_min       = 0.05
lambda_max       = 0.80
sigma_m          = 0.90
sigma_memory     = 1.20
window_scale     = 1.0
gate_threshold   = 2.5    # Surprise/volatility — calibrated to actual distribution
coherence_var_th = 0.15   # relative variance: Var(V)/mean(V)^2

# ─── Buffer / geometry ───────────────────────────────────────────────────────
BUFFER_N      = 512
COHERENCE_WIN = 40   # cycles between hull evaluations

# ─── Field volatility baseline: σ of free diffusion over tau_meas steps ──────
field_volatility = np.sqrt(2 * D_diff * tau_meas * dt)

print(f"  lambda_init={lambda_coup_init}  range=[{lambda_min}, {lambda_max}]")
print(f"  sigma_m={sigma_m}  sigma_memory={sigma_memory}")
print(f"  gate_threshold={gate_threshold}  coherence_var_th={coherence_var_th}")
print(f"  Field volatility σ = {field_volatility:.4f}")
print(f"  Buffer: R^({BUFFER_N}×3)  [Campo | Alignment | Surprise]")
print()


# ═════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER with selective gating
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    """
    R^(N×3) static memory block.
    Selective gate: write only when Surprise < field_volatility * gate_threshold.
    When gate=0: buffer holds previous state unchanged.
    """
    def __init__(self, N, cols=3):
        self.N    = N
        self.cols = cols
        if TORCH:
            self.buf = torch.zeros(N, cols, dtype=torch.float32)
        else:
            self.buf = np.zeros((N, cols), dtype=np.float32)
        self.filled       = 0
        self.writes       = 0
        self.total_cycles = 0

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
        if TORCH:
            return self.buf[:self.filled].numpy()
        return self.buf[:self.filled]

    def normalize(self):
        """Z-score per column: T_hat = (T - mu) / sigma."""
        data = self.get_numpy()
        if len(data) < 4:
            return data, np.zeros(self.cols), np.ones(self.cols)
        if TORCH:
            T     = self.buf[:self.filled]
            mu    = T.mean(dim=0)
            sigma = T.std(dim=0) + 1e-8
            return ((T - mu) / sigma).numpy(), mu.numpy(), sigma.numpy()
        mu    = data.mean(axis=0)
        sigma = data.std(axis=0) + 1e-8
        return (data - mu) / sigma, mu, sigma

    @property
    def gate_ratio(self):
        return self.writes / max(self.total_cycles, 1)


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
    return rho / np.trapezoid(rho, x_grid)

def fp_flux(rho, F_arr):
    return (F_arr / gamma) * rho - D_diff * np.gradient(rho, x_grid)

def fp_step(rho, F_arr):
    N_  = len(rho)
    v   = F_arr / gamma
    df  = np.zeros(N_ + 1)
    ff  = np.zeros(N_ + 1)
    for i in range(1, N_):
        vf    = 0.5 * (v[i - 1] + v[i])
        df[i] = vf * rho[i - 1] if vf >= 0 else vf * rho[i]
        ff[i] = D_diff * (rho[i] - rho[i - 1]) / dx
    rho_new = np.maximum(rho - (dt / dx) * np.diff(df - ff), 0.0)
    norm    = np.trapezoid(rho_new, x_grid)
    return rho_new / norm if norm > 1e-12 else rho_new

def langevin_step(x, rho_true, F_free, rng, lam):
    J_arr  = fp_flux(rho_true, F_free)
    J_at   = float(np.interp(x, x_grid, J_arr))
    F_self = lam * J_at / (abs(J_at) + 1e-10)
    xi     = np.sqrt(2 * kBT * gamma) * rng.standard_normal()
    dx_    = (F_self / gamma) * dt + xi * np.sqrt(dt) / gamma
    return float(np.clip(x + dx_, x_min + 0.1, x_max - 0.1)), abs(J_at)


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN + PERCEIVED FIELD
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
        K          = self.P / (self.P + sigma_m ** 2)
        innov      = z - self.x_hat
        self.x_hat += K * innov
        self.P     *= (1 - K)
        return P_prior, abs(innov)

    def I_gain(self, P_prior):
        return max(0.5 * np.log2(1.0 + P_prior / sigma_m ** 2), 0.0)

    def reset(self):
        self.x_hat = 0.0
        self.P     = 2 * kBT / gamma * tau_meas * dt


class PerceivedField:
    def __init__(self, max_samples=300):
        self.samples = []
        self.weights = []
        self.max_n   = max_samples

    def add(self, xp, w=1.0):
        self.samples.append(xp)
        self.weights.append(w)
        if len(self.samples) > self.max_n:
            self.samples.pop(0)
            self.weights.pop(0)

    def get_rho(self):
        if len(self.samples) < 2:
            return gaussian_rho(0.0, 1.0)
        rho   = np.zeros(Nx)
        w_tot = sum(self.weights)
        for xp, w in zip(self.samples, self.weights):
            rho += (w / w_tot) * np.exp(-0.5 * ((x_grid - xp) / sigma_memory) ** 2)
        norm = np.trapezoid(rho, x_grid)
        return rho / norm if norm > 1e-12 else rho


def compute_alignment(x_pos, rho_true, rho_perc):
    g_t = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_p = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_t) < 1e-10 or abs(g_p) < 1e-10:
        return 0.0
    return float(np.clip(g_t * g_p / (abs(g_t) * abs(g_p)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY: Convex Hull 3D + PCA
# ═════════════════════════════════════════════════════════════════════════════

def convex_hull_3d(pts):
    """Returns (volume, area, hull) or (0.0, 0.0, None)."""
    if len(pts) < 5:
        return 0.0, 0.0, None
    try:
        h = ConvexHull(pts)
        return h.volume, h.area, h
    except QhullError:
        return 0.0, 0.0, None


def pca_axes(pts):
    """
    PCA of point cloud.
    Returns: explained_variance_ratio (3,), principal_axes (3×3) rows=eigvecs.
    """
    if len(pts) < 4:
        return np.array([1.0, 0.0, 0.0]), np.eye(3)
    c       = pts - pts.mean(axis=0)
    cov     = np.cov(c.T)
    if cov.ndim < 2:
        return np.array([1.0, 0.0, 0.0]), np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    order      = np.argsort(vals)[::-1]
    vals       = np.maximum(vals[order], 0.0)
    vecs       = vecs[:, order]
    ratio      = vals / (vals.sum() + 1e-12)
    return ratio, vecs.T   # axes as rows


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE EVENT RECORDER
# ═════════════════════════════════════════════════════════════════════════════

class CoherenceRecorder:
    """
    At each hull evaluation, check:
        A: Var(V) / mean(V)^2 < coherence_var_th
        B: mean_recent_surprise / field_volatility < gate_threshold
    When A and B hold simultaneously, record the geometry.
    The recorded shape is the emergent attractor — not prescribed.
    """

    def __init__(self, var_th, field_vol, gate_th, window):
        self.var_th   = var_th
        self.fvol     = field_vol
        self.gate_th  = gate_th
        self.window   = max(window, 4)
        self.events   = []
        self._vols    = []

    def check(self, vol, mean_surp, pts, lam, cycle):
        """
        Returns (is_event, rel_var, surp_ratio).
        Records event if conditions met.
        """
        self._vols.append(vol)
        if len(self._vols) > self.window:
            self._vols.pop(0)
        if len(self._vols) < 4:
            return False, 0.0, 0.0

        mean_v    = np.mean(self._vols) + 1e-8
        rel_var   = float(np.var(self._vols)) / (mean_v ** 2)
        surp_rat  = mean_surp / (self.fvol + 1e-10)

        is_event = (rel_var < self.var_th) and (surp_rat < self.gate_th)
        if is_event and len(pts) >= 5:
            vol_e, area_e, hull_e = convex_hull_3d(pts)
            ratio_e, axes_e       = pca_axes(pts)
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
            })
        return is_event, rel_var, surp_rat

    @property
    def n(self):
        return len(self.events)

    def mean_vol(self):
        if not self.events:
            return 0.0
        return float(np.mean([e['vol'] for e in self.events]))

    def mean_pca(self):
        if not self.events:
            return np.ones(3) / 3, np.eye(3)
        ratios = np.array([e['pca_ratio'] for e in self.events])
        axes   = np.array([e['pca_axes']  for e in self.events])
        return ratios.mean(axis=0), axes.mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Field navigation — Attractor Definition")

def run_navigation(seed=42, N=N):
    rng  = np.random.default_rng(seed)
    kf   = KalmanOD()
    pf   = PerceivedField(max_samples=300)
    cbuf = CircularBuffer(BUFFER_N, cols=3)

    # Adaptive window: longer when field calm, shorter when volatile
    sigma_field_max = field_volatility * 2.0
    window_len = max(int(N_cycles * field_volatility / sigma_field_max * window_scale), 8)

    recorder = CoherenceRecorder(
        var_th   = coherence_var_th,
        field_vol = field_volatility,
        gate_th   = gate_threshold,
        window    = window_len,
    )

    rho_true = gaussian_rho(0.0, 2.0)
    F_free   = np.zeros(Nx)
    x        = 0.0
    pf.add(x)
    lam      = lambda_coup_init

    # Logs
    surprise_log   = []
    alignment_log  = []
    campo_log      = []
    lambda_log     = []
    surp_ratio_log = []
    volume_log     = []
    area_log       = []
    rel_var_log    = []
    event_cycles   = []

    # Temporal snapshots
    snap_at       = {N // 8: 'early', N // 2: 'mid', 7 * N // 8: 'late'}
    snap_vol_area = {}

    vol_window  = []      # rolling window for tension dynamics
    surp_window = []      # rolling window for surprise condition
    cycle       = 0
    n_reset     = max(N_cycles // 8, 1)
    I_cum       = 0.0

    for i in range(N):

        # ── Measurement cycle ──────────────────────────────────────────────
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas       = x + sigma_m * rng.standard_normal()
            P_prior, surp = kf.update(x_meas)
            I_step        = kf.I_gain(P_prior)
            I_cum        += kBT * Landauer * I_step

            rho_perc  = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)

            J_arr = fp_flux(rho_true, F_free)
            campo = float(abs(np.interp(x, x_grid, J_arr)))

            # ── Selective Memory Gating ───────────────────────────────────
            gate = 1 if surp < field_volatility * gate_threshold else 0
            cbuf.push([campo, alignment, surp], gate=gate)

            surprise_log.append(surp)
            alignment_log.append(alignment)
            campo_log.append(campo)
            lambda_log.append(lam)
            surp_ratio_log.append(surp / (field_volatility + 1e-10))

            # Rolling surprise window for coherence check
            surp_window.append(surp)
            if len(surp_window) > COHERENCE_WIN:
                surp_window.pop(0)

            cycle += 1

            # ── Hull evaluation every COHERENCE_WIN cycles ────────────────
            if cycle % COHERENCE_WIN == 0 and cbuf.filled >= 5:
                T_hat, _, _ = cbuf.normalize()
                vol, area, _ = convex_hull_3d(T_hat)
                volume_log.append(vol)
                area_log.append(area)
                vol_window.append(vol)
                if len(vol_window) > window_len:
                    vol_window.pop(0)

                # ── Manifold Tension Dynamics ─────────────────────────────
                if len(vol_window) >= 3:
                    v_mean = np.mean(vol_window)
                    v_std  = np.std(vol_window) + 1e-8
                    z      = (vol - v_mean) / v_std
                    if z > 1.0:
                        lam = min(lam * 1.05, lambda_max)
                    elif z < -0.5:
                        lam = max(lam * 0.98, lambda_min)

                # ── Coherence event check ─────────────────────────────────
                mean_surp_recent = float(np.mean(surp_window))
                is_ev, rel_var, surp_rat = recorder.check(
                    vol, mean_surp_recent, T_hat, lam, cycle
                )
                rel_var_log.append(rel_var)

                if is_ev:
                    event_cycles.append(cycle)
                    print(f"    ★ Coherence event  cycle={cycle:4d}  "
                          f"vol={vol:.3f}  surp/vol={surp_rat:.3f}  "
                          f"rel_var={rel_var:.4f}  λ={lam:.3f}")

        # ── Kalman reset ──────────────────────────────────────────────────
        if (i + 1) % (n_reset * tau_meas) == 0 and i > 0:
            kf.reset()

        # ── Langevin step ─────────────────────────────────────────────────
        x_new, _ = langevin_step(x, rho_true, F_free, rng, lam)
        rho_true  = fp_step(rho_true, F_free)
        pf.add(x_new, w=1.0 / (sigma_m + 0.1))
        x = x_new

        # ── Snapshots ─────────────────────────────────────────────────────
        if i in snap_at:
            T_hat, _, _ = cbuf.normalize()
            lbl          = snap_at[i]
            vol, area, hull = convex_hull_3d(T_hat)
            snap_vol_area[lbl] = (vol, area, hull, T_hat.copy())

    # Final
    T_hat_f, _, _ = cbuf.normalize()
    vol_f, area_f, hull_f = convex_hull_3d(T_hat_f)
    snap_vol_area['final'] = (vol_f, area_f, hull_f, T_hat_f.copy())

    vol_arr     = np.array(volume_log)
    rel_var_arr = np.array(rel_var_log)

    return dict(
        buf_final       = T_hat_f,
        snap_vol_area   = snap_vol_area,
        volume_log      = vol_arr,
        area_log        = np.array(area_log),
        rel_var_log     = rel_var_arr,
        lambda_log      = np.array(lambda_log),
        surprise_log    = np.array(surprise_log),
        alignment_log   = np.array(alignment_log),
        campo_log       = np.array(campo_log),
        surp_ratio_log  = np.array(surp_ratio_log),
        events          = recorder.events,
        n_events        = recorder.n,
        mean_vol_events = recorder.mean_vol(),
        mean_pca        = recorder.mean_pca(),
        gate_ratio      = cbuf.gate_ratio,
        event_cycles    = event_cycles,
        window_len      = window_len,
        lam_final       = lam,
        n_cycles        = cycle,
        vol_final       = vol_f,
        area_final      = area_f,
        I_cum           = I_cum,
    )


print("  Running...", end='', flush=True)
r = run_navigation(seed=42)
print(f"\n  done. {r['n_cycles']} cycles.")

snaps = r['snap_vol_area']
print(f"\n  Hull volumes:")
for lbl in ['early', 'mid', 'late', 'final']:
    e = snaps.get(lbl)
    if e:
        print(f"    {lbl:8s}  vol={e[0]:.4f}  area={e[1]:.4f}")

mean_surp = float(np.mean(r['surprise_log']))
surp_rat  = mean_surp / field_volatility
print(f"\n  Mean surprise         = {mean_surp:.4f}")
print(f"  Field volatility σ    = {field_volatility:.4f}")
print(f"  Surprise / volatility = {surp_rat:.4f}")
print(f"  Gate ratio            = {r['gate_ratio']:.3f}")
print(f"  Lambda final          = {r['lam_final']:.4f}")
print(f"  Coherence events      = {r['n_events']}")
if r['n_events'] > 0:
    pca_ratio, pca_axes_ = r['mean_pca']
    print(f"  Mean vol at events    = {r['mean_vol_events']:.4f}")
    print(f"  Mean PCA ratio        = [{pca_ratio[0]:.3f}, {pca_ratio[1]:.3f}, {pca_ratio[2]:.3f}]")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Manifold World
# ═════════════════════════════════════════════════════════════════════════════
print("\nRendering...", end='', flush=True)

BG     = '#07080f'
FG     = '#dde1ec'
GOLD   = '#f5c842'
TEAL   = '#3dd6c8'
VIOLET = '#b87aff'
ROSE   = '#ff5f7e'
GREEN  = '#4ade80'
AMBER  = '#fb923c'
COH    = '#fde68a'   # coherence event highlight
DIM    = '#1e2235'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor':   DIM, 'text.color':    FG,
    'axes.labelcolor':  FG,  'xtick.color':   '#555870',
    'ytick.color':      '#555870',
})

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    "SFE-05.4  ·  Attractor Definition\n"
    "Adaptive λ  ·  Selective Memory Gating  ·  Coherence Event Recording",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold'
)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34,
              top=0.966, bottom=0.06, left=0.04, right=0.97)


def draw_hull(ax, pts, hull, fc, ec, af=0.10, ae=0.30):
    if hull is None or pts is None:
        return
    step  = max(len(hull.simplices) // 300, 1)
    verts = [pts[s] for s in hull.simplices[::step]]
    poly  = Poly3DCollection(verts, alpha=af, linewidth=0.3)
    poly.set_facecolor(fc)
    poly.set_edgecolor(ec)
    ax.add_collection3d(poly)


def style_3d(ax, xl, yl, zl, title):
    ax.set_xlabel(xl, fontsize=8, labelpad=2)
    ax.set_ylabel(yl, fontsize=8, labelpad=2)
    ax.set_zlabel(zl, fontsize=8, labelpad=2)
    ax.set_title(title, color=FG, fontsize=9, pad=4)
    ax.tick_params(labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(DIM)
    ax.grid(True, alpha=0.10)


# ── PANEL 0: Final point cloud + hull + coherence event geometry ──────────────
ax0 = fig.add_subplot(gs[0, :2], projection='3d')
ax0.set_facecolor(BG)

pts_f   = r['buf_final']
_, _, hull_f = convex_hull_3d(pts_f)
n_pts   = len(pts_f)
t_col   = np.linspace(0, 1, n_pts)

sc = ax0.scatter(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2],
                 c=t_col, cmap='plasma', s=7, alpha=0.70, linewidths=0)
draw_hull(ax0, pts_f, hull_f, TEAL, TEAL, af=0.07, ae=0.18)

# Coherence event hulls in COH color
for ev in r['events'][:3]:
    draw_hull(ax0, ev['pts'], ev['hull'], COH, COH, af=0.22, ae=0.60)
    ctr = ev['pts'].mean(axis=0)
    ax0.scatter(*ctr, color=COH, s=80, marker='*', zorder=10, alpha=0.95)

# PCA arrows of mean coherence signature
if r['n_events'] > 0:
    pca_ratio, pca_axes_ = r['mean_pca']
    ctr_f   = pts_f.mean(axis=0)
    pca_col = [ROSE, GREEN, AMBER]
    for k in range(3):
        scale = pca_ratio[k] * 2.5
        ax0.quiver(*ctr_f, *(pca_axes_[k] * scale),
                   color=pca_col[k], lw=2.0, alpha=0.85,
                   arrow_length_ratio=0.25)

# S-U plane at z=0
try:
    Xp, Yp = np.meshgrid([-2.5, 2.5], [-2.5, 2.5])
    ax0.plot_surface(Xp, Yp, np.zeros_like(Xp),
                     alpha=0.05, color=GOLD, linewidth=0)
    ax0.text(2.2, 2.2, 0.08, "S-U", color=GOLD, fontsize=7, alpha=0.55)
except Exception:
    pass

plt.colorbar(sc, ax=ax0, shrink=0.45, pad=0.02).set_label(
    'Time (oldest→newest)', color=FG, fontsize=7)
style_3d(ax0,
         "Campo  |J(x)|", "Alignment", "Surprise  |innov|",
         f"Point Cloud + Convex Hull  [gate ratio={r['gate_ratio']:.3f}  "
         f"λ_final={r['lam_final']:.3f}  events={r['n_events']}]\n"
         f"Gold stars = coherence events  |  Arrows = PCA axes")
ax0.view_init(elev=22, azim=-52)


# ── PANEL 1: Hull evolution — early / final / coherence events ────────────────
ax1 = fig.add_subplot(gs[0, 2], projection='3d')
ax1.set_facecolor(BG)

for lbl, col, af, ae in [('early', ROSE, 0.12, 0.35), ('final', TEAL, 0.07, 0.20)]:
    e = snaps.get(lbl)
    if e:
        v_, a_, h_, p_ = e
        ax1.scatter(p_[:, 0], p_[:, 1], p_[:, 2], c=col, s=4, alpha=0.40, linewidths=0)
        draw_hull(ax1, p_, h_, col, col, af, ae)

# First coherence event hull
if r['n_events'] > 0:
    ev0 = r['events'][0]
    if ev0['hull'] is not None:
        draw_hull(ax1, ev0['pts'], ev0['hull'], COH, COH, af=0.28, ae=0.70)
        ax1.scatter(ev0['pts'][:, 0], ev0['pts'][:, 1], ev0['pts'][:, 2],
                    c=COH, s=5, alpha=0.55, linewidths=0)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ROSE, ms=7,
           label=f"early  V={snaps.get('early', (0,))[0]:.3f}"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, ms=7,
           label=f"final  V={snaps.get('final', (0,))[0]:.3f}"),
]
if r['n_events'] > 0:
    handles.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor=COH, ms=10,
                          label=f"coherence V={r['mean_vol_events']:.3f}"))
ax1.legend(handles=handles, fontsize=7, facecolor='#0d0f18', edgecolor='none')
style_3d(ax1, "Campo", "Alignment", "Surprise",
         "Hull Evolution\nRose=early  Teal=final  Gold=coherence")
ax1.view_init(elev=25, azim=-40)


# ── PANEL 2: Hull volume + relative variance + event markers ──────────────────
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

    ax2b = ax2.twinx()
    ax2b.set_facecolor(BG)
    if len(rv_arr) > 0:
        ax2b.plot(t_rv, rv_arr, color=GOLD, lw=1.5, ls='--',
                  label=f'Rel. var  Var(V)/mean(V)²')
        ax2b.axhline(coherence_var_th, color=GREEN, lw=1.2, ls=':',
                     label=f'threshold={coherence_var_th}')
        ax2b.fill_between(t_rv, 0, rv_arr,
                          where=(rv_arr < coherence_var_th),
                          alpha=0.28, color=GREEN)
    ax2b.set_ylabel("Var(V) / mean(V)²", color=GOLD, fontsize=8)
    ax2b.tick_params(axis='y', labelcolor=GOLD, labelsize=7)

    # Event markers
    for ec in r['event_cycles']:
        ax2.axvline(ec, color=COH, lw=1.3, ls='--', alpha=0.8)

    ax2.set_xlabel("Cycle", fontsize=9)
    ax2.set_ylabel("Hull Volume", color=VIOLET, fontsize=9)
    ax2.set_title("Volume + Relative Variance\nGreen = below coherence threshold  |  "
                  "Dashed gold = coherence events",
                  color=FG, fontsize=8)
    ax2.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper right')
    ax2b.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper left')
    ax2.grid(True, alpha=0.12)
    ax2.tick_params(labelsize=7)


# ── PANEL 3: Adaptive lambda + Surprise/Volatility ────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)

sr_log  = r['surp_ratio_log']
lam_log = r['lambda_log']
cyc_t   = np.arange(len(sr_log))

if len(sr_log) > 0:
    win    = max(len(sr_log) // 40, 3)
    sr_sm  = uniform_filter1d(sr_log, size=win)
    lam_sm = uniform_filter1d(lam_log, size=win)

    ax3.fill_between(cyc_t, 0, sr_log, alpha=0.10, color=ROSE)
    ax3.plot(cyc_t, sr_sm, color=ROSE, lw=2.0, label='Surprise/Volatility')
    ax3.axhline(gate_threshold, color=AMBER, lw=1.2, ls='--',
                label=f'gate_threshold = {gate_threshold}')
    ax3.axhline(1.0, color=GREEN, lw=1.0, ls=':',
                label='Coherence target = 1.0')
    ax3.fill_between(cyc_t, 0, sr_sm, where=(sr_sm < gate_threshold),
                     alpha=0.15, color=AMBER)

    ax3b = ax3.twinx()
    ax3b.set_facecolor(BG)
    ax3b.plot(cyc_t, lam_sm, color=TEAL, lw=1.8, ls='--', label='λ (adaptive)')
    ax3b.axhline(lambda_min, color=TEAL, lw=0.6, ls=':', alpha=0.4)
    ax3b.axhline(lambda_max, color=TEAL, lw=0.6, ls=':', alpha=0.4)
    ax3b.set_ylabel("λ_coupling", color=TEAL, fontsize=8)
    ax3b.tick_params(axis='y', labelcolor=TEAL, labelsize=7)
    ax3b.set_ylim(0, lambda_max * 1.25)

    ax3.set_xlabel("Cycle", fontsize=9)
    ax3.set_ylabel("Surprise / Field Volatility", fontsize=9)
    ax3.set_title("Manifold Tension Dynamics\n"
                  "λ increases under structural stress  |  relaxes on contraction",
                  color=FG, fontsize=9)
    ax3.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper right')
    ax3b.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper left')
    ax3.grid(True, alpha=0.12)
    ax3.tick_params(labelsize=7)
    ax3.set_xlim(0, len(sr_log))


# ── PANEL 4: Campo × Alignment projection + PCA inset ────────────────────────
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

    # S-U boundary
    ax4.axhline(0, color=GOLD, lw=1.0, ls='--', alpha=0.55, label='S-U plane')
    ax4.fill_between([-3.5, 3.5], -3.5, 0, alpha=0.04, color=ROSE)
    ax4.fill_between([-3.5, 3.5],  0,  3.5, alpha=0.04, color=GREEN)

    # PCA arrows projected onto (x,y)
    if r['n_events'] > 0:
        pca_ratio, pca_ax = r['mean_pca']
        cx = pts_f2[:, 0].mean()
        cy = pts_f2[:, 1].mean()
        for k, col in enumerate([ROSE, GREEN, AMBER]):
            vx = pca_ax[k, 0] * pca_ratio[k] * 2.2
            vy = pca_ax[k, 1] * pca_ratio[k] * 2.2
            ax4.annotate('', xy=(cx + vx, cy + vy), xytext=(cx, cy),
                         arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
        ax4.text(cx + 0.1, cy + 0.1, 'PCA', color=FG, fontsize=7, alpha=0.6)

    ax4.set_xlabel("Campo (z-score)", fontsize=9)
    ax4.set_ylabel("Alignment (z-score)", fontsize=9)
    ax4.set_title("State Space Projection\nCampo × Alignment  [color = Surprise]\n"
                  "Arrows = PCA axes of coherence signature",
                  color=FG, fontsize=9)
    ax4.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='lower right')
    ax4.grid(True, alpha=0.12)
    ax4.set_xlim(-3.5, 3.5)
    ax4.set_ylim(-3.5, 3.5)
    ax4.tick_params(labelsize=7)


# ── Summary text ──────────────────────────────────────────────────────────────
surp_mean  = float(np.mean(r['surprise_log']))
surp_ratio = surp_mean / field_volatility
pca_r, _   = r['mean_pca']

coh_state = ('COHERENT'        if r['n_events'] > 0 and surp_ratio < 1.0 else
             'EVENTS DETECTED' if r['n_events'] > 0 else
             'EXPLORING')

summary = [
    "SFE-05.4  ATTRACTOR DEFINITION",
    "─" * 32,
    "",
    "[1] Coherence event recorder",
    "[2] Manifold tension dynamics",
    "[3] Selective memory gating",
    "",
    "─" * 32,
    f"lambda_init  = {lambda_coup_init}",
    f"lambda_final = {r['lam_final']:.4f}",
    f"range        = [{lambda_min}, {lambda_max}]",
    f"gate_th      = {gate_threshold}",
    f"coh_var_th   = {coherence_var_th}",
    f"window       = {r['window_len']} cycles",
    "",
    "─" * 32,
    "COHERENCE EVENTS",
    f"  N detected  = {r['n_events']}",
    f"  Mean vol    = {r['mean_vol_events']:.4f}",
    f"  PCA ratio   = [{pca_r[0]:.3f},",
    f"               {pca_r[1]:.3f},",
    f"               {pca_r[2]:.3f}]",
    "",
    "GATE",
    f"  gate_ratio  = {r['gate_ratio']:.3f}",
    "",
    "SURPRISE",
    f"  mean surp   = {surp_mean:.4f}",
    f"  field vol.  = {field_volatility:.4f}",
    f"  ratio       = {surp_ratio:.3f}",
    "",
    "─" * 32,
    f"STATE: {coh_state}",
]

fig.text(0.695, 0.456, "\n".join(summary),
         fontsize=7.2, fontfamily='monospace', color=FG, va='top',
         bbox=dict(boxstyle='round,pad=0.75', facecolor='#0a0c14',
                   edgecolor=GOLD, linewidth=1.3, alpha=0.97))


plt.savefig(os.path.join(FIG_DIR, 'sfe054_attractor.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SFE-05.4  Final Summary")
print("=" * 70)
print()
print("  STATE TENSOR  R^(512×3)")
print("    X — Campo:     |J(x,t)|")
print("    Y — Alignment: cos(grad_true, grad_perceived)")
print("    Z — Surprise:  |x_true - x_hat|  (Kalman innovation)")
print()
print("  GEOMETRY: Convex Hull 3D")
for lbl in ['early', 'mid', 'late', 'final']:
    e = snaps.get(lbl)
    if e:
        print(f"    {lbl:8s}  vol={e[0]:.4f}  area={e[1]:.4f}")
print()
print("  MANIFOLD TENSION DYNAMICS")
print(f"    lambda: {lambda_coup_init} → {r['lam_final']:.4f}  (adaptive)")
print()
print("  SELECTIVE MEMORY GATING")
print(f"    gate_ratio = {r['gate_ratio']:.3f}")
print()
print("  SURPRISE / VOLATILITY")
print(f"    mean surprise  = {surp_mean:.4f}")
print(f"    field vol. σ   = {field_volatility:.4f}")
print(f"    ratio          = {surp_ratio:.3f}  (target < 1.0)")
print()
print("  COHERENCE EVENTS")
print(f"    N detected     = {r['n_events']}")
if r['n_events'] > 0:
    print(f"    Mean vol       = {r['mean_vol_events']:.4f}")
    print(f"    PCA variance   = {pca_r}")
    print(f"    State          = {coh_state}")
    print()
    print("  COHERENCE SIGNATURE GEOMETRY")
    print("    (emergent — recorded at event, not prescribed)")
    for k, ev in enumerate(r['events'][:5]):
        print(f"    event {k+1}: cycle={ev['cycle']:4d}  vol={ev['vol']:.4f}  "
              f"λ={ev['lambda']:.3f}  surp/vol={ev['surp_ratio']:.3f}")
print()
print("  EVOLUTION FROM SFE-05.3")
print("    Fixed lambda         →  Adaptive λ (tension dynamics)")
print("    Unconditional buffer →  Gated buffer (low-surprise only)")
print("    No event detection   →  Coherence event recorder + PCA")
print("    No attractor         →  Emergent attractor geometry captured")
print("=" * 70)
