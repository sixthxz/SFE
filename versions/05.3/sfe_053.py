# -*- coding: utf-8 -*-
"""
SFE-05.3 — Field Navigation System · Manifold World
=====================================================
Stochastic Field Engine, revision 05.3

EVOLUTION FROM SFE-05.2:
    SFE-05.2 established the circular buffer and z-score normalization,
    but retained 2D time-series panels alongside the 3D geometry.
    It used Delaunay triangulation projected onto the (x,y) plane,
    treating one variable as "height" rather than a true volumetric axis.
    The KL divergence occupied the Z axis, which measured accumulated
    divergence rather than the moment-to-moment cost of being wrong.

    SFE-05.3 makes the manifold the only visual world.
    No 2D time-series panels. No external scaffolding.
    The agent's state space is the entire output.

STATE TENSOR — R^(N x 3):
    Axis X — Campo:     |J(x,t)|  local probability flux magnitude
    Axis Y — Alignment: cos(grad_rho_true, grad_rho_perceived)  in [-1, 1]
    Axis Z — Surprise:  |x_true - x_hat|  Kalman innovation residual

    Surprise replaces KL divergence as the primary Z axis.
    Rationale: KL is a global integral over the full field. Surprise is local —
    the gap between what the agent predicted and what the field delivered at
    this step. It is the direct cost of belief update: the moment where
    the agent's model fails to anticipate the field.

    When Surprise is low: agent predicts correctly, manifold compresses.
    When Surprise is high: agent is wrong, manifold expands into volume.
    This makes the manifold literally the map of predictive error.

GEOMETRY — Convex Hull (replaces Delaunay 2D):
    SFE-05.2 projected onto (x,y) and used z as height — a terrain model.
    This imposed a structural assumption: one axis dominates.

    SFE-05.3 uses scipy.spatial.ConvexHull in full 3D.
    The hull wraps the point cloud in all directions simultaneously.
    No axis is privileged. The volume is the agent's occupied state space.

    Hull volume as coherence metric:
        High volume  = agent explores many (Campo, Alignment, Surprise) combos
                       = incoherent, high predictive error, dispersed states
        Low volume   = agent's states concentrate in a compact region
                       = coherent, low surprise, stable navigation

    Coherence criterion (Gemini analysis):
        Not minimum volume — minimum variance of volume over a rolling window.
        A small but stable volume = agent has learned the grammar of the noise.
        A shrinking but unstable volume = transient compression, not coherence.

S-U HORIZON — Reframed as Structural Boundary:
    In SFE-05.2, S-U was a diagnostic line chart (vestige of the motor frame).
    In SFE-05.3, S-U is rendered as a threshold plane in the 3D state space.

    Below the S-U plane: measurement cost exceeds information gain.
                         The agent is spending more energy sampling noise
                         than the noise contains usable structure.
                         Manifold is amorphous here.

    Above the S-U plane: information gain exceeds measurement cost.
                         The field contains more structure than the agent
                         is currently paying to measure.
                         Manifold crystallizes here.

    The plane is defined in the (Campo, Alignment) dimensions at the
    S-U threshold value of kBT * I_meas per cycle.

COHERENCE DEFINITION:
    Coherence is not a single number. It is a regime.
    The agent has achieved coherence when:
        1. Hull volume stabilizes (rolling std of volume < threshold)
        2. Mean Surprise decreases below field volatility baseline
        3. Most points lie above the S-U threshold plane
    These three conditions together define "being" rather than "learning."
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
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import os
FIG_DIR = '/tmp/sfe053_figs'
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.3  —  Field Navigation System  ·  Manifold World")
print("=" * 70)
print(f"  Backend: {'PyTorch ' + torch.__version__ if TORCH else 'NumPy'}")
print()


# ─── Units ───────────────────────────────────────────────────────────────────
kBT      = 1.0
gamma    = 1.0
D_diff   = kBT / gamma
Landauer = np.log(2)

# ─── Simulation parameters ───────────────────────────────────────────────────
dt        = 0.01
N         = 20000
tau_meas  = 10
N_cycles  = N // tau_meas

# ─── Field grid ──────────────────────────────────────────────────────────────
x_min, x_max = -8.0, 8.0
Nx           = 400
x_grid       = np.linspace(x_min, x_max, Nx)
dx           = x_grid[1] - x_grid[0]

# ─── Navigation parameters ───────────────────────────────────────────────────
lambda_coup  = 0.3
sigma_m      = 0.9
sigma_memory = 1.2

# ─── Circular buffer ─────────────────────────────────────────────────────────
BUFFER_N = 512    # temporal depth of observation window

# ─── Coherence rolling window ────────────────────────────────────────────────
COHERENCE_WIN = 40   # cycles over which volume variance is computed

print(f"  lambda={lambda_coup}  sigma_m={sigma_m}  sigma_memory={sigma_memory}")
print(f"  Buffer: R^({BUFFER_N} x 3)  axes: [Campo, Alignment, Surprise]")
print(f"  Geometry: Convex Hull 3D  |  Coherence window: {COHERENCE_WIN} cycles")
print()


# ═════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER — R^(N x 3)
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    """
    Static memory block R^(N x 3).
    Columns: [campo, alignment, surprise]
    Push: roll + overwrite index 0. O(N) in NumPy, O(1) conceptually in torch.
    """
    def __init__(self, N, cols=3):
        self.N    = N
        self.cols = cols
        if TORCH:
            self.buf = torch.zeros(N, cols, dtype=torch.float32)
        else:
            self.buf = np.zeros((N, cols), dtype=np.float32)
        self.filled = 0

    def push(self, row):
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
        """Z-score per column: T_hat = (T - mu) / sigma"""
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


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma)**2)
    return rho / np.trapezoid(rho, x_grid)

def fp_flux(rho, F_arr):
    return (F_arr / gamma) * rho - D_diff * np.gradient(rho, x_grid)

def fp_step(rho, F_arr):
    N_  = len(rho)
    v   = F_arr / gamma
    df  = np.zeros(N_ + 1)
    ff  = np.zeros(N_ + 1)
    for i in range(1, N_):
        vf    = 0.5 * (v[i-1] + v[i])
        df[i] = vf * rho[i-1] if vf >= 0 else vf * rho[i]
        ff[i] = D_diff * (rho[i] - rho[i-1]) / dx
    rho_new = np.maximum(rho - (dt / dx) * np.diff(df - ff), 0.0)
    norm    = np.trapezoid(rho_new, x_grid)
    return rho_new / norm if norm > 1e-12 else rho_new

def kl_div(rho_p, rho_q):
    p = np.maximum(rho_p, 1e-12)
    q = np.maximum(rho_q, 1e-12)
    return float(np.trapezoid(p * np.log(p / q), x_grid))

def langevin_step(x, rho_true, F_free, rng):
    J_arr  = fp_flux(rho_true, F_free)
    J_at   = float(np.interp(x, x_grid, J_arr))
    F_self = lambda_coup * J_at / (abs(J_at) + 1e-10)
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
        return self.x_hat   # predicted position (prior)

    def update(self, z):
        P_prior    = self.P
        K          = self.P / (self.P + sigma_m**2)
        innov      = z - self.x_hat   # ← Surprise: prediction error
        self.x_hat += K * innov
        self.P     *= (1 - K)
        return P_prior, abs(innov)    # return raw innovation magnitude

    def I_gain(self, P_prior):
        return max(0.5 * np.log2(1.0 + P_prior / sigma_m**2), 0.0)

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
            rho += (w / w_tot) * np.exp(-0.5 * ((x_grid - xp) / sigma_memory)**2)
        norm = np.trapezoid(rho, x_grid)
        return rho / norm if norm > 1e-12 else rho


def compute_alignment(x_pos, rho_true, rho_perc):
    g_t = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_p = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_t) < 1e-10 or abs(g_p) < 1e-10:
        return 0.0
    return float(np.clip(g_t * g_p / (abs(g_t) * abs(g_p)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# CONVEX HULL UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def convex_hull_volume_and_faces(pts):
    """
    Compute ConvexHull in full 3D.
    Returns: (volume, area, hull object) or (0, 0, None) on failure.
    Volume is the primary coherence metric.
    Area is secondary.
    """
    if len(pts) < 5:
        return 0.0, 0.0, None
    try:
        hull = ConvexHull(pts)
        return hull.volume, hull.area, hull
    except QhullError:
        return 0.0, 0.0, None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Field navigation — Manifold World")

def run_navigation(seed=42, N=N):
    rng  = np.random.default_rng(seed)
    kf   = KalmanOD()
    pf   = PerceivedField(max_samples=300)
    cbuf = CircularBuffer(BUFFER_N, cols=3)

    rho_true = gaussian_rho(0.0, 2.0)
    F_free   = np.zeros(Nx)
    x        = 0.0
    pf.add(x)

    # Per-cycle logs for coherence tracking
    volume_log    = []    # hull volume per snapshot
    area_log      = []    # hull area per snapshot
    surprise_log  = []    # raw surprise per cycle
    alignment_log = []    # raw alignment per cycle
    campo_log     = []    # raw campo per cycle
    I_cum_log     = []    # cumulative I (for S-U plane)

    # Buffer snapshots
    buf_snapshots = {}
    snap_at       = {N//8: 'early', N//2: 'mid', 7*N//8: 'late'}
    snap_vol_area = {}

    W_cum = I_cum = 0.0
    n_reset = max(N_cycles // 8, 1)

    # Track hull volume every COHERENCE_WIN cycles
    hull_vol_series = []
    cycle_count     = 0

    for i in range(N):
        if i % tau_meas == 0:
            # Kalman predict → get prior prediction
            x_pred  = kf.predict_n(tau_meas)
            x_meas  = x + sigma_m * rng.standard_normal()
            P_prior, surprise = kf.update(x_meas)
            I_step  = kf.I_gain(P_prior)
            I_cum  += kBT * Landauer * I_step

            rho_perc  = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)

            # Flux at particle
            J_arr  = fp_flux(rho_true, F_free)
            campo  = float(abs(np.interp(x, x_grid, J_arr)))

            # Push state vector: [campo, alignment, surprise]
            cbuf.push([campo, alignment, surprise])

            surprise_log.append(surprise)
            alignment_log.append(alignment)
            campo_log.append(campo)
            I_cum_log.append(I_cum)

            cycle_count += 1

            # Compute hull every COHERENCE_WIN cycles
            if cycle_count % COHERENCE_WIN == 0 and cbuf.filled >= 5:
                T_hat, _, _ = cbuf.normalize()
                vol, area, _ = convex_hull_volume_and_faces(T_hat)
                volume_log.append(vol)
                area_log.append(area)
                hull_vol_series.append(vol)

        if (i + 1) % (n_reset * tau_meas) == 0 and i > 0:
            kf.reset()

        x_new, campo_step = langevin_step(x, rho_true, F_free, rng)
        rho_true = fp_step(rho_true, F_free)
        pf.add(x_new, w=1.0 / (sigma_m + 0.1))
        x = x_new

        if i in snap_at:
            T_hat, mu_, sig_ = cbuf.normalize()
            label = snap_at[i]
            vol, area, hull = convex_hull_volume_and_faces(T_hat)
            buf_snapshots[label]  = T_hat.copy()
            snap_vol_area[label]  = (vol, area, hull, T_hat.copy())

    # Final snapshot
    T_hat_f, mu_f, sig_f = cbuf.normalize()
    vol_f, area_f, hull_f = convex_hull_volume_and_faces(T_hat_f)
    buf_snapshots['final']  = T_hat_f.copy()
    snap_vol_area['final']  = (vol_f, area_f, hull_f, T_hat_f.copy())

    # Rolling variance of hull volume = coherence stability metric
    vol_arr = np.array(volume_log)
    if len(vol_arr) >= 4:
        roll_std = np.array([
            np.std(vol_arr[max(0, k-4):k+1])
            for k in range(len(vol_arr))
        ])
    else:
        roll_std = np.zeros(len(vol_arr))

    # S-U threshold: mean kBT*I per cycle
    mean_I_per_cycle = I_cum / max(cycle_count, 1)
    su_threshold     = kBT * Landauer * (mean_I_per_cycle / (kBT * Landauer + 1e-10))

    return dict(
        buf_final      = T_hat_f,
        buf_snapshots  = buf_snapshots,
        snap_vol_area  = snap_vol_area,
        volume_log     = vol_arr,
        area_log       = np.array(area_log),
        roll_std       = roll_std,
        surprise_log   = np.array(surprise_log),
        alignment_log  = np.array(alignment_log),
        campo_log      = np.array(campo_log),
        I_cum_log      = np.array(I_cum_log),
        su_threshold   = su_threshold,
        mean_surprise  = float(np.mean(surprise_log)) if surprise_log else 0.0,
        mean_alignment = float(np.mean(alignment_log)) if alignment_log else 0.0,
        n_cycles       = cycle_count,
        vol_final      = vol_f,
        area_final     = area_f,
    )

print("  Running...", end='', flush=True)
r = run_navigation(seed=42)
print(f" done. {r['n_cycles']} cycles.")

snaps = r['snap_vol_area']
vols  = {k: v[0] for k, v in snaps.items()}
print(f"  Hull volumes — early:{vols.get('early',0):.3f}  mid:{vols.get('mid',0):.3f}  "
      f"late:{vols.get('late',0):.3f}  final:{vols.get('final',0):.3f}")
print(f"  Mean surprise    = {r['mean_surprise']:.4f}")
print(f"  Mean alignment   = {r['mean_alignment']:+.4f}")
print(f"  S-U threshold    = {r['su_threshold']:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Manifold World (no 2D time-series panels)
# ═════════════════════════════════════════════════════════════════════════════
print("Rendering...", end='', flush=True)

BG     = '#07080f'
FG     = '#dde1ec'
GOLD   = '#f5c842'
TEAL   = '#3dd6c8'
VIOLET = '#b87aff'
ROSE   = '#ff5f7e'
GREEN  = '#4ade80'
AMBER  = '#fb923c'
DIM    = '#1e2235'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor':   DIM, 'text.color':     FG,
    'axes.labelcolor':  FG,  'xtick.color':    '#555870',
    'ytick.color':      '#555870',
})

fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle(
    "SFE-05.3  ·  Manifold World\n"
    "State Space: Campo × Alignment × Surprise  —  Convex Hull 3D",
    fontsize=13, color=GOLD, y=0.999, fontweight='bold'
)

# Layout: 2 rows × 3 cols — all 3D or geometry-focused
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
              top=0.965, bottom=0.06, left=0.04, right=0.97)


# ─── Helper: draw convex hull faces ──────────────────────────────────────────
def draw_hull(ax, pts, hull, facecolor, edgecolor, alpha_face=0.10, alpha_edge=0.35):
    if hull is None or pts is None:
        return
    verts = [pts[s] for s in hull.simplices]
    poly  = Poly3DCollection(verts, alpha=alpha_face, linewidth=0.3)
    poly.set_facecolor(facecolor)
    poly.set_edgecolor(edgecolor)
    ax.add_collection3d(poly)

def style_3d(ax, xlabel, ylabel, zlabel, title):
    ax.set_xlabel(xlabel, fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.set_title(title, color=FG, fontsize=9, pad=4)
    ax.tick_params(labelsize=6)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(DIM)
    ax.yaxis.pane.set_edgecolor(DIM)
    ax.zaxis.pane.set_edgecolor(DIM)
    ax.grid(True, alpha=0.12)


# ── PANEL 0: Final point cloud + Convex Hull ──────────────────────────────────
ax0 = fig.add_subplot(gs[0, :2], projection='3d')
ax0.set_facecolor(BG)

pts_f = r['buf_final']
_, _, hull_f = convex_hull_volume_and_faces(pts_f)

# Point cloud colored by time (oldest=dark, newest=bright)
n_pts    = len(pts_f)
t_color  = np.linspace(0, 1, n_pts)
sc = ax0.scatter(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2],
                 c=t_color, cmap='plasma', s=7, alpha=0.75, linewidths=0)

# Convex hull envelope
draw_hull(ax0, pts_f, hull_f, facecolor=TEAL, edgecolor=TEAL,
          alpha_face=0.07, alpha_edge=0.25)

# S-U threshold plane at z = su_threshold_normalized
# In z-score space, the mean is 0. The S-U plane sits at z=0 (neutral).
# Points with positive z (high surprise) are above; negative z (low surprise) below.
# Draw a semi-transparent horizontal plane at z=0.
xlim_ = ax0.get_xlim() if ax0.get_xlim() != (0,1) else (-2.5, 2.5)
try:
    xr = np.array([-2.5, 2.5])
    yr = np.array([-2.5, 2.5])
    Xp, Yp = np.meshgrid(xr, yr)
    Zp = np.zeros_like(Xp)
    ax0.plot_surface(Xp, Yp, Zp, alpha=0.06, color=GOLD,
                     linewidth=0, antialiased=False)
    # Annotation
    ax0.text(2.5, 2.5, 0.05, "S-U plane", color=GOLD, fontsize=7, alpha=0.7)
except Exception:
    pass

cbar = plt.colorbar(sc, ax=ax0, shrink=0.5, pad=0.02)
cbar.set_label('Time (oldest→newest)', color=FG, fontsize=7)
cbar.ax.yaxis.set_tick_params(labelsize=6, color=FG)

vol_str  = f"{r['vol_final']:.3f}"
area_str = f"{r['area_final']:.3f}"
style_3d(ax0,
         "Campo  |J(x)|", "Alignment  cos(θ)", "Surprise  |innov|",
         f"Point Cloud + Convex Hull  [final state]\n"
         f"Volume = {vol_str}  |  Area = {area_str}  |  N = {n_pts} pts")
ax0.view_init(elev=22, azim=-52)


# ── PANEL 1: Early vs Final hull overlay ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2], projection='3d')
ax1.set_facecolor(BG)

for label, color, alpha_f, alpha_e in [
    ('early', ROSE,  0.14, 0.40),
    ('final', TEAL,  0.09, 0.25),
]:
    entry = snaps.get(label)
    if entry is None:
        continue
    vol_, area_, hull_, pts_ = entry
    ax1.scatter(pts_[:, 0], pts_[:, 1], pts_[:, 2],
                c=color, s=4, alpha=0.45, linewidths=0)
    draw_hull(ax1, pts_, hull_, color, color, alpha_f, alpha_e)

# Legend proxies
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ROSE, ms=7,
           label=f"early  V={snaps.get('early',(0,))[0]:.2f}"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, ms=7,
           label=f"final  V={snaps.get('final',(0,))[0]:.2f}"),
]
ax1.legend(handles=handles, fontsize=7.5, facecolor='#0d0f18', edgecolor='none',
           loc='upper left')
style_3d(ax1, "Campo", "Alignment", "Surprise",
         "Hull Evolution\nRose=early  Teal=final")
ax1.view_init(elev=25, azim=-40)


# ── PANEL 2: Hull volume over time (coherence tracking) ───────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG)

vol_arr  = r['volume_log']
roll_std = r['roll_std']

if len(vol_arr) > 0:
    t_vol = np.arange(len(vol_arr)) * COHERENCE_WIN   # in cycles

    ax2.fill_between(t_vol, 0, vol_arr, alpha=0.20, color=VIOLET)
    ax2.plot(t_vol, vol_arr, color=VIOLET, lw=1.8, label='Hull volume')

    ax2b = ax2.twinx()
    ax2b.set_facecolor(BG)
    ax2b.plot(t_vol, roll_std, color=GOLD, lw=1.5, ls='--',
              label='Rolling σ (stability)')
    ax2b.set_ylabel("Volume σ  [coherence stability]", color=GOLD, fontsize=8)
    ax2b.tick_params(axis='y', labelcolor=GOLD, labelsize=7)
    ax2b.spines['right'].set_color(GOLD)

    # Shade region where rolling_std is low (potential coherence)
    std_thresh = np.percentile(roll_std, 25) if len(roll_std) > 4 else 1e9
    coherent_mask = roll_std <= std_thresh
    ax2.fill_between(t_vol, 0, vol_arr,
                     where=coherent_mask, alpha=0.35, color=GREEN,
                     label='Low σ region (coherence candidate)')
    ax2.axhline(np.mean(vol_arr), color=TEAL, lw=1.0, ls=':', alpha=0.7,
                label=f'mean = {np.mean(vol_arr):.3f}')

    ax2.set_xlabel("Cycle", fontsize=9, color=FG)
    ax2.set_ylabel("Convex Hull Volume", color=VIOLET, fontsize=9)
    ax2.set_title("Hull Volume over Time\nGreen = low variance (coherence candidate)",
                  color=FG, fontsize=9)
    ax2.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper right')
    ax2.grid(True, alpha=0.15)
    ax2b.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none', loc='upper left')
    ax2.tick_params(axis='both', labelsize=7, colors=FG)


# ── PANEL 3: Surprise over time ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)

surp = r['surprise_log']
cyc_t = np.arange(len(surp))

if len(surp) > 0:
    win = max(len(surp) // 40, 3)
    surp_sm = uniform_filter1d(surp, size=win)

    ax3.fill_between(cyc_t, 0, surp, alpha=0.15, color=ROSE)
    ax3.plot(cyc_t, surp, color='#443', lw=0.4, alpha=0.5)
    ax3.plot(cyc_t, surp_sm, color=ROSE, lw=2.0, label='Surprise (smoothed)')
    ax3.axhline(np.mean(surp), color=GOLD, lw=1.2, ls='--',
                label=f'μ = {np.mean(surp):.4f}')

    # Field volatility baseline: std of particle position would be ~sqrt(2D*t_cycle)
    field_vol = np.sqrt(2 * D_diff * tau_meas * dt)
    ax3.axhline(field_vol, color=GREEN, lw=1.2, ls=':',
                label=f'Field volatility σ = {field_vol:.4f}')

    ax3.set_xlabel("Measurement Cycle", fontsize=9, color=FG)
    ax3.set_ylabel("|x_true − x̂|  [Kalman innovation]", fontsize=9)
    ax3.set_title("Surprise (Axis Z of State Tensor)\n"
                  "When below field volatility: agent anticipates the field",
                  color=FG, fontsize=9)
    ax3.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
    ax3.grid(True, alpha=0.15)
    ax3.tick_params(labelsize=7)
    ax3.set_xlim(0, len(surp))


# ── PANEL 4: 2D projection colored by Surprise ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(BG)

pts_f2 = r['buf_final']
if len(pts_f2) > 0:
    sc4 = ax4.scatter(pts_f2[:, 0], pts_f2[:, 1],
                      c=pts_f2[:, 2], cmap='inferno',
                      s=6, alpha=0.75, linewidths=0)
    cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.85)
    cbar4.set_label('Surprise (z-score)', color=FG, fontsize=7)
    cbar4.ax.yaxis.set_tick_params(labelsize=6, color=FG)

    # Draw S-U threshold as horizontal line in alignment dimension
    ax4.axhline(0, color=GOLD, lw=1.0, ls='--', alpha=0.6,
                label='S-U plane (alignment=0)')
    ax4.fill_between(ax4.get_xlim() if ax4.get_xlim() != (0, 1) else [-3, 3],
                     -3, 0, alpha=0.04, color=ROSE,
                     label='Below S-U: noise dominates')
    ax4.fill_between([-3, 3], 0, 3, alpha=0.04, color=GREEN,
                     label='Above S-U: structure emerges')

    ax4.set_xlabel("Campo (z-score)", fontsize=9)
    ax4.set_ylabel("Alignment (z-score)", fontsize=9)
    ax4.set_title("State Space Projection\nCampo × Alignment  [colored by Surprise]\n"
                  "Gold line = S-U structural boundary",
                  color=FG, fontsize=9)
    ax4.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none', loc='lower right')
    ax4.grid(True, alpha=0.15)
    ax4.tick_params(labelsize=7)
    ax4.set_xlim(-3.5, 3.5)
    ax4.set_ylim(-3.5, 3.5)


# ── Coherence summary text (right side of panel 1 space, inlined) ─────────────
# Place as text on the figure rather than a separate axes
vol_early = snaps.get('early', (0,))[0]
vol_final = snaps.get('final', (0,))[0]
delta_vol = vol_early - vol_final
pct_vol   = 100 * delta_vol / (vol_early + 1e-8)
mean_surp = r['mean_surprise']
fv        = np.sqrt(2 * D_diff * tau_meas * dt)
surp_vs_fv = mean_surp / (fv + 1e-8)

coherence_achieved = (
    len(r['roll_std']) > 0 and
    np.mean(r['roll_std'][-max(len(r['roll_std'])//4, 1):]) < np.std(r['volume_log']) * 0.5
    and mean_surp < fv
)

summary_lines = [
    "SFE-05.3  MANIFOLD WORLD",
    "─" * 30,
    "",
    "State tensor:  R^(512 x 3)",
    "Axes: Campo | Alignment | Surprise",
    "Geometry: Convex Hull 3D",
    "",
    "─" * 30,
    f"lambda       = {lambda_coup}",
    f"sigma_m      = {sigma_m}",
    f"sigma_memory = {sigma_memory}",
    f"N cycles     = {r['n_cycles']}",
    "",
    "─" * 30,
    "MANIFOLD VOLUMES",
    f"  early  = {vol_early:.4f}",
    f"  final  = {vol_final:.4f}",
    f"  delta  = {delta_vol:+.4f} ({pct_vol:+.1f}%)",
    "",
    "COHERENCE METRICS",
    f"  mean surprise   = {mean_surp:.4f}",
    f"  field vol. σ    = {fv:.4f}",
    f"  surp / field    = {surp_vs_fv:.3f}",
    f"  mean alignment  = {r['mean_alignment']:+.4f}",
    "",
    "─" * 30,
    "COHERENCE STATE",
    f"  {'COHERENT' if coherence_achieved else 'EXPLORING'}",
    "",
    "Coherence when:",
    "  hull volume stabilizes",
    "  surprise < field volatility",
    "  points above S-U plane",
]

# Add as text box to figure
fig.text(0.705, 0.47, "\n".join(summary_lines),
         fontsize=7.0, fontfamily='monospace', color=FG, va='top',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0c14',
                   edgecolor=GOLD, linewidth=1.2, alpha=0.96))


plt.savefig(os.path.join(FIG_DIR, 'sfe053_manifold_world.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SFE-05.3  Final Summary")
print("=" * 70)
print()
print("  STATE TENSOR  R^(512 x 3)")
print("    Axis X — Campo:     |J(x,t)|  local flux magnitude")
print("    Axis Y — Alignment: cos(grad_true, grad_perceived)")
print("    Axis Z — Surprise:  |x_true - x_hat|  Kalman innovation")
print()
print("  GEOMETRY: Convex Hull 3D (no axis treated as height)")
snap_order = ['early', 'mid', 'late', 'final']
for lbl in snap_order:
    entry = snaps.get(lbl)
    if entry:
        print(f"    {lbl:8s}  volume={entry[0]:.4f}  area={entry[1]:.4f}")
print()
print("  COHERENCE METRICS")
print(f"    Mean surprise          = {r['mean_surprise']:.4f}")
print(f"    Field volatility σ     = {np.sqrt(2*D_diff*tau_meas*dt):.4f}")
print(f"    Surprise / volatility  = {surp_vs_fv:.3f}  (< 1.0 = anticipating field)")
print(f"    Mean alignment         = {r['mean_alignment']:+.4f}")
print()
print("  EVOLUTION FROM SFE-05.2")
print("    KL divergence (Z axis)  ->  Surprise |innovation|  (Z axis)")
print("    Delaunay 2D projection  ->  Convex Hull 3D (volumetric)")
print("    2D time-series panels   ->  Manifold-only visual world")
print("    Surface area metric     ->  Volume + rolling variance")
print("    S-U as line chart       ->  S-U as structural plane in state space")
print("=" * 70)
