# -*- coding: utf-8 -*-
"""
SFE-05.2 — Field Navigation System · Tensor Architecture
==========================================================
Stochastic Field Engine, revision 05.2

ARCHITECTURAL EVOLUTION FROM SFE-05.1:
    SFE-05.1 stored simulation variables in sequential Python lists,
    producing separate scalar time series analyzed in isolated 2D panels.
    Relations between Campo, Alignment, and Entropy were only visible
    after post-hoc comparison across plots, and memory cost grew linearly.

    SFE-05.2 introduces a multidimensional state tensor architecture:
    a static circular buffer in R^(N x 3) that integrates the three
    state variables into a single geometric object. Each row is a point
    in a 3D state space. The simulation history becomes a Point Cloud,
    and after z-score normalization, a Delaunay surface (manifold) is
    fitted to reveal the latent topology of the agent's navigation.

DATA STRUCTURE — Circular Buffer:
    buffer in R^(N x 3),  columns: [Campo, Alignment, Entropy]
    N  = temporal depth (observation window)
    At each cycle t:
        buffer[1:] = buffer[:-1]   (shift: O(N) in numpy, O(1) with indexing)
        buffer[0]  = [campo_t, align_t, kl_t]
    Memory footprint: constant, independent of total simulation length.

NORMALIZATION — Z-Score per column:
    T_hat = (T - mu) / sigma
    mu    = column-wise mean vector
    sigma = column-wise standard deviation
    Effect: centers origin at (0,0,0), axes in units of std deviation.
    Prevents any single variable's absolute range from dominating geometry.

GEOMETRY:
    Point Cloud:
        Each row of T_hat maps to P = (x, y, z) in R^3.
        Dispersion indicates stochastic regime.
        Clustering toward low-dimensional subsets indicates coherence.

    Manifold Surface:
        Delaunay triangulation on the normalized point cloud.
        Projects onto (x,y) plane for triangulation, uses z as height.
        Irregular topology with long edges: low structure phase.
        Convergent, compact surface: coherence phase.
        Normalized surface area is a quantitative coherence indicator.

STATE VECTOR MAPPING:
    Axis X  — Campo:     local probability flux magnitude seen by particle
    Axis Y  — Alignment: cos(grad_rho_true, grad_rho_perceived) in [-1, 1]
    Axis Z  — Entropy:   D_KL(rho_perceived || rho_true)  [nats]

PHYSICS (inherited from SFE-05.1, unchanged):
    True field:   drho/dt = D * d^2rho/dx^2  (free diffusion, Fokker-Planck)
    Particle:     gamma*dx/dt = lambda*J(x)/|J(x)| + xi(t)
    Kalman:       (x_hat, P) updated at each measurement cycle
    Perceived:    rho_perceived = KDE from position history
    Alignment:    cos(grad_rho_true, grad_rho_perceived) at particle position
    S-U horizon:  cumulative kBT*I_meas (diagnostic reference, not target)
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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import os
FIG_DIR = '/tmp/sfe052_figs'
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05.2  —  Field Navigation System  ·  Tensor Architecture")
print("=" * 70)
print(f"  Backend: {'PyTorch ' + torch.__version__ if TORCH else 'NumPy (torch not found)'}")
print()


# ─── Units ───────────────────────────────────────────────────────────────────
kBT      = 1.0
gamma    = 1.0
D_diff   = kBT / gamma
Landauer = np.log(2)

# ─── Simulation parameters ───────────────────────────────────────────────────
dt         = 0.01
N          = 20000
tau_meas   = 10
N_cycles   = N // tau_meas

# ─── Field grid ──────────────────────────────────────────────────────────────
x_min, x_max = -8.0, 8.0
Nx           = 400
x_grid       = np.linspace(x_min, x_max, Nx)
dx           = x_grid[1] - x_grid[0]

# ─── Navigation parameters ───────────────────────────────────────────────────
lambda_coup  = 0.3
sigma_m      = 0.9
sigma_memory = 1.2

# ─── Circular buffer parameters ──────────────────────────────────────────────
BUFFER_N     = 512       # temporal depth — observation window
#   columns: [campo, alignment, kl_divergence]
#   campo = |J(x_particle)| — local flux magnitude

print(f"  Grid: [{x_min}, {x_max}], Nx={Nx}")
print(f"  lambda={lambda_coup}  sigma_m={sigma_m}  sigma_memory={sigma_memory}")
print(f"  Circular buffer: N={BUFFER_N}, shape=({BUFFER_N}, 3)")
print()


# ═════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER (PyTorch tensor or NumPy array)
# ═════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    """
    Static memory block in R^(N x 3).
    Columns: [campo, alignment, kl_divergence]
    Insertion: O(N) shift — acceptable at N=512.
    In PyTorch this becomes a pure tensor roll operation.
    """
    def __init__(self, N, cols=3):
        self.N = N
        if TORCH:
            self.buf = torch.zeros(N, cols, dtype=torch.float32)
        else:
            self.buf = np.zeros((N, cols), dtype=np.float32)
        self.filled = 0

    def push(self, row):
        """Insert new state vector at index 0, shift history down."""
        if TORCH:
            self.buf = torch.roll(self.buf, shifts=1, dims=0)
            self.buf[0] = torch.tensor(row, dtype=torch.float32)
        else:
            self.buf = np.roll(self.buf, shift=1, axis=0)
            self.buf[0] = np.array(row, dtype=np.float32)
        self.filled = min(self.filled + 1, self.N)

    def get_numpy(self):
        """Return filled portion as numpy array."""
        if TORCH:
            return self.buf[:self.filled].numpy()
        return self.buf[:self.filled]

    def normalize(self):
        """
        Z-score normalization per column.
        Returns normalized numpy array and (mu, sigma) vectors.
        T_hat = (T - mu) / sigma
        """
        data = self.get_numpy()
        if len(data) < 2:
            return data, np.zeros(3), np.ones(3)
        if TORCH:
            T = self.buf[:self.filled]
            mu    = T.mean(dim=0)
            sigma = T.std(dim=0) + 1e-8
            T_hat = ((T - mu) / sigma).numpy()
            return T_hat, mu.numpy(), sigma.numpy()
        else:
            mu    = data.mean(axis=0)
            sigma = data.std(axis=0) + 1e-8
            T_hat = (data - mu) / sigma
            return T_hat, mu, sigma


# ═════════════════════════════════════════════════════════════════════════════
# FIELD PHYSICS (inherited from SFE-05.1)
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma)**2)
    return rho / np.trapezoid(rho, x_grid)

def fp_flux(rho, F_arr):
    drho = np.gradient(rho, x_grid)
    return (F_arr / gamma) * rho - D_diff * drho

def fp_step(rho, F_arr):
    N_  = len(rho)
    v   = F_arr / gamma
    drift_flux = np.zeros(N_ + 1)
    for i in range(1, N_):
        vf = 0.5 * (v[i-1] + v[i])
        drift_flux[i] = vf * rho[i-1] if vf >= 0 else vf * rho[i]
    diff_flux = np.zeros(N_ + 1)
    for i in range(1, N_):
        diff_flux[i] = D_diff * (rho[i] - rho[i-1]) / dx
    total_flux = drift_flux - diff_flux
    rho_new = np.maximum(rho - (dt / dx) * np.diff(total_flux), 0.0)
    norm = np.trapezoid(rho_new, x_grid)
    return rho_new / norm if norm > 1e-12 else rho_new

def field_entropy(rho):
    safe = np.maximum(rho, 1e-12)
    return -float(np.trapezoid(safe * np.log(safe), x_grid))

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
    x_new  = float(np.clip(x + dx_, x_min + 0.1, x_max - 0.1))
    return x_new, abs(J_at), F_self


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN + PERCEIVED FIELD (inherited)
# ═════════════════════════════════════════════════════════════════════════════

class KalmanOD:
    def __init__(self):
        self.x_hat = 0.0
        self.P     = 2 * kBT / gamma * tau_meas * dt
        self.Q     = 2 * kBT / gamma * dt

    def predict_n(self, n):  self.P += n * self.Q

    def update(self, z):
        P_prior    = self.P
        K          = self.P / (self.P + sigma_m**2)
        self.x_hat += K * (z - self.x_hat)
        self.P     *= (1 - K)
        return P_prior

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

    def add(self, x_pos, w=1.0):
        self.samples.append(x_pos)
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
    g_true = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_perc = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_true) < 1e-10 or abs(g_perc) < 1e-10:
        return 0.0
    return float(np.clip(g_true * g_perc / (abs(g_true) * abs(g_perc)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# MANIFOLD UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def delaunay_surface_area(pts_normalized):
    """
    Compute Delaunay triangulation on (x,y) projection, sum triangle areas in 3D.
    Normalized surface area is a coherence indicator:
        high area  = dispersed, irregular topology
        low area   = compact, converged manifold
    """
    if len(pts_normalized) < 4:
        return 0.0, None, None
    xy  = pts_normalized[:, :2]
    z   = pts_normalized[:, 2]
    try:
        tri = Delaunay(xy)
    except Exception:
        return 0.0, None, None

    simplices = tri.simplices
    total_area = 0.0
    for s in simplices:
        p0, p1, p2 = pts_normalized[s[0]], pts_normalized[s[1]], pts_normalized[s[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        total_area += 0.5 * np.linalg.norm(cross)

    return total_area, tri, pts_normalized


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION — Navigation with circular buffer
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Field navigation with circular buffer")

def run_navigation(seed=42, N=N, record_every=40):
    rng      = np.random.default_rng(seed)
    kf       = KalmanOD()
    pf       = PerceivedField(max_samples=300)
    cbuf     = CircularBuffer(BUFFER_N, cols=3)

    rho_true = gaussian_rho(0.0, 2.0)
    F_free   = np.zeros(Nx)
    x        = 0.0
    pf.add(x)

    # Per-cycle scalar logs (kept for 2D diagnostic panels)
    align_log  = []
    I_tot_log  = []
    I_act_log  = []
    kl_log     = []
    H_true_log = []
    H_perc_log = []
    su_I_log   = []
    su_W_log   = []
    campo_log  = []

    # Snapshots for 3D field panel
    rho_stack      = []
    rho_perc_stack = []
    rho_t_idx      = []
    x_traj         = []

    # Buffer snapshots at key time points (for manifold rendering)
    buf_snapshots  = {}   # { label: normalized_array }
    snap_at        = {N//8: 'early', N//2: 'mid', 7*N//8: 'late'}

    W_cum = I_cum = 0.0
    n_reset = max(N // tau_meas // 8, 1)

    for i in range(N):
        if i % tau_meas == 0:
            kf.predict_n(tau_meas)
            x_meas  = x + sigma_m * rng.standard_normal()
            P_prior = kf.update(x_meas)
            I_step  = kf.I_gain(P_prior)
            I_cum  += kBT * Landauer * I_step

            rho_perc  = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)
            I_act     = I_step * (alignment + 1.0) / 2.0
            kl        = kl_div(rho_perc, rho_true)
            H_t       = field_entropy(rho_true)
            H_p       = field_entropy(rho_perc)

            # State vector pushed into circular buffer
            J_arr   = fp_flux(rho_true, F_free)
            J_at    = float(abs(np.interp(x, x_grid, J_arr)))
            cbuf.push([J_at, alignment, kl])

            align_log.append(alignment)
            I_tot_log.append(I_step)
            I_act_log.append(I_act)
            kl_log.append(kl)
            H_true_log.append(H_t)
            H_perc_log.append(H_p)
            campo_log.append(J_at)
            su_I_log.append(I_cum)
            su_W_log.append(W_cum)

        if (i + 1) % (n_reset * tau_meas) == 0 and i > 0:
            kf.reset()

        x_new, J_mag, F_self = langevin_step(x, rho_true, F_free, rng)
        W_cum += max(F_self * (x_new - x), 0.0)

        rho_true = fp_step(rho_true, F_free)
        pf.add(x_new, w=1.0 / (sigma_m + 0.1))
        x = x_new
        x_traj.append(x)

        if i % record_every == 0:
            rho_stack.append(rho_true.copy())
            rho_perc_stack.append(pf.get_rho())
            rho_t_idx.append(i)

        # Capture buffer snapshot
        if i in snap_at:
            T_hat, mu_, sigma_ = cbuf.normalize()
            buf_snapshots[snap_at[i]] = T_hat.copy()

    # Final buffer snapshot
    T_hat_final, mu_final, sigma_final = cbuf.normalize()
    buf_snapshots['final'] = T_hat_final.copy()

    return dict(
        x_traj        = np.array(x_traj),
        align_log     = np.array(align_log),
        I_tot_log     = np.array(I_tot_log),
        I_act_log     = np.array(I_act_log),
        kl_log        = np.array(kl_log),
        H_true_log    = np.array(H_true_log),
        H_perc_log    = np.array(H_perc_log),
        campo_log     = np.array(campo_log),
        su_I_log      = np.array(su_I_log),
        su_W_log      = np.array(su_W_log),
        rho_stack     = np.array(rho_stack),
        rho_perc_stack= np.array(rho_perc_stack),
        rho_t_idx     = np.array(rho_t_idx),
        buf_snapshots = buf_snapshots,
        buf_final     = T_hat_final,
        mu_final      = mu_final,
        sigma_final   = sigma_final,
        n_cycles      = N // tau_meas,
    )

print("  Main run...", end='', flush=True)
r = run_navigation(seed=42)
print(f" done. {r['n_cycles']} cycles.")

align     = r['align_log']
kl_arr    = r['kl_log']
I_tot     = r['I_tot_log']
I_act     = r['I_act_log']
H_true    = r['H_true_log']
H_perc    = r['H_perc_log']
campo     = r['campo_log']
buf_snaps = r['buf_snapshots']
buf_final = r['buf_final']

frac = float(np.mean(I_act)) / (float(np.mean(I_tot)) + 1e-10)

print(f"  Mean alignment         = {np.mean(align):+.4f}  (±{np.std(align):.4f})")
print(f"  Mean KL divergence     = {np.mean(kl_arr):.4f}")
print(f"  Actionable fraction    = {frac:.3f}")
print(f"  Buffer shape (final)   = {buf_final.shape}")

# Compute manifold surface areas at each snapshot
print("  Computing manifold surface areas...", end='', flush=True)
manifold_data = {}
for label, pts in buf_snaps.items():
    area, tri, pts_used = delaunay_surface_area(pts)
    manifold_data[label] = {'area': area, 'tri': tri, 'pts': pts_used}
    print(f"\n    {label:6s}: {len(pts)} pts, area = {area:.3f}", end='')
print("\n  done.")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
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

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': '#1e2235', 'text.color': FG,
    'axes.labelcolor': FG, 'xtick.color': '#555870',
    'ytick.color': '#555870', 'grid.color': '#12152a',
    'grid.linewidth': 0.5,
})

fig = plt.figure(figsize=(24, 20), facecolor=BG)
fig.suptitle(
    "SFE-05.2  ·  Field Navigation System  ·  Tensor Architecture\n"
    "Circular Buffer  →  Point Cloud  →  Z-Score Normalization  →  Manifold Surface",
    fontsize=12, color=GOLD, y=0.999, fontweight='bold'
)
gs = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40,
              top=0.968, bottom=0.05, left=0.06, right=0.97)

rho_stack  = r['rho_stack']
rho_t_idx  = r['rho_t_idx']
t_arr      = np.array(rho_t_idx) * dt
x_traj     = r['x_traj']
t_full     = np.arange(len(x_traj)) * dt
cyc_t      = np.arange(len(align))


# ── PANEL 0: 3D Point Cloud — final buffer state (normalized) ────────────────
ax0 = fig.add_subplot(gs[0, :2], projection='3d')
ax0.set_facecolor(BG)

pts = buf_final  # shape (BUFFER_N, 3), z-score normalized

# Color by time index (older = darker, newer = brighter)
n_pts   = len(pts)
t_color = np.linspace(0, 1, n_pts)  # 0=oldest, 1=newest

sc = ax0.scatter(
    pts[:, 0], pts[:, 1], pts[:, 2],
    c=t_color, cmap='plasma',
    s=8, alpha=0.75, linewidths=0
)

# Connect consecutive points with faint lines (trajectory in state space)
step_line = max(n_pts // 200, 1)
for k in range(0, n_pts - step_line, step_line):
    alpha_ = 0.08 + 0.3 * t_color[k]
    ax0.plot(
        pts[k:k+step_line+1, 0],
        pts[k:k+step_line+1, 1],
        pts[k:k+step_line+1, 2],
        color=TEAL, lw=0.5, alpha=alpha_
    )

cbar = plt.colorbar(sc, ax=ax0, shrink=0.55, pad=0.02)
cbar.set_label('Time (oldest→newest)', color=FG, fontsize=7)
cbar.ax.yaxis.set_tick_params(color=FG, labelsize=6)

ax0.set_xlabel("Campo  |J(x)|", fontsize=8, labelpad=2)
ax0.set_ylabel("Alignment", fontsize=8, labelpad=2)
ax0.set_zlabel("Entropy (KL)", fontsize=8, labelpad=2)
ax0.set_title(
    f"Point Cloud — State Space (z-score normalized)\n"
    f"Buffer: {n_pts} pts  |  Axes in σ units",
    color=FG, fontsize=10, pad=4
)
ax0.tick_params(labelsize=6)
ax0.xaxis.pane.fill = ax0.yaxis.pane.fill = ax0.zaxis.pane.fill = False
ax0.grid(False)
ax0.view_init(elev=22, azim=-50)


# ── PANEL 1: Manifold surface — early vs final ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2], projection='3d')
ax1.set_facecolor(BG)

snap_colors = {'early': ROSE, 'final': TEAL}
snap_labels = {'early': f"Early  area={manifold_data.get('early',{}).get('area',0):.2f}",
               'final': f"Final  area={manifold_data.get('final',{}).get('area',0):.2f}"}

for label, color in snap_colors.items():
    md = manifold_data.get(label)
    if md is None or md['tri'] is None:
        continue
    pts_s = md['pts']
    tri_s = md['tri']

    # Draw triangulated surface
    verts = []
    for s in tri_s.simplices[::max(len(tri_s.simplices)//300, 1)]:
        triangle = pts_s[s]
        verts.append(triangle)

    poly = Poly3DCollection(verts, alpha=0.12, linewidth=0.3)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax1.add_collection3d(poly)

    ax1.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                c=color, s=4, alpha=0.5, linewidths=0)

ax1.set_xlabel("Campo", fontsize=7, labelpad=1)
ax1.set_ylabel("Alignment", fontsize=7, labelpad=1)
ax1.set_zlabel("Entropy", fontsize=7, labelpad=1)
ax1.set_title(
    "Manifold Surface\n"
    f"Rose=early  Teal=final",
    color=FG, fontsize=9, pad=3
)
ax1.tick_params(labelsize=5)
ax1.xaxis.pane.fill = ax1.yaxis.pane.fill = ax1.zaxis.pane.fill = False
ax1.grid(False)
ax1.view_init(elev=25, azim=-45)


# ── PANEL 2: Surface area over time ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 3])

snap_order  = ['early', 'mid', 'late', 'final']
snap_t      = [N//8, N//2, 7*N//8, N-1]
snap_areas  = [manifold_data.get(k, {}).get('area', 0) for k in snap_order]
snap_labels_short = ['early', 'mid', 'late', 'final']
snap_c      = [ROSE, AMBER, VIOLET, TEAL]

ax2.bar(snap_labels_short, snap_areas, color=snap_c, alpha=0.8, edgecolor='none')
for xi_, (lbl, area) in enumerate(zip(snap_labels_short, snap_areas)):
    ax2.text(xi_, area + max(snap_areas)*0.02, f'{area:.2f}',
             ha='center', fontsize=8, color=FG)

ax2.set_ylabel("Normalized Surface Area", fontsize=9)
ax2.set_title(
    "Manifold Surface Area over Time\n"
    "Decrease = coherence emergence",
    color=FG, fontsize=9
)
ax2.grid(True, axis='y', alpha=0.2)
ax2.set_ylim(0, max(snap_areas) * 1.18 if snap_areas else 1)


# ── PANEL 3: Alignment — full time series ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
al_sm = gaussian_filter1d(align, sigma=max(len(align)//30, 2))
for ci in range(len(cyc_t) - 1):
    c = GREEN if align[ci] > 0 else ROSE
    ax3.fill_between([cyc_t[ci], cyc_t[ci+1]], [align[ci], align[ci+1]], 0,
                     alpha=0.18, color=c)
ax3.plot(cyc_t, align, color='#333650', lw=0.5, alpha=0.6)
ax3.plot(cyc_t, al_sm, color=TEAL, lw=2.0, label='smoothed')
ax3.axhline(0, color=FG, lw=0.6, ls='--', alpha=0.35)
ax3.axhline(np.mean(align), color=GOLD, lw=1.5,
            label=f'μ={np.mean(align):.3f}  σ={np.std(align):.3f}')
ax3.set_xlabel("Measurement Cycle", fontsize=9)
ax3.set_ylabel("cos(∇ρ_true, ∇ρ_perceived)", fontsize=9)
ax3.set_title(
    "Alignment — Axis Y of State Tensor\n"
    "+1 = perceived gradient matches true  |  0 = blind  |  -1 = inverted",
    color=FG, fontsize=9)
ax3.legend(fontsize=8, facecolor='#0d0f18', edgecolor='none')
ax3.set_ylim(-1.3, 1.3)
ax3.grid(True, alpha=0.2)
ax3.set_xlim(0, len(align))


# ── PANEL 4: Campo — Axis X of state tensor ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
campo_sm = gaussian_filter1d(campo, sigma=max(len(campo)//30, 2))
ax4.fill_between(cyc_t, 0, campo, alpha=0.15, color=AMBER)
ax4.plot(cyc_t, campo, color='#444', lw=0.5, alpha=0.5)
ax4.plot(cyc_t, campo_sm, color=AMBER, lw=1.8, label='smoothed')
ax4.axhline(np.mean(campo), color=GOLD, lw=1.2, ls='--',
            label=f'μ={np.mean(campo):.4f}')
ax4.set_xlabel("Cycle", fontsize=9)
ax4.set_ylabel("|J(x)|  [flux magnitude]", fontsize=9)
ax4.set_title("Campo — Axis X of State Tensor\nLocal probability flux at particle", color=FG, fontsize=9)
ax4.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
ax4.grid(True, alpha=0.2)


# ── PANEL 5: KL Divergence — Axis Z of state tensor ──────────────────────────
ax5 = fig.add_subplot(gs[1, 3])
kl_sm = gaussian_filter1d(kl_arr, sigma=max(len(kl_arr)//30, 3))
ax5.fill_between(cyc_t, 0, kl_arr, alpha=0.2, color=ROSE)
ax5.plot(cyc_t, kl_arr, color=ROSE, lw=0.7, alpha=0.6)
ax5.plot(cyc_t, kl_sm, color=GOLD, lw=2.2, label=f'smoothed  μ={np.mean(kl_arr):.3f}')
ax5.axhline(np.mean(kl_arr), color=TEAL, lw=1.0, ls='--')
ax5.set_xlabel("Cycle", fontsize=9)
ax5.set_ylabel("D_KL [nats]", fontsize=9)
ax5.set_title("Entropy (KL) — Axis Z of State Tensor\nD_KL(perceived || true)", color=FG, fontsize=9)
ax5.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')
ax5.grid(True, alpha=0.2)


# ── PANEL 6: S-U horizon (diagnostic) ────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
su_cyc = np.arange(len(r['su_I_log']))
su_I   = r['su_I_log']
su_W   = r['su_W_log']
ax6.fill_between(su_cyc, 0, su_I, alpha=0.10, color=GOLD)
ax6.plot(su_cyc, su_I, color=GOLD, lw=1.5, label='Σ kBT·I  (horizon)')
ax6.plot(su_cyc, su_W, color=GREEN, lw=1.5, label='Σ W_aligned  (diagnostic)')
if len(su_I) > 0 and su_I[-1] > 0:
    ax6.text(0.97, 0.05, f'W / I_horizon = {su_W[-1]/su_I[-1]:.3f}',
             transform=ax6.transAxes, ha='right', fontsize=9, color=TEAL)
ax6.set_xlabel("Cycle", fontsize=9)
ax6.set_ylabel("[kBT]", fontsize=9)
ax6.set_title("S-U Horizon — Diagnostic Reference\n(Not a target. Upper bound on coherent extraction.)",
              color=FG, fontsize=9)
ax6.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
ax6.grid(True, alpha=0.2)


# ── PANEL 7: 2D projection of point cloud (Campo vs Alignment) ───────────────
ax7 = fig.add_subplot(gs[2, 2])
sc7 = ax7.scatter(buf_final[:, 0], buf_final[:, 1],
                  c=buf_final[:, 2], cmap='inferno',
                  s=6, alpha=0.7, linewidths=0)
cbar7 = plt.colorbar(sc7, ax=ax7, shrink=0.8)
cbar7.set_label('Entropy (z-score)', color=FG, fontsize=7)
cbar7.ax.yaxis.set_tick_params(color=FG, labelsize=6)
ax7.set_xlabel("Campo (z-score)", fontsize=9)
ax7.set_ylabel("Alignment (z-score)", fontsize=9)
ax7.set_title("State Space Projection\nCampo × Alignment  [colored by Entropy]",
              color=FG, fontsize=9)
ax7.grid(True, alpha=0.2)


# ── PANEL 8: Summary ─────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 3])
ax8.axis('off')

area_early = manifold_data.get('early', {}).get('area', 0)
area_final = manifold_data.get('final', {}).get('area', 0)
area_delta = area_early - area_final
area_pct   = 100 * area_delta / (area_early + 1e-8)

summary = f"""SFE-05.2  TENSOR ARCHITECTURE
{'─'*32}

Data structure:
  CircularBuffer R^({BUFFER_N}x3)
  Cols: Campo | Align | Entropy
  Normalization: Z-score/column

{'─'*32}
lambda     = {lambda_coup}
sigma_m    = {sigma_m}
sigma_mem  = {sigma_memory}
N cycles   = {r['n_cycles']}
Buffer N   = {BUFFER_N}

{'─'*32}
COHERENCE
  Alignment  = {np.mean(align):+.5f}
           ± {np.std(align):.5f}
  KL div mu  = {np.mean(kl_arr):.4f}
  Act. frac  = {frac:.3f}

MANIFOLD SURFACE AREA
  Early      = {area_early:.3f}
  Final      = {area_final:.3f}
  Delta      = {area_delta:+.3f} ({area_pct:+.1f}%)

{'─'*32}
Backend: {'PyTorch' if TORCH else 'NumPy'}
"""

ax8.text(0.03, 0.97, summary, transform=ax8.transAxes,
         fontsize=7.5, va='top', fontfamily='monospace', color=FG,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#0a0c14',
                   edgecolor=GOLD, linewidth=1.3, alpha=0.97))


plt.savefig(os.path.join(FIG_DIR, 'sfe052_manifold.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SFE-05.2  Final Summary")
print("=" * 70)
print()
print(f"  DATA STRUCTURE")
print(f"    Circular buffer:       R^({BUFFER_N} x 3)  [Campo, Alignment, Entropy]")
print(f"    Normalization:         Z-score per column")
print(f"    Backend:               {'PyTorch' if TORCH else 'NumPy'}")
print()
print(f"  COHERENCE METRICS")
print(f"    Alignment μ ± σ        = {np.mean(align):+.4f} ± {np.std(align):.4f}")
print(f"    KL(perceived||true) μ  = {np.mean(kl_arr):.4f}")
print(f"    Actionable fraction    = {frac:.3f}")
print()
print(f"  MANIFOLD SURFACE AREA (z-score normalized)")
for label in ['early', 'mid', 'late', 'final']:
    a = manifold_data.get(label, {}).get('area', 0)
    print(f"    {label:8s}:  {a:.4f}")
print(f"    Delta (early→final):   {area_delta:+.4f}  ({area_pct:+.1f}%)")
print()
print(f"  EVOLUTION FROM SFE-05.1")
print(f"    Python lists           ->  Circular buffer R^({BUFFER_N}x3)")
print(f"    Scalar time series     ->  3D point cloud + manifold surface")
print(f"    Separate 2D panels     ->  Unified geometric state object")
print(f"    Raw variable scale     ->  Z-score normalization per column")
print(f"    No topology measure    ->  Delaunay surface area as coherence metric")
print("=" * 70)
