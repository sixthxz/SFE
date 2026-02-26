# -*- coding: utf-8 -*-
"""
SFE-05 — Field Navigation System
==================================
Stochastic Field Engine, revision 05

CONCEPTUAL FRAME:
    SFE-04 and its predecessors were structured as information motors:
    an external controller observes a Brownian particle, applies feedback
    tilt, and attempts to extract net work bounded by the Sagawa-Ueda
    relation. The frame is outside-in — the agent acts on the system.

    SFE-05 inverts the frame. The probability field rho(x,t) is the
    primary object. The particle is not controlled; it is embedded. It
    navigates by weakly coupling to existing probability flux rather than
    injecting an independent force. There is no optimization target.
    The system's output is coherence — how accurately the agent's internal
    model of the field tracks the true field over time.

    This reflects the underlying intuition: the work (or structure) is
    already present in the field. The agent does not create it. It either
    aligns with it or it does not.

PHYSICS:
    True field:
        drho/dt = -d/dx[(F/gamma)*rho] + (kBT/gamma)*d^2rho/dx^2
        Solved via upwind finite differences on a spatial grid.
        External force F = 0 (free diffusion). Field evolves independently.

    Particle trajectory:
        gamma * dx/dt = F_self + xi(t)
        F_self = lambda * J(x,t) / (|J(x,t)| + eps)
        J(x,t) = (F/gamma)*rho(x,t) - D * drho/dx  [probability flux]
        xi ~ N(0, 2*gamma*kBT/dt)  [thermal noise]
        lambda << 1: weak coupling. Noise dominates.

    Kalman filter:
        Internal belief (x_hat, P) updated at each measurement.
        Measurement: z = x + sigma_m * eta,  eta ~ N(0,1)
        Process model: free diffusion (Q_proc = 2*kBT/gamma * dt)

    Perceived field:
        rho_perceived(x) = sum_i w_i * G(x - x_i, sigma_memory)
        Built from particle's position history via Gaussian kernel density.
        Represents the agent's implicit map of where it has been.

MEASUREMENTS (primary outputs):
    Alignment:
        cos(theta) = [grad rho_true(x) . grad rho_perceived(x)] /
                     [|grad rho_true(x)| * |grad rho_perceived(x)|]
        Evaluated at particle position each cycle.
        Measures agreement between true and perceived field topology locally.

    KL divergence:
        D_KL(rho_perceived || rho_true) = integral p*log(p/q) dx
        Global measure of how well perceived field matches true field.

    Actionable information:
        I_act = I_meas * (alignment + 1) / 2
        Fraction of acquired information that was geometrically aligned
        with true field structure. The rest is acquired but not usable.

    Field entropy:
        H[rho] = -integral rho * log(rho) dx
        Tracked for both true and perceived fields over time.

    Sagawa-Ueda horizon (diagnostic):
        Cumulative kBT * I_meas plotted as an upper bound on coherent
        extraction. Not an optimization target. A reference curve.

PARAMETERS (normalized units, kBT=1, gamma=1):
    lambda_coup  : flux coupling strength (default 0.3)
    sigma_m      : measurement noise std (default 0.9)
    sigma_memory : perceived field kernel width (default 1.2)
    dt           : Langevin time step (default 0.01)
    tau_meas     : steps between measurements (default 10)

EVOLUTION FROM SFE-04:
    SFE-04  F_ctrl = F_tilt * sign(x_hat)     [injected tilt, position-based]
    SFE-05  F_self = lambda * J(x)/|J(x)|     [flux coupling, field-driven]

    SFE-04  W_net > 0 as success criterion
    SFE-05  D_KL -> 0 and alignment -> 1 as coherence criteria

    SFE-04  single probability density rho(x,t)
    SFE-05  rho_true(x,t) and rho_perceived(x,t) as distinct objects

    SFE-04  S-U bound as compliance check (violations counted)
    SFE-05  S-U bound as informational horizon (diagnostic reference)
"""

import numpy as np
import matplotlib
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import os
FIG_DIR = '/tmp/sfe05_figs'
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("SFE-05  —  Field Navigation System")
print("=" * 70)
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

print(f"  Grid: [{x_min}, {x_max}], Nx={Nx}")
print(f"  lambda_coupling={lambda_coup}  sigma_m={sigma_m}  sigma_memory={sigma_memory}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# FIELD UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_rho(mu, sigma):
    rho = np.exp(-0.5 * ((x_grid - mu) / sigma)**2)
    norm = np.trapezoid(rho, x_grid)
    return rho / norm

def fp_flux(rho, F_arr):
    """J(x) = (F/gamma)*rho - D*d(rho)/dx"""
    drho = np.gradient(rho, x_grid)
    return (F_arr / gamma) * rho - D_diff * drho

def fp_step(rho, F_arr):
    """Fokker-Planck upwind step."""
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
    rho_new = rho - (dt / dx) * np.diff(total_flux)
    rho_new = np.maximum(rho_new, 0.0)
    norm = np.trapezoid(rho_new, x_grid)
    return rho_new / norm if norm > 1e-12 else rho_new

def field_entropy(rho):
    eps = 1e-12
    safe = np.maximum(rho, eps)
    return -float(np.trapezoid(safe * np.log(safe), x_grid))

def kl_div(rho_p, rho_q):
    """D_KL(P||Q)"""
    eps = 1e-12
    p = np.maximum(rho_p, eps)
    q = np.maximum(rho_q, eps)
    return float(np.trapezoid(p * np.log(p / q), x_grid))


# ═════════════════════════════════════════════════════════════════════════════
# LANGEVIN — particle follows local flux weakly
# ═════════════════════════════════════════════════════════════════════════════

def langevin_step(x, rho_true, F_free, rng):
    J_arr = fp_flux(rho_true, F_free)
    J_at  = float(np.interp(x, x_grid, J_arr))
    eps   = 1e-10
    F_self = lambda_coup * J_at / (abs(J_at) + eps)
    noise_amp = np.sqrt(2 * kBT * gamma)
    xi        = noise_amp * rng.standard_normal()
    dx_       = (F_self / gamma) * dt + xi * np.sqrt(dt) / gamma
    x_new     = float(np.clip(x + dx_, x_min + 0.1, x_max - 0.1))
    dW        = F_self * dx_
    return x_new, J_at, F_self, dW


# ═════════════════════════════════════════════════════════════════════════════
# KALMAN — internal belief
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
        self.x_hat += K * (z - self.x_hat)
        self.P     *= (1 - K)
        return P_prior

    def I_gain(self, P_prior):
        return max(0.5 * np.log2(1.0 + P_prior / sigma_m**2), 0.0)

    def reset(self):
        self.x_hat = 0.0
        self.P     = 2 * kBT / gamma * tau_meas * dt


# ═════════════════════════════════════════════════════════════════════════════
# PERCEIVED FIELD — built from position memory
# ═════════════════════════════════════════════════════════════════════════════

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
        rho = np.zeros(Nx)
        w_tot = sum(self.weights)
        for xp, w in zip(self.samples, self.weights):
            k = np.exp(-0.5 * ((x_grid - xp) / sigma_memory)**2)
            rho += (w / w_tot) * k
        norm = np.trapezoid(rho, x_grid)
        return rho / norm if norm > 1e-12 else rho


# ═════════════════════════════════════════════════════════════════════════════
# ALIGNMENT — cos(grad_true, grad_perceived) at particle position
# ═════════════════════════════════════════════════════════════════════════════

def compute_alignment(x_pos, rho_true, rho_perc):
    g_true = float(np.interp(x_pos, x_grid, np.gradient(rho_true, x_grid)))
    g_perc = float(np.interp(x_pos, x_grid, np.gradient(rho_perc, x_grid)))
    if abs(g_true) < 1e-10 or abs(g_perc) < 1e-10:
        return 0.0
    return float(np.clip((g_true * g_perc) / (abs(g_true) * abs(g_perc)), -1.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION LOOP
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("RUNNING: Field navigation")

def run_navigation(seed=42, N=N, lam=lambda_coup, sm=sigma_m, record_every=40):
    rng    = np.random.default_rng(seed)
    kf     = KalmanOD()
    pf     = PerceivedField(max_samples=300)
    rho_true = gaussian_rho(0.0, 2.0)
    F_free   = np.zeros(Nx)
    x = 0.0
    pf.add(x)

    x_traj = []; xhat_traj = []; J_traj = []; Fself_traj = []
    align_traj = []; I_tot_traj = []; I_act_traj = []
    kl_traj = []; H_true_traj = []; H_perc_traj = []
    su_I_traj = []; su_W_traj = []
    rho_stack = []; rho_perc_stack = []; rho_t_idx = []

    W_cum = I_cum = 0.0
    dW_cycle = I_cycle = I_act_cycle = 0.0
    n_reset = max(N // tau_meas // 8, 1)

    for i in range(N):
        if i % tau_meas == 0:
            if i > 0:
                su_I_traj.append(I_cum)
                su_W_traj.append(W_cum)
                dW_cycle = I_cycle = I_act_cycle = 0.0

            kf.predict_n(tau_meas)
            x_meas  = x + sm * rng.standard_normal()
            P_prior = kf.update(x_meas)
            I_step  = kf.I_gain(P_prior)
            I_cycle += I_step
            I_cum   += kBT * Landauer * I_step

            rho_perc = pf.get_rho()
            alignment = compute_alignment(x, rho_true, rho_perc)
            I_act_step = I_step * (alignment + 1.0) / 2.0
            I_act_cycle += I_act_step

            kl  = kl_div(rho_perc, rho_true)
            H_t = field_entropy(rho_true)
            H_p = field_entropy(rho_perc)

            align_traj.append(alignment)
            I_tot_traj.append(I_step)
            I_act_traj.append(I_act_step)
            kl_traj.append(kl)
            H_true_traj.append(H_t)
            H_perc_traj.append(H_p)

        if (i + 1) % (n_reset * tau_meas) == 0 and i > 0:
            kf.reset()

        x_new, J_at, F_self, dW = langevin_step(x, rho_true, F_free, rng)
        dW_cycle += dW
        W_cum    += max(dW, 0.0)

        rho_true = fp_step(rho_true, F_free)
        pf.add(x_new, w=1.0 / (sm + 0.1))

        x = x_new
        x_traj.append(x)
        xhat_traj.append(kf.x_hat)
        J_traj.append(J_at)
        Fself_traj.append(F_self)

        if i % record_every == 0:
            rho_stack.append(rho_true.copy())
            rho_perc_stack.append(pf.get_rho())
            rho_t_idx.append(i)

    # final cycle
    su_I_traj.append(I_cum)
    su_W_traj.append(W_cum)

    return dict(
        x_traj=np.array(x_traj), xhat_traj=np.array(xhat_traj),
        J_traj=np.array(J_traj), Fself_traj=np.array(Fself_traj),
        align_traj=np.array(align_traj),
        I_tot_traj=np.array(I_tot_traj), I_act_traj=np.array(I_act_traj),
        kl_traj=np.array(kl_traj),
        H_true_traj=np.array(H_true_traj), H_perc_traj=np.array(H_perc_traj),
        su_I_traj=np.array(su_I_traj), su_W_traj=np.array(su_W_traj),
        rho_stack=np.array(rho_stack),
        rho_perc_stack=np.array(rho_perc_stack),
        rho_t_idx=np.array(rho_t_idx),
        n_cycles=N // tau_meas,
    )

print("  Main run...", end='', flush=True)
r = run_navigation(seed=42)
print(f" done. {r['n_cycles']} cycles.")

align  = r['align_traj']
I_tot  = r['I_tot_traj']
I_act  = r['I_act_traj']
kl_arr = r['kl_traj']
H_true = r['H_true_traj']
H_perc = r['H_perc_traj']

print(f"  Mean alignment         = {np.mean(align):+.4f}  (±{np.std(align):.4f})")
print(f"  Mean KL divergence     = {np.mean(kl_arr):.4f}")
print(f"  Actionable fraction    = {np.mean(I_act)/(np.mean(I_tot)+1e-10):.3f}")

# Sweeps
print("  Sweep: lambda...", end='', flush=True)
lambda_vals = np.linspace(0.0, 2.0, 10)
sweep_lam = []
for lv in lambda_vals:
    rv = run_navigation(lam=lv, seed=7, N=5000)
    sweep_lam.append(dict(
        lam=lv, align=float(np.mean(rv['align_traj'])),
        kl=float(np.mean(rv['kl_traj'])),
        act_frac=float(np.mean(rv['I_act_traj'])/(np.mean(rv['I_tot_traj'])+1e-10))
    ))
print(" done.")

print("  Sweep: sigma_m...", end='', flush=True)
sm_vals = np.linspace(0.3, 2.5, 10)
sweep_sm = []
for sv in sm_vals:
    rv = run_navigation(sm=sv, seed=13, N=5000)
    sweep_sm.append(dict(
        sm=sv, align=float(np.mean(rv['align_traj'])),
        kl=float(np.mean(rv['kl_traj'])),
    ))
print(" done.")


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
BLUE   = '#60a5fa'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': '#1e2235', 'text.color': FG,
    'axes.labelcolor': FG, 'xtick.color': '#555870',
    'ytick.color': '#555870', 'grid.color': '#12152a',
    'grid.linewidth': 0.5,
})

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    "SFE-05  ·  Field Navigation System\n"
    "A noisy agent navigating a stochastic field  —  coherence, not extraction",
    fontsize=13, color=GOLD, y=0.998, fontweight='bold'
)
gs = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
              top=0.965, bottom=0.05, left=0.06, right=0.97)

rho_stack      = r['rho_stack']
rho_perc_stack = r['rho_perc_stack']
rho_t_idx      = r['rho_t_idx']
t_arr          = np.array(rho_t_idx) * dt
x_traj         = r['x_traj']
t_full         = np.arange(len(x_traj)) * dt


# ── PANEL 0: 3D true field ρ(x,t) ────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :2], projection='3d')
ax0.set_facecolor(BG)

step_t = max(len(rho_stack) // 120, 1)
step_x = max(Nx // 80, 1)
t_sub  = t_arr[::step_t]
x_sub  = x_grid[::step_x]
Z_sub  = rho_stack[::step_t, ::step_x]
T_m, X_m = np.meshgrid(t_sub, x_sub, indexing='ij')

vmax_ = float(np.percentile(Z_sub, 97))
surf = ax0.plot_surface(
    T_m, X_m, Z_sub, cmap='inferno', alpha=0.80,
    linewidth=0, antialiased=True,
    norm=mcolors.PowerNorm(0.45, vmin=0, vmax=max(vmax_, 1e-6))
)

# Particle path
pt_step = max(len(x_traj) // 500, 1)
t_pt = t_full[::pt_step]
x_pt = x_traj[::pt_step]
z_pt = np.zeros(len(t_pt))
for k_, (tp_, xp_) in enumerate(zip(t_pt, x_pt)):
    ti_ = int(tp_ / dt / max(rho_t_idx[1] - rho_t_idx[0], 1)) if len(rho_t_idx) > 1 else 0
    ti_ = np.clip(ti_, 0, len(rho_stack) - 1)
    z_pt[k_] = float(np.interp(xp_, x_grid[::step_x], rho_stack[ti_, ::step_x])) + 0.001

ax0.plot(t_pt, x_pt, z_pt, color=TEAL, lw=0.9, alpha=0.9)

ax0.set_xlabel("Time", fontsize=8, labelpad=1)
ax0.set_ylabel("Position x", fontsize=8, labelpad=1)
ax0.set_zlabel("rho(x,t)", fontsize=8, labelpad=1)
ax0.set_title("True Field rho(x,t)  [teal = particle path]", color=FG, fontsize=10, pad=3)
ax0.tick_params(labelsize=6)
ax0.xaxis.pane.fill = ax0.yaxis.pane.fill = ax0.zaxis.pane.fill = False
ax0.grid(False)
ax0.view_init(elev=28, azim=-55)


# ── PANEL 1: True vs Perceived snapshots ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])
ns = len(rho_stack)
snaps = [ns // 8, ns // 2, 7 * ns // 8]
scols = [TEAL, GOLD, VIOLET]
slabs = ['early', 'mid', 'late']
for si, sc, sl in zip(snaps, scols, slabs):
    ax1.plot(x_grid, rho_stack[si],      color=sc, lw=2.0, alpha=0.9, label=f'true {sl}')
    ax1.plot(x_grid, rho_perc_stack[si], color=sc, lw=1.1, ls='--', alpha=0.45)
    ti_ = rho_t_idx[si]
    if ti_ < len(x_traj):
        xp = x_traj[ti_]
        yp = float(np.interp(xp, x_grid, rho_stack[si]))
        ax1.scatter([xp], [yp], color=sc, s=35, zorder=8, marker='D')
ax1.text(0.02, 0.97, 'solid=true  dashed=perceived  diamond=particle',
         transform=ax1.transAxes, fontsize=6.5, color='#888', va='top')
ax1.set_xlabel("x", fontsize=9); ax1.set_ylabel("rho(x)", fontsize=9)
ax1.set_title("True vs Perceived Field", color=FG, fontsize=9)
ax1.legend(fontsize=6.5, facecolor='#0d0f18', edgecolor='none')
ax1.grid(True, alpha=0.2)


# ── PANEL 2: KL divergence ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 3])
cyc_t = np.arange(len(kl_arr))
ax2.fill_between(cyc_t, 0, kl_arr, alpha=0.25, color=ROSE)
ax2.plot(cyc_t, kl_arr, color=ROSE, lw=0.8, alpha=0.7)
kl_sm = gaussian_filter1d(kl_arr, sigma=max(len(kl_arr)//30, 3))
ax2.plot(cyc_t, kl_sm, color=GOLD, lw=2.2, label=f'smoothed  mu={np.mean(kl_arr):.3f}')
ax2.axhline(np.mean(kl_arr), color=TEAL, lw=1.0, ls='--')
ax2.set_xlabel("Cycle", fontsize=9); ax2.set_ylabel("D_KL [nats]", fontsize=9)
ax2.set_title("KL Divergence\nPerceived || True", color=FG, fontsize=9)
ax2.legend(fontsize=7, facecolor='#0d0f18', edgecolor='none')
ax2.grid(True, alpha=0.2)


# ── PANEL 3: Alignment trajectory ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
al_sm = gaussian_filter1d(align, sigma=max(len(align)//30, 2))
for ci in range(len(cyc_t) - 1):
    c = GREEN if align[ci] > 0 else ROSE
    ax3.fill_between([cyc_t[ci], cyc_t[ci+1]], [align[ci], align[ci+1]], 0,
                     alpha=0.18, color=c)
ax3.plot(cyc_t, align, color='#333650', lw=0.5, alpha=0.7)
ax3.plot(cyc_t, al_sm, color=TEAL, lw=2.0, label='smoothed alignment')
ax3.axhline(0, color=FG, lw=0.6, ls='--', alpha=0.35)
ax3.axhline(np.mean(align), color=GOLD, lw=1.5,
            label=f'mean={np.mean(align):.3f}  std={np.std(align):.3f}')
ax3.set_xlabel("Measurement Cycle", fontsize=9)
ax3.set_ylabel("cos(grad_true, grad_perceived)", fontsize=9)
ax3.set_title(
    "Alignment — perceived gradient vs true gradient at particle position\n"
    "+1 = reads field correctly  |  0 = blind  |  -1 = inverted map",
    color=FG, fontsize=9)
ax3.legend(fontsize=8, facecolor='#0d0f18', edgecolor='none')
ax3.set_ylim(-1.3, 1.3)
ax3.grid(True, alpha=0.2)
ax3.set_xlim(0, len(align))


# ── PANEL 4: Actionable vs Total I ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.fill_between(cyc_t, 0, I_tot, alpha=0.18, color=VIOLET, label='I_total')
ax4.fill_between(cyc_t, 0, I_act, alpha=0.5,  color=GREEN,  label='I_actionable')
ax4.plot(cyc_t, I_tot, color=VIOLET, lw=0.8, alpha=0.8)
ax4.plot(cyc_t, I_act, color=GREEN, lw=1.2)
frac = np.mean(I_act) / (np.mean(I_tot) + 1e-10)
ax4.set_title(f"Actionable Information\nI_act/I_total = {frac:.3f}", color=FG, fontsize=9)
ax4.set_xlabel("Cycle", fontsize=9); ax4.set_ylabel("bits", fontsize=9)
ax4.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
ax4.grid(True, alpha=0.2)


# ── PANEL 5: Field entropy ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 3])
ax5.plot(cyc_t, H_true, color=GOLD, lw=2.0, label='H[rho_true]')
ax5.plot(cyc_t, H_perc, color=TEAL, lw=1.5, ls='--', label='H[rho_perceived]')
ax5.fill_between(cyc_t, np.minimum(H_true, H_perc), np.maximum(H_true, H_perc),
                 alpha=0.15, color=ROSE, label='entropy gap')
ax5.set_xlabel("Cycle", fontsize=9); ax5.set_ylabel("Entropy [nats]", fontsize=9)
ax5.set_title("Field Entropy\nTrue vs Perceived", color=FG, fontsize=9)
ax5.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
ax5.grid(True, alpha=0.2)


# ── PANEL 6: S-U horizon (diagnostic) ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
su_cyc = np.arange(len(r['su_I_traj']))
su_I   = r['su_I_traj']
su_W   = r['su_W_traj']
ax6.fill_between(su_cyc, 0, su_I, alpha=0.10, color=GOLD)
ax6.plot(su_cyc, su_I, color=GOLD, lw=1.5, label='Sigma kBT*I  (horizon)')
ax6.plot(su_cyc, su_W, color=GREEN, lw=1.5, label='Sigma W_aligned  (diagnostic)')
if su_I[-1] > 0:
    ratio = su_W[-1] / su_I[-1]
    ax6.text(0.97, 0.05, f'W/I_horizon = {ratio:.3f}',
             transform=ax6.transAxes, ha='right', fontsize=9, color=TEAL)
ax6.set_xlabel("Cycle", fontsize=9); ax6.set_ylabel("[kBT]", fontsize=9)
ax6.set_title("S-U Horizon — Diagnostic Only\n(Not a target. The edge of what the field makes available.)",
              color=FG, fontsize=9)
ax6.legend(fontsize=7.5, facecolor='#0d0f18', edgecolor='none')
ax6.grid(True, alpha=0.2)


# ── PANEL 7: Lambda sweep ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
lv_a  = np.array([s['lam']   for s in sweep_lam])
al_a  = np.array([s['align'] for s in sweep_lam])
kl_a  = np.array([s['kl']    for s in sweep_lam])
ax7b  = ax7.twinx()
ax7.plot(lv_a, al_a, 'o-', color=TEAL, lw=2, ms=5, label='alignment')
ax7.axhline(0, color=FG, lw=0.4, alpha=0.3)
ax7.axvline(lambda_coup, color=GOLD, ls=':', lw=1.5)
ax7b.plot(lv_a, kl_a, 's--', color=ROSE, lw=1.5, ms=4, alpha=0.8, label='KL div')
ax7b.set_ylabel("D_KL", color=ROSE, fontsize=8)
ax7b.tick_params(axis='y', labelcolor=ROSE)
ax7.set_xlabel("lambda_coupling", fontsize=9)
ax7.set_ylabel("Mean alignment", color=TEAL, fontsize=9)
ax7.set_title("Coupling Sweep", color=FG, fontsize=9)
ax7.legend(fontsize=7, loc='upper left', facecolor='#0d0f18', edgecolor='none')
ax7b.legend(fontsize=7, loc='upper right', facecolor='#0d0f18', edgecolor='none')
ax7.grid(True, alpha=0.2)


# ── PANEL 8: Summary ─────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 3])
ax8.axis('off')
frac = float(np.mean(I_act)) / (float(np.mean(I_tot)) + 1e-10)
summary = f"""SFE-05  FIELD NAVIGATOR
{'─'*30}

Frame: field-first, inside-out
No objective function.
No extraction target.
Primary output: coherence.

{'─'*30}
lambda_coupling = {lambda_coup}
sigma_m         = {sigma_m}
sigma_memory    = {sigma_memory}
N cycles        = {r['n_cycles']}

{'─'*30}
COHERENCE
  Alignment  = {np.mean(align):+.4f}
           +/- {np.std(align):.4f}
  Stability  = {1.0-np.std(align):.4f}
  KL div mu  = {np.mean(kl_arr):.4f}
  (0 = perceived = true)

INFORMATION
  I_total    = {np.mean(I_tot):.4f} bits/cyc
  I_act      = {np.mean(I_act):.4f} bits/cyc
  Act.frac   = {frac:.3f}

ENTROPY
  H[true]    = {H_true[-1]:.4f} nats
  H[perc]    = {H_perc[-1]:.4f} nats

{'─'*30}
S-U bound: informational
horizon, not target.
"""
ax8.text(0.03, 0.97, summary, transform=ax8.transAxes,
         fontsize=7.5, va='top', fontfamily='monospace', color=FG,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#0a0c14',
                   edgecolor=GOLD, linewidth=1.3, alpha=0.97))

plt.savefig(os.path.join(FIG_DIR, 'sfe05_field_navigator.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print(" done.")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL PRINT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SFE-05  Final Summary")
print("=" * 70)
print()
print("  Field Navigation System — coherence-based, no extraction objective.")
print()
print(f"  COHERENCE")
print(f"    Alignment mu +/- sigma  = {np.mean(align):+.4f} +/- {np.std(align):.4f}")
print(f"    Alignment stability     = {1.0-np.std(align):.4f}")
print(f"    KL(perceived||true) mu  = {np.mean(kl_arr):.4f}")
print()
print(f"  INFORMATION")
print(f"    I_total / cycle         = {np.mean(I_tot):.4f} bits")
print(f"    I_actionable / cycle    = {np.mean(I_act):.4f} bits")
print(f"    Actionable fraction     = {frac:.3f}")
print()
print(f"  FIELD ENTROPY")
print(f"    H[rho_true]  (final)    = {H_true[-1]:.4f} nats")
print(f"    H[rho_perceived] (final)= {H_perc[-1]:.4f} nats")
print()
print("  EVOLUTION FROM SFE-04")
print("    F_ctrl = F_tilt*m_t      ->  F_self = lambda*J(x)/|J(x)|")
print("    W_net as success metric  ->  D_KL + alignment as coherence metrics")
print("    Single rho (FP)          ->  rho_true + rho_perceived")
print("    S-U as violation count   ->  S-U as informational horizon (diagnostic)")
print("    Optimization objective   ->  Coherence tracking")
print("=" * 70)
