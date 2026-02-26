# -*- coding: utf-8 -*-
"""SFE-05.11 — Negative Delta Threshold · Structural Security Check"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import uniform_filter1d
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)
import os; FIG_DIR = '/tmp/sfe0511_figs'; os.makedirs(FIG_DIR, exist_ok=True)

print("="*70); print("SFE-05.11  —  Negative Delta Threshold · Structural Security Check"); print("="*70)

kBT=1.0; gamma=1.0; D_diff=kBT/gamma
dt=0.01; N=20000; tau_meas=10; N_cycles=N//tau_meas
x_min,x_max=-8.0,8.0; Nx=400
x_grid=np.linspace(x_min,x_max,Nx); dx=x_grid[1]-x_grid[0]

lambda_coup_init=0.30; lambda_min=0.05; lambda_max=0.80
sigma_memory=1.20; coherence_var_th=0.15; window_scale=1.0
alpha_attract=0.15; min_events_before_pull=10
n_candidates=5; rehearsal_on=True
dt_coherent=dt; dt_surprised=dt*0.5
BUFFER_N=512; COHERENCE_WIN=40
gate_multiplier=1.0; gate_window=50
anomaly_buffer_size=256; min_core_rate=0.20
x0=0.0; sigma_m=0.40

field_volatility=np.sqrt(2*D_diff*tau_meas*dt)
coh_floor=np.sqrt(2/np.pi)*sigma_m/field_volatility
delta_threshold=-0.05   # CHANGE: was +0.05
gate_ceil=coh_floor+delta_threshold

print(f"\n  sigma_m={sigma_m}  field_vol={field_volatility:.4f}  floor={coh_floor:.4f}")
print(f"  delta_threshold={delta_threshold}  gate: ratio < {gate_ceil:.4f}  (floor{delta_threshold:+})")
print(f"  Free diffusion ratio ~0.744 — cannot sustain < {gate_ceil:.4f}")
print()

# ─── UPDATED PCA CLASSIFIER ───────────────────────────────────────────────────
def classify_geometry(pca_ratio):
    v0,v1,v2=pca_ratio
    if v0>0.70:   return "LINEAR"
    elif v0>0.50: return "PLANAR"
    elif v2>0.25: return "VOLUMETRIC"
    else:         return "PLANAR"

# ─── CLASSES (identical to SFE-05.10 except recorder uses new gate) ───────────
class CircularBuffer:
    def __init__(self,N,cols=3):
        self.N=N; self.cols=cols; self.buf=np.zeros((N,cols),dtype=np.float32); self.filled=self.writes=0
    def push(self,row):
        self.writes+=1; self.buf=np.roll(self.buf,shift=1,axis=0); self.buf[0]=np.array(row,dtype=np.float32); self.filled=min(self.filled+1,self.N)
    def get_numpy(self): return self.buf[:self.filled]
    def normalize(self):
        d=self.get_numpy()
        if len(d)<4: return d,np.zeros(self.cols),np.ones(self.cols)
        mu=d.mean(0); sig=d.std(0)+1e-8; return (d-mu)/sig,mu,sig

class AdaptiveGate:
    def __init__(self,mult,window,floor):
        self.mult=mult; self.window=window; self.floor=floor; self._h=[]; self.threshold=floor
    def update(self,s):
        self._h.append(s)
        if len(self._h)>self.window: self._h.pop(0)
        if len(self._h)>=4: self.threshold=max(float(np.std(self._h))*self.mult,self.floor)
        return self.threshold
    def core_gate(self,s): return s<=self.threshold
    def anomaly_gate(self,s): return s>self.threshold

class KalmanOU:
    def __init__(self,k=0.0): self.k=k; self.x_hat=0.0; self.P=2*kBT/gamma*tau_meas*dt
    def predict_n(self,n):
        if self.k==0.0: self.P+=n*2*kBT/gamma*dt
        else:
            a=np.exp(-self.k*dt*n); sou=kBT/self.k; self.x_hat*=a; self.P=self.P*a**2+sou*(1-a**2)
    def update(self,z):
        Pp=self.P; K=self.P/(self.P+sigma_m**2); inn=z-self.x_hat; self.x_hat+=K*inn; self.P*=(1-K); return Pp,abs(inn)
    def reset(self): self.x_hat=0.0; self.P=2*kBT/gamma*tau_meas*dt

class PerceivedField:
    def __init__(self,mx=300): self.s=[]; self.w=[]; self.mx=mx
    def add(self,xp,w=1.0):
        self.s.append(xp); self.w.append(w)
        if len(self.s)>self.mx: self.s.pop(0); self.w.pop(0)
    def get_rho(self):
        if len(self.s)<2: return _gr(0.,1.)
        rho=np.zeros(Nx); wt=sum(self.w)
        for xp,w in zip(self.s,self.w): rho+=(w/wt)*np.exp(-0.5*((x_grid-xp)/sigma_memory)**2)
        nm=np.trapezoid(rho,x_grid); return rho/nm if nm>1e-12 else rho

def _gr(m,s): r=np.exp(-0.5*((x_grid-m)/s)**2); return r/np.trapezoid(r,x_grid)
def fp_flux(rho,F): return (F/gamma)*rho-D_diff*np.gradient(rho,x_grid)
def fp_step(rho,F,k=0.,x0w=0.):
    Ft=F-k*(x_grid-x0w); N_=len(rho); v=Ft/gamma; df=np.zeros(N_+1); ff=np.zeros(N_+1)
    for i in range(1,N_):
        vf=0.5*(v[i-1]+v[i]); df[i]=vf*rho[i-1] if vf>=0 else vf*rho[i]
        ff[i]=D_diff*(rho[i]-rho[i-1])/dx
    rn=np.maximum(rho-(dt/dx)*np.diff(df-ff),0.); nm=np.trapezoid(rn,x_grid); return rn/nm if nm>1e-12 else rn

def compute_alignment(xp,rt,rp):
    gt=float(np.interp(xp,x_grid,np.gradient(rt,x_grid))); gp=float(np.interp(xp,x_grid,np.gradient(rp,x_grid)))
    if abs(gt)<1e-10 or abs(gp)<1e-10: return 0.
    return float(np.clip(gt*gp/(abs(gt)*abs(gp)),-1.,1.))

def convex_hull_3d(pts):
    if len(pts)<5: return 0.,0.,None
    try: h=ConvexHull(pts); return h.volume,h.area,h
    except QhullError: return 0.,0.,None

def pca_axes(pts):
    if len(pts)<4: return np.array([1.,0.,0.]),np.eye(3)
    c=pts-pts.mean(0); cov=np.cov(c.T)
    if cov.ndim<2: return np.array([1.,0.,0.]),np.eye(3)
    vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=np.maximum(vals[order],0.); vecs=vecs[:,order]
    return vals/(vals.sum()+1e-12),vecs.T

class CoherenceCentroid:
    def __init__(self): self.centroid=np.zeros(3); self.n=0
    def update(self,pos): self.n+=1; self.centroid=((self.n-1)*self.centroid+pos)/self.n; return self.centroid.copy()
    def attraction_force_x(self,c): return alpha_attract*(self.centroid[0]-c) if self.n>0 else 0.
    @property
    def is_ready(self): return self.n>=min_events_before_pull
    @property
    def position(self): return self.centroid.copy()

class CoherenceRecorder:
    def __init__(self,var_th,fvol,window,centroid,floor,dth):
        self.var_th=var_th; self.fvol=fvol; self.window=max(window,4); self.centroid=centroid
        self.events=[]; self._vols=[]; self.floor=floor; self.dth=dth
    def check(self,vol,ms,pts,msr,lam,cycle,core_rate):
        self._vols.append(vol)
        if len(self._vols)>self.window: self._vols.pop(0)
        if len(self._vols)<4: return False,0.,0.,0.
        mv=np.mean(self._vols)+1e-8; rv=float(np.var(self._vols))/(mv**2)
        sr=ms/(self.fvol+1e-10); delta=sr-self.floor
        is_ev=(rv<self.var_th)and(delta<self.dth)and(core_rate>=min_core_rate)
        if is_ev and len(pts)>=5:
            ve,ae,he=convex_hull_3d(pts); re,axe=pca_axes(pts); ctype=classify_geometry(re)
            self.centroid.update(msr)
            self.events.append({'cycle':cycle,'vol':ve,'area':ae,'hull':he,'pts':pts.copy(),
                'pca_ratio':re,'pca_axes':axe,'rel_var':rv,'surp_ratio':sr,'delta':delta,
                'lambda':lam,'centroid':self.centroid.position,'core_rate':core_rate,'ctype':ctype})
        return is_ev,rv,sr,delta
    @property
    def n(self): return len(self.events)
    def mean_vol(self): return float(np.mean([e['vol'] for e in self.events])) if self.events else 0.
    def mean_pca(self):
        if not self.events: return np.ones(3)/3,np.eye(3)
        return (np.array([e['pca_ratio'] for e in self.events]).mean(0),
                np.array([e['pca_axes'] for e in self.events]).mean(0))
    def type_counts(self):
        from collections import Counter; return Counter(e['ctype'] for e in self.events)

# ─── CORE RUN FUNCTION ────────────────────────────────────────────────────────
def run_one(seed=42, k=0.0, verbose=True):
    rng=np.random.default_rng(seed); kf=KalmanOU(k=k); pf=PerceivedField()
    cb=CircularBuffer(BUFFER_N); ab=CircularBuffer(anomaly_buffer_size)
    gate=AdaptiveGate(gate_multiplier,gate_window,floor=field_volatility)
    centroid=CoherenceCentroid()
    wl=max(int(N_cycles*0.5*window_scale),8)
    recorder=CoherenceRecorder(coherence_var_th,field_volatility,wl,centroid,coh_floor,delta_threshold)
    rho=_gr(x0,np.sqrt(kBT/k)) if k>0 else _gr(0.,2.)
    F0=np.zeros(Nx); x=0.; pf.add(x); lam=lambda_coup_init
    sr_log=[]; vol_log=[]; rv_log=[]; delta_log=[]; cr_log=[]
    event_cycles=[]; centroid_hist=[]
    snap_at={N//8:'early',N//2:'mid',7*N//8:'late'}; snap_va={}
    vw=[]; sw=[]; total=0; cw=0; aw=0; cycle=0; nr=max(N_cycles//8,1); cur_sr=2.; campo_last=0.

    for i in range(N):
        if i%tau_meas==0:
            kf.predict_n(tau_meas)
            xm=x+sigma_m*rng.standard_normal(); _,surp=kf.update(xm)
            rp=pf.get_rho(); aln=compute_alignment(x,rho,rp)
            J=fp_flux(rho,F0); campo=float(abs(np.interp(x,x_grid,J))); campo_last=campo
            cur_sr=surp/(field_volatility+1e-10)
            gate.update(surp); total+=1
            if gate.core_gate(surp): cb.push([campo,aln,surp]); cw+=1
            if gate.anomaly_gate(surp): ab.push([campo,aln,surp]); aw+=1
            sr_log.append(cur_sr); sw.append(surp)
            if len(sw)>COHERENCE_WIN: sw.pop(0)
            cycle+=1
            if cycle%COHERENCE_WIN==0 and cb.filled>=5:
                T,_,_=cb.normalize(); v,a,_=convex_hull_3d(T); vol_log.append(v); vw.append(v)
                if len(vw)>wl: vw.pop(0)
                core_rate=cw/max(total,1)
                cr_log.append(core_rate)
                if len(vw)>=3:
                    vm=np.mean(vw); vs=np.std(vw)+1e-8; z=(v-vm)/vs
                    if z>1.: lam=min(lam*1.05,lambda_max)
                    elif z<-0.5: lam=max(lam*0.98,lambda_min)
                ms=float(np.mean(sw)); msr=cb.get_numpy().mean(0)
                is_ev,rv,sr,delta=recorder.check(v,ms,T,msr,lam,cycle,core_rate)
                rv_log.append(rv); delta_log.append(delta)
                if is_ev:
                    event_cycles.append(cycle); centroid_hist.append(centroid.position.copy())
                    if verbose:
                        ev=recorder.events[-1]
                        print(f"      ★ c={cycle:4d} r={ev['surp_ratio']:.4f} "
                              f"δ={ev['delta']:+.4f} vol={ev['vol']:.1f} "
                              f"type={ev['ctype']:10s} core={core_rate:.3f}")
        if (i+1)%(nr*tau_meas)==0 and i>0: kf.reset()
        dt_eff=dt_coherent if cur_sr<1.0 else dt_surprised
        J=fp_flux(rho,F0); Jat=float(np.interp(x,x_grid,J))
        Ff=lam*Jat/(abs(Jat)+1e-10); Fou=-k*(x-x0)
        Fatt=centroid.attraction_force_x(campo_last) if centroid.is_ready else 0.
        Fext=Fatt+Fou
        def step(Fe):
            xi=np.sqrt(2*kBT*gamma)*rng.standard_normal()
            return float(np.clip(x+((Ff+Fe)/gamma)*dt_eff+xi*np.sqrt(dt_eff)/gamma,x_min+.1,x_max-.1))
        xs=step(Fext); bs=abs(xs-kf.x_hat); bx=xs
        if rehearsal_on and n_candidates>1:
            for _ in range(n_candidates-1):
                xc=step(Fext); sc=abs(xc-kf.x_hat)
                if sc<bs: bx=xc; bs=sc
        rho=fp_step(rho,F0,k,x0); pf.add(bx,w=1./(sigma_m+.1)); x=bx
        if i in snap_at:
            T,_,_=cb.normalize(); lbl=snap_at[i]; v,a,h=convex_hull_3d(T); snap_va[lbl]=(v,a,h,T.copy())
    T,_,_=cb.normalize(); v,a,h=convex_hull_3d(T); snap_va['final']=(v,a,h,T.copy())
    crf=cw/max(total,1); arf=aw/max(total,1)
    sr_arr=np.array(sr_log)
    return dict(k=k, mean_sr=float(sr_arr.mean()), min_sr=float(sr_arr.min()),
                mean_delta=float(sr_arr.mean())-coh_floor, min_delta=float(sr_arr.min())-coh_floor,
                n_events=recorder.n, first_event_cycle=event_cycles[0] if event_cycles else None,
                core_rate=crf, anom_rate=arf, event_cycles=event_cycles,
                centroid_final=centroid.position, centroid_n=centroid.n, centroid_hist=centroid_hist,
                events=recorder.events, mean_vol_events=recorder.mean_vol(),
                mean_pca=recorder.mean_pca(), lam_final=lam, gate_thresh_final=gate.threshold,
                type_counts=recorder.type_counts(), snap_va=snap_va, buf_final=T,
                vol_log=np.array(vol_log), rv_log=np.array(rv_log),
                delta_log=np.array(delta_log), sr_log=sr_arr,
                cr_log=np.array(cr_log))

# ─── THREE RUNS ───────────────────────────────────────────────────────────────
runs_spec=[(0.0,"Run 1","free diffusion — must produce ZERO events"),
           (1.0,"Run 2","k=1.0 OU — expected events"),
           (0.1,"Run 3","k=0.1 OU — boundary")]
results=[]
for kv,rl,rd in runs_spec:
    print(f"{'─'*70}"); print(f"{rl}: k={kv}  —  {rd}"); print(f"{'─'*70}")
    r=run_one(seed=42,k=kv,verbose=True); results.append(r)
    tc=r['type_counts']
    print(f"  mean_sr={r['mean_sr']:.4f}  min_sr={r['min_sr']:.4f}  "
          f"mean_δ={r['mean_delta']:+.4f}  min_δ={r['min_delta']:+.4f}")
    print(f"  events={r['n_events']}  core={r['core_rate']:.3f}")
    print(f"  types: LIN={tc.get('LINEAR',0)} PLA={tc.get('PLANAR',0)} VOL={tc.get('VOLUMETRIC',0)}\n")
r1,r2,r3=results

# ─── VALIDATION TABLE ─────────────────────────────────────────────────────────
print("="*70); print("VALIDATION TABLE  (gate: ratio < {:.4f})".format(gate_ceil)); print("="*70)
print(f"\n  floor={coh_floor:.4f}  delta_threshold={delta_threshold}  gate_ceil={gate_ceil:.4f}\n")
print(f"  {'Run':>3}  {'k':>5}  {'mean_r':>7}  {'min_r':>7}  "
      f"{'mean_δ':>7}  {'min_δ':>7}  {'events':>6}  {'LIN':>4}  {'PLA':>4}  {'VOL':>4}  {'core':>6}")
print("  "+"─"*80)
for i,r in enumerate(results,1):
    tc=r['type_counts']
    below_gate = "✓ REACH" if r['min_sr'] < gate_ceil else "✗ ABOVE"
    print(f"  {i:>3}  {r['k']:>5.2f}  {r['mean_sr']:>7.4f}  {r['min_sr']:>7.4f}  "
          f"{r['mean_delta']:>+7.4f}  {r['min_delta']:>+7.4f}  {r['n_events']:>6}  "
          f"{tc.get('LINEAR',0):>4}  {tc.get('PLANAR',0):>4}  {tc.get('VOLUMETRIC',0):>4}  "
          f"{r['core_rate']:>6.3f}  {below_gate}")
print()
print(f"  Gate = {gate_ceil:.4f}  (floor {delta_threshold:+})")
print(f"  Run1 min_r={r1['min_sr']:.4f}  {'REACHES GATE' if r1['min_sr']<gate_ceil else 'DOES NOT REACH'}")
print(f"  Run2 min_r={r2['min_sr']:.4f}  {'REACHES GATE' if r2['min_sr']<gate_ceil else 'DOES NOT REACH'}")
print(f"  Run3 min_r={r3['min_sr']:.4f}  {'REACHES GATE' if r3['min_sr']<gate_ceil else 'DOES NOT REACH'}")

# ─── FIGURES ──────────────────────────────────────────────────────────────────
BG='#07080f'; FG='#dde1ec'; GOLD='#f5c842'; TEAL='#3dd6c8'; VIOLET='#b87aff'
ROSE='#ff5f7e'; GREEN='#4ade80'; AMBER='#fb923c'; COH='#fde68a'; DIM='#1e2235'; WH='#ffffff'

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':DIM,
    'text.color':FG,'axes.labelcolor':FG,'xtick.color':'#555870','ytick.color':'#555870'})

def draw_hull(ax,pts,hull,fc,ec,af=0.10,ae=0.28):
    if hull is None or pts is None: return
    step=max(len(hull.simplices)//300,1)
    poly=Poly3DCollection([pts[s] for s in hull.simplices[::step]],alpha=af,linewidth=0.3)
    poly.set_facecolor(fc); poly.set_edgecolor(ec); ax.add_collection3d(poly)

def style_3d(ax,xl,yl,zl,title):
    ax.set_xlabel(xl,fontsize=8,labelpad=12); ax.set_ylabel(yl,fontsize=8,labelpad=12)
    ax.set_zlabel(zl,fontsize=8,labelpad=12); ax.set_title(title,color=FG,fontsize=8,pad=6)
    ax.tick_params(labelsize=6)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False; p.set_edgecolor(DIM)
    ax.grid(True,alpha=0.09)

# Pick primary display run: r2 if events, else r3, else r1
primary = r2 if r2['n_events']>0 else (r3 if r3['n_events']>0 else r1)
primary_lbl = f"k={primary['k']:.1f}"
ev0=primary['events'][0] if primary['events'] else None
pca_rp,_=primary['mean_pca']
tcc=primary['type_counts']
coh_state='COHERENT' if primary['n_events']>0 and primary['min_sr']<gate_ceil else \
          'EVENTS DETECTED' if primary['n_events']>0 else 'EXPLORING'

print(f"\nRendering manifold world ({primary_lbl})...", end='', flush=True)

fig=plt.figure(figsize=(20,14),facecolor=BG)
fig.subplots_adjust(left=0.06,right=0.97,top=0.93,bottom=0.07,hspace=0.50,wspace=0.40)
gs=GridSpec(2,3,figure=fig,hspace=0.50,wspace=0.40,top=0.93,bottom=0.07,left=0.06,right=0.97)
fig.suptitle(
    f"SFE-05.11  ·  δ_th={delta_threshold}  gate={gate_ceil:.4f}  "
    f"σ_m={sigma_m}  floor={coh_floor:.4f}\n"
    f"Primary: {primary_lbl}  |  state={coh_state}  events={primary['n_events']}  "
    f"min_ratio={primary['min_sr']:.4f}",
    fontsize=11,color=GOLD,fontweight='bold')

ax0=fig.add_subplot(gs[0,:2],projection='3d'); ax0.set_facecolor(BG)
pts_f=primary['buf_final']; _,_,hf=convex_hull_3d(pts_f)
sc=ax0.scatter(pts_f[:,0],pts_f[:,1],pts_f[:,2],c=np.linspace(0,1,len(pts_f)),
               cmap='plasma',s=6,alpha=0.70,linewidths=0)
draw_hull(ax0,pts_f,hf,TEAL,TEAL,0.06,0.18)
for ev_i in primary['events'][:3]: draw_hull(ax0,ev_i['pts'],ev_i['hull'],COH,COH,0.18,0.50)
cen=primary['centroid_final']
if primary['centroid_n']>=min_events_before_pull:
    ax0.scatter(*cen,color=WH,s=180,marker='*',zorder=15)
    cm=pts_f.mean(0); ax0.quiver(*cm,*(cen-cm),color=GOLD,lw=2.0,alpha=0.80,arrow_length_ratio=0.20)
cb_=plt.colorbar(sc,ax=ax0,shrink=0.40,pad=0.04); cb_.set_label('Time',color=FG,fontsize=7); cb_.ax.tick_params(labelsize=6)
style_3d(ax0,"Campo |J|","Alignment","Surprise",
         f"Core Buffer {primary_lbl}  events={primary['n_events']}  types={dict(tcc)}\n"
         f"ratio={primary['mean_sr']:.3f}  min={primary['min_sr']:.3f}  λ={primary['lam_final']:.3f}")
ax0.view_init(elev=22,azim=-52)

snaps=primary['snap_va']
ax1=fig.add_subplot(gs[0,2],projection='3d'); ax1.set_facecolor(BG)
for lbl,col,af,ae in [('early',ROSE,0.12,0.35),('final',TEAL,0.07,0.20)]:
    e=snaps.get(lbl)
    if e: ax1.scatter(e[3][:,0],e[3][:,1],e[3][:,2],c=col,s=3,alpha=0.35,linewidths=0); draw_hull(ax1,e[3],e[2],col,col,af,ae)
if primary['n_events']>0 and primary['events'][0]['hull'] is not None:
    ev0h=primary['events'][0]; draw_hull(ax1,ev0h['pts'],ev0h['hull'],COH,COH,0.26,0.60)
if len(primary['centroid_hist'])>=2:
    ch=np.array(primary['centroid_hist']); chn=(ch-ch.mean(0))/(ch.std(0)+1e-8)
    ax1.plot(chn[:,0],chn[:,1],chn[:,2],color=WH,lw=1.4,alpha=0.65)
hl=[Line2D([0],[0],marker='o',color='w',markerfacecolor=ROSE,ms=4,label=f"early V={snaps.get('early',(0,))[0]:.2f}"),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=TEAL,ms=4,label=f"final V={snaps.get('final',(0,))[0]:.2f}")]
if primary['n_events']>0:
    ev0h=primary['events'][0]; hl.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COH,ms=7,
                                                  label=f"coh [{ev0h['ctype']}]"))
ax1.legend(handles=hl,fontsize=7,loc='upper left',facecolor='#0d0f18',edgecolor='none')
style_3d(ax1,"Campo","Alignment","Surprise",f"Hull Evolution {primary_lbl}"); ax1.view_init(elev=25,azim=-40)

ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(BG)
va=primary['vol_log']; rv_=primary['rv_log']
if len(va)>0:
    tv=np.arange(len(va))*COHERENCE_WIN; tr2=np.arange(len(rv_))*COHERENCE_WIN
    ax2.fill_between(tv,0,va,alpha=0.15,color=VIOLET); ax2.plot(tv,va,color=VIOLET,lw=1.6,label='Hull vol')
    ax2b=ax2.twinx(); ax2b.set_facecolor(BG)
    ax2b.plot(tr2,rv_,color=GOLD,lw=1.4,ls='--',label='Rel. var')
    ax2b.axhline(coherence_var_th,color=GREEN,lw=1.0,ls=':')
    ax2b.fill_between(tr2,0,rv_,where=(rv_<coherence_var_th),alpha=0.20,color=GREEN)
    ax2b.set_ylabel("Var/mean²",color=GOLD,fontsize=8); ax2b.tick_params(axis='y',labelcolor=GOLD,labelsize=6)
    for ec in primary['event_cycles']: ax2.axvline(ec,color=COH,lw=0.8,alpha=0.55,ls='--')
    ax2.set_xlabel("Cycle",fontsize=8); ax2.set_ylabel("Hull Volume",color=VIOLET,fontsize=8)
    ax2.set_title(f"Volume + Variance  {primary_lbl}",color=FG,fontsize=9)
    ax2.grid(True,alpha=0.10); ax2.tick_params(labelsize=7)

# Panel 3: surprise ratio all 3 + gate line + min_ratio markers
ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(BG)
cyc=np.arange(N_cycles); win=max(N_cycles//40,3)
run_cols=[(r1,TEAL,f'Run1 k=0  r={r1["mean_sr"]:.3f} min={r1["min_sr"]:.3f}'),
          (r2,ROSE,f'Run2 k=1  r={r2["mean_sr"]:.3f} min={r2["min_sr"]:.3f}'),
          (r3,AMBER,f'Run3 k=.1 r={r3["mean_sr"]:.3f} min={r3["min_sr"]:.3f}')]
for rr,col,lbl in run_cols:
    if len(rr['sr_log'])>0:
        sm=uniform_filter1d(rr['sr_log'],size=win); ax3.plot(cyc[:len(sm)],sm,lw=1.6,color=col,label=lbl)
        ax3.axhline(rr['min_sr'],color=col,lw=0.7,ls=':',alpha=0.6)
ax3.axhline(gate_ceil,color=GREEN,lw=1.6,ls='--',label=f'Gate={gate_ceil:.4f}')
ax3.axhline(coh_floor,color=WH,lw=0.9,ls=':',alpha=0.6,label=f'Floor={coh_floor:.4f}')
for ec in primary['event_cycles']: ax3.axvline(ec,color=ROSE,lw=0.6,alpha=0.30,ls='--')
ax3.set_xlabel("Cycle",fontsize=8); ax3.set_ylabel("Surprise / Field Volatility",fontsize=8)
ax3.set_title(f"All Runs  δ_th={delta_threshold}  gate={gate_ceil:.4f}\n(dotted=min_ratio per run)",color=FG,fontsize=9)
ax3.legend(fontsize=6.5,facecolor='#0d0f18',edgecolor='none',loc='upper right')
ax3.grid(True,alpha=0.10); ax3.tick_params(labelsize=7); ax3.set_xlim(0,N_cycles)

# Panel 4: min_ratio diagnostic bar chart
ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(BG)
run_labels=['Run1\nk=0.0','Run2\nk=1.0','Run3\nk=0.1']
mean_vals=[r['mean_sr'] for r in results]; min_vals=[r['min_sr'] for r in results]
xp=np.array([0,1,2]); w=0.3
bars_mean=ax4.bar(xp-w/2,mean_vals,width=w,color=[TEAL,ROSE,AMBER],alpha=0.6,label='mean ratio')
bars_min=ax4.bar(xp+w/2,min_vals,width=w,color=[TEAL,ROSE,AMBER],alpha=1.0,label='min ratio',edgecolor=WH,linewidth=0.8)
ax4.axhline(gate_ceil,color=GREEN,lw=1.6,ls='--',label=f'Gate={gate_ceil:.4f}')
ax4.axhline(coh_floor,color=WH,lw=0.9,ls=':',alpha=0.7,label=f'Floor={coh_floor:.4f}')
for i,(mv,mnv) in enumerate(zip(mean_vals,min_vals)):
    ax4.text(i-w/2,mv+0.002,f'{mv:.3f}',ha='center',fontsize=7,color=FG)
    ax4.text(i+w/2,mnv+0.002,f'{mnv:.3f}',ha='center',fontsize=7,color=FG)
    if mnv < gate_ceil: ax4.text(i+w/2,mnv-0.012,'★REACHES',ha='center',fontsize=6.5,color=COH)
ax4.set_xticks(xp); ax4.set_xticklabels(run_labels,fontsize=8)
ax4.set_ylabel("Surprise / Field Volatility",fontsize=8)
ax4.set_title("Mean vs Min Ratio per Run\n★ = reaches gate threshold",color=FG,fontsize=9)
ax4.legend(fontsize=7,facecolor='#0d0f18',edgecolor='none',loc='upper right')
ax4.grid(True,alpha=0.10); ax4.tick_params(labelsize=7)
ax4.set_ylim(min(min_vals)-0.05, max(mean_vals)+0.05)

# Summary
tc1_=r1['type_counts']; tc2_=r2['type_counts']; tc3_=r3['type_counts']
slines=[
    "SFE-05.11  Negative δ Gate","─"*30,
    f"floor  = {coh_floor:.4f}",
    f"δ_th   = {delta_threshold}",
    f"gate   = {gate_ceil:.4f}  (strict)",
    f"σ_m    = {sigma_m}","",
    "Run  k   mean_r  min_r  evts",
    f"1  0.0  {r1['mean_sr']:.3f}  {r1['min_sr']:.3f}  {r1['n_events']:>4}",
    f"2  1.0  {r2['mean_sr']:.3f}  {r2['min_sr']:.3f}  {r2['n_events']:>4}",
    f"3  0.1  {r3['mean_sr']:.3f}  {r3['min_sr']:.3f}  {r3['n_events']:>4}","",
    "GATE REACH",
    f"R1 min={'✓' if r1['min_sr']<gate_ceil else '✗'}  ({r1['min_sr']:.4f})",
    f"R2 min={'✓' if r2['min_sr']<gate_ceil else '✗'}  ({r2['min_sr']:.4f})",
    f"R3 min={'✓' if r3['min_sr']<gate_ceil else '✗'}  ({r3['min_sr']:.4f})","",
    "STRUCT TAGS",
    f"R1: L={tc1_.get('LINEAR',0)} P={tc1_.get('PLANAR',0)} V={tc1_.get('VOLUMETRIC',0)}",
    f"R2: L={tc2_.get('LINEAR',0)} P={tc2_.get('PLANAR',0)} V={tc2_.get('VOLUMETRIC',0)}",
    f"R3: L={tc3_.get('LINEAR',0)} P={tc3_.get('PLANAR',0)} V={tc3_.get('VOLUMETRIC',0)}",
]
if ev0:
    slines+=["","FIRST COH",
             f"  c={primary['first_event_cycle']}",
             f"  r={ev0['surp_ratio']:.4f}",
             f"  δ={ev0['delta']:+.4f}",
             f"  type={ev0['ctype']}",
             f"  vol={ev0['vol']:.2f}"]
slines+=["","─"*30,f"STATE: {coh_state}"]
ax_sum=fig.add_axes([0.698,0.07,0.265,0.37]); ax_sum.axis('off')
ax_sum.text(0.04,0.97,"\n".join(slines),transform=ax_sum.transAxes,fontsize=7.5,
            fontfamily='monospace',color=FG,va='top',linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.5',facecolor='#0a0c14',edgecolor=GOLD,linewidth=1.3))

plt.savefig(os.path.join(FIG_DIR,'sfe0511_manifold.png'),dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print(" done.")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n"+"="*70); print("SFE-05.11  Final Summary"); print("="*70)
print(f"\n  gate = {gate_ceil:.4f}  (floor={coh_floor:.4f}  δ_th={delta_threshold})")
print(f"\n  {'Run':>3}  {'k':>5}  {'mean_r':>7}  {'min_r':>7}  {'mean_δ':>7}  {'min_δ':>7}  {'events':>6}  {'REACH?':>6}  {'core':>6}")
print("  "+"─"*72)
for i,r in enumerate(results,1):
    tc=r['type_counts']; reach='✓YES' if r['min_sr']<gate_ceil else '✗NO'
    print(f"  {i:>3}  {r['k']:>5.2f}  {r['mean_sr']:>7.4f}  {r['min_sr']:>7.4f}  "
          f"{r['mean_delta']:>+7.4f}  {r['min_delta']:>+7.4f}  {r['n_events']:>6}  "
          f"{reach:>6}  {r['core_rate']:>6.3f}")
print()
for i,r in enumerate(results,1):
    tc=r['type_counts']; dom=max(tc,key=tc.get) if tc else '—'
    print(f"  Run{i} k={r['k']:.1f}: dom={dom:10s}  {dict(tc)}")
print()
print("  SECURITY CHECK:")
r1_pass = r1['n_events']==0
r2_pass = r2['n_events']>0
print(f"    Run1 zero events:  {'PASS ✓' if r1_pass else 'FAIL ✗'}  (events={r1['n_events']})")
print(f"    Run2 has events:   {'PASS ✓' if r2_pass else 'FAIL ✗'}  (events={r2['n_events']})")
if not r2_pass:
    print(f"\n    Run2 min_ratio = {r2['min_sr']:.4f}  gate = {gate_ceil:.4f}")
    if r2['min_sr'] < gate_ceil:
        print(f"    System reaches gate transiently but hull_var condition not met simultaneously.")
        print(f"    Diagnosis: relax coherence_var_th or increase k.")
    else:
        margin = r2['min_sr'] - gate_ceil
        print(f"    System never reaches gate. Margin = +{margin:.4f}")
        print(f"    Options: reduce sigma_m below {sigma_m:.2f}  OR  increase k > 1.0")
        print(f"    Recommended: try sigma_m=0.20 OR k=5.0")
print()
print("  LINEAGE")
print(f"    05.9   σ_m=0.40  k=1.0   ratio=0.746  events=1 (gate ratio<1.0)")
print(f"    05.10  δ_th=+0.05 gate={0.7136+0.05:.4f}  Run1=31 Run2=0 (gate inverted)")
print(f"    05.11  δ_th={delta_threshold}  gate={gate_ceil:.4f}  Run1={r1['n_events']}  Run2={r2['n_events']}")
print("="*70)
