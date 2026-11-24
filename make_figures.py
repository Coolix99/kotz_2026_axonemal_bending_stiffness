"""Reproduce Kotz et al. figures 2–4 and S2–S6 from processed CSVs. Run: python make_figures.py"""
from pathlib import Path
import logging
logging.getLogger('fontTools').setLevel(logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
B, K, L, A = 840., 2000., 10., .2   # pN um², pN/um², um, um
DB, DETA, E_ATP = 280., 1., .100   # stiffness SD, relative viscosity uncertainty, pN um/ATP (100 zJ)
FREQ = {'WT': 68.68953759, 'ODA': 29.49269462}
COL = {'WT': '#1744dd', 'ODA': '#e51c23'}
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12, 'axes.labelsize': 12,
                     'axes.titlesize': 15, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'axes.linewidth': .9, 'mathtext.fontset': 'cm', 'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})
for fmt in ('png', 'pdf'):
    (ROOT / f'output_{fmt}').mkdir(exist_ok=True)
    for old in ('Fig1', 'FigS1'):
        (ROOT / f'output_{fmt}' / f'{old}.{fmt}').unlink(missing_ok=True)


def csv(path, header=True):
    return np.loadtxt(DATA / path, delimiter=',', skiprows=int(header))


def calc(gamma, hydro_file, frequency, motor_fraction=1., ds=None):
    nphase, ns = gamma.shape
    ds = L / (ns - 1) if ds is None else ds
    dtgamma = np.gradient(gamma, axis=0) * frequency * nphase  # original MATLAB discretization
    d2 = np.full_like(gamma, np.nan)
    d2[:, 1:-1] = np.diff(gamma, n=2, axis=1) / ds**2
    bend, slide = B*d2, K*A**2*gamma
    hydro = -hydro_file  # CSV stores -H
    passive = bend - hydro
    force = -passive/(A*(17500/L/8)*motor_fraction)
    trust = (passive*dtgamma < 0) & (passive*gamma < 0)
    return dict(gamma=gamma, dt=dtgamma, d2=d2, bend=bend, slide=slide,
                hydro=hydro, passive=passive, force=force, trust=trust, f0=frequency, ds=ds, n=nphase)


def load(g, mode='group', ident=None):
    if mode == 'ideal':
        p = Path(f'idealized_beat_patterns/01_thesis_linear_A_matched/{g}')
        gamma, h = csv(p/'periodic_beat.csv')[:, 1:], csv(p/'hydrodynamic_torque_free.csv')[:, 1:]
    else:
        path = f'{g}/KCl_000mM'
        p = Path('phase_averaged_beat_patterns_individual_axonemes')/path/ident if ident else Path('phase_averaged_beat_patterns')/path
        q = Path('precomputed_hydrodynamic_force_moment')/path/(ident or 'group_average')
        suffix = '' if ident else '_mean'
        dynamic, static = csv(p/f'periodic_avg{suffix}.csv')[:, 1:], csv(p/f'mean_shape{suffix}.csv')[:, 1]
        gamma = dynamic if mode == 'dynamic' else dynamic + static[None, :]
        h = csv(q/'hydrodynamic_torque_free.csv')[:, 1:]
    return calc(gamma, h, FREQ[g], .429 if g == 'ODA' else 1.)


def bound(d, pct=95):
    return np.percentile(abs(d['force'][d['trust']]), pct)


def save(name, fig):
    for fmt in ('png', 'pdf'):
        fig.savefig(ROOT/f'output_{fmt}'/f'{name}.{fmt}', dpi=250, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def tag(ax, letter):
    ax.text(-.13, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=15)


def phasebar(ax):
    """Use the same cyclic HSV legend wherever oscillator phase is an axis."""
    strip = ax.inset_axes([0, -.21, 1, .05], transform=ax.transAxes)
    strip.imshow(np.arange(256)[None, :], cmap='hsv', aspect='auto', extent=(0, 2*np.pi, 0, 1))
    strip.set(xticks=[], yticks=[], xlim=(0, 2*np.pi))
    for spine in strip.spines.values(): spine.set_linewidth(.45)
    ax.set_xlabel(r'$\varphi$', labelpad=27)


def xy(gamma, ds=.2):
    return (np.c_[np.zeros(len(gamma)), np.cumsum(trig(gamma), axis=1)]*ds
            for trig in (np.cos, np.sin))


def scatter(ax, d, g, histogram=False):
    v, f, good = d['dt'], d['force'], d['trust']
    # A single flat series of VECTOR markers fixes the displaced PDF point clouds.
    ax.plot(v.ravel(), f.ravel(), '.', c='.80', ms=2.2, linestyle='none', clip_on=True)
    ax.plot(v[good], f[good], '.', c=COL[g], ms=2.5, linestyle='none', clip_on=True)
    ax.set(xlim=(-40/.06736, 40/.06736), ylim=(-25, 25), xticks=[-500, 0, 500],
           xlabel=r'$\partial_t\gamma$ [rad/s]', ylabel=r'$F_{\rm head}$ [pN]')
    ax.text(.06, .85, g, color=COL[g], transform=ax.transAxes, fontweight='bold', fontsize=15)
    ax.text(.59, .08, r'$\Omega_{\rm trust}$', color=COL[g], transform=ax.transAxes)
    top = ax.secondary_xaxis('top', functions=(lambda x: .06736*x, lambda x: x/.06736))
    top.set_xlabel(r'$\partial_t\Delta_3$ [$\mu$m/s]', color='.45', fontsize=13)
    top.set_xticks([-40, -20, 0, 20, 40]); top.tick_params(colors='.45', labelsize=11)
    if histogram:
        hi = ax.inset_axes([1.04, 0, .23, 1.])
        hi.hist(f[good], bins=np.arange(-25, 26), orientation='horizontal', color=COL[g], alpha=.83)
        for pct, style in ((95, '-'), (98, ':')):
            hi.axhline(bound(d, pct), c=COL[g], ls=style, lw=1)
            hi.axhline(-bound(d, pct), c=COL[g], ls=style, lw=1)
        hi.set(ylim=ax.get_ylim(), xticks=[], yticks=[])
        hi.spines[['top', 'right', 'bottom']].set_visible(False)


def heat(ax, d, vmin=-20, vmax=20):
    z, good = d['force'], d['trust']
    im = ax.imshow(z.T, extent=(0, 2*np.pi, 0, (z.shape[1]-1)*d['ds']), origin='lower',
                   aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='nearest')
    phi, s = np.meshgrid(np.linspace(0, 2*np.pi, z.shape[0]), np.arange(z.shape[1])*d['ds'])
    ax.contour(phi, s, good.T.astype(float), [.5], colors='k', linewidths=.9)
    ax.contourf(phi, s, (~good).T.astype(float), levels=[.5, 1.5], colors='none', hatches=['////'])
    ax.set(ylabel=r'$s$ [$\mu$m]', xticks=[0, np.pi, 2*np.pi], xticklabels=['0', r'$\pi$', r'$2\pi$'])
    phasebar(ax)
    return im


def parameter_bound(d, g, b=B, eta_scale=1., pct=95):
    # Fixed waveform; H scales with viscosity. Recompute the region-of-trust.
    p = b*d['d2'] - eta_scale*d['hydro']
    good = (p*d['dt']<0) & (p*d['gamma']<0)
    return np.percentile(abs(p[good])/(A*(17500/L/8)*(.429 if g=='ODA' else 1)), pct)


def sweep(g):
    d = load(g); bs = np.geomspace(50, 2000, 90)
    return bs, np.array([parameter_bound(d,g,b,pct=[95,98]) for b in bs])


def fig2(wt):
    friction = .2*(4.7*2.7e-3*B/(A*L**2))*wt['dt']
    arrays = [wt['gamma'], wt['bend'], wt['slide'], wt['hydro'], friction]
    titles = [r'$\gamma$', r'$B\partial_s^2\gamma$', r'$Ka^2\gamma$', r'$\mathcal{H}$', r'$\mathcal{F}$']
    fig, axes = plt.subplots(1, 5, figsize=(13.7, 4.2), sharey=True)
    for i, (ax, z, title) in enumerate(zip(axes, arrays, titles)):
        lo, hi = (np.pi/4, np.pi) if i==0 else (-750, 750)
        im = ax.imshow(z.T, origin='lower', extent=(0, 2*np.pi, 0, L), aspect='auto', cmap='turbo', vmin=lo, vmax=hi)
        ax.set(title=title, xticks=[0, np.pi, 2*np.pi], xticklabels=['0', r'$\pi$', r'$2\pi$'])
        phasebar(ax); ax.tick_params(labelsize=13); ax.set_title(title, fontsize=18)
        if i == 0: ax.set_ylabel(r'$s$ [$\mu$m]', fontsize=15)
        if i >= 2:
            ax.contour(np.linspace(0,2*np.pi,z.shape[0]),np.linspace(0,L,z.shape[1]),z.T,[0],colors='k',linewidths=.8)
        if i in (0, 4):
            ticks, labels = ([np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                             [r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']) if i==0 else ([-750,0,750], ['-750','0','750'])
            cb=fig.colorbar(im, ax=ax, shrink=.76, pad=.025, ticks=ticks)
            cb.ax.set_yticklabels(labels); cb.ax.tick_params(labelsize=12)
            cb.set_label('[rad]' if i==0 else '[pN]', fontsize=14)
    fig.subplots_adjust(left=.07, right=.97, top=.81, bottom=.32, wspace=.36)
    save('Fig2', fig)


def fig3(data):
    fig = plt.figure(figsize=(11.4, 11.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05,1,.8], hspace=.65, wspace=.57,
                          left=.09, right=.91, top=.93, bottom=.07)
    for i, g in enumerate(('WT','ODA')):
        ax = fig.add_subplot(gs[0,i]); scatter(ax, data[g], g, histogram=i==1); tag(ax,'AB'[i]); ax.tick_params(labelsize=13)
        ax = fig.add_subplot(gs[1,i]); im=heat(ax,data[g]); tag(ax,'CD'[i]); ax.tick_params(labelsize=13)
        cb=fig.colorbar(im, ax=ax, shrink=.8, label=r'$F_{\rm head}$ [pN]'); cb.ax.tick_params(labelsize=12)
    ax = fig.add_subplot(gs[2,:]); tag(ax,'E')
    ax.axvspan(560,1120, color='.88', zorder=-2); ax.axvline(840,color='.3',lw=1.3)
    for g in ('WT','ODA'):
        bs,ps=sweep(g)
        ax.loglog(bs,ps[:,0],color=COL[g],lw=1.8,label=g)
        ax.loglog(bs,ps[:,1],color=COL[g],ls=':',lw=1.1)
    ax.set(xlim=(50,2000), ylim=(1,30), xlabel=r'Bending stiffness $B$ [pN $\mu$m$^2$]',
           ylabel=r'$F_{\rm head}^{95\%}$ [pN]')
    ax.set_xticks([50,100,200,500,1000,2000], labels=['50','100','200','500','1000','2000'])
    ax.set_yticks([1,2,5,10,20], labels=['1','2','5','10','20']); ax.grid(which='major',alpha=.15)
    ax.legend(frameon=False,loc='upper left'); ax.tick_params(labelsize=13)
    save('Fig3',fig)


def velocity(d):
    v=.26*A*abs(d['dt'][d['trust']]); f=abs(d['force'][d['trust']]); v=np.maximum(v,1e-9)
    f0=np.linspace(np.percentile(f,95),max(f)*3,80)
    slopes=np.array([np.percentile((q-f)/v,5) for q in f0]); ok=slopes>0
    return v,f,f0[ok],slopes[ok],f0[ok]/slopes[ok]


def atp(d, g):
    """ATP-equivalent minimum from all positive motor work (thesis Eqs. 10.36–10.38)."""
    motor = -d['passive'] + d['slide']  # b=0; includes bending, sliding elasticity, hydrodynamics
    power = np.nan_to_num(np.maximum(motor*d['dt'], 0))
    local = power.mean(axis=0)/d['f0'] / ((17500/L)*(.429 if g=='ODA' else 1)*E_ATP)
    return local, np.trapz(local, dx=d['ds'])/L


@plt.rc_context({'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
def fig4(data, ind):
    fig=plt.figure(figsize=(15.3,4.5)); gs=fig.add_gridspec(2,4,width_ratios=[.55,1.05,.9,1.55],hspace=.26,wspace=.45,
                                             left=.055,right=.98,top=.87,bottom=.19)
    ax=fig.add_subplot(gs[:,0]); tag(ax,'A'); ax.plot([0,1],[1,0],'k')
    ax.set(xlim=(0,1.3),ylim=(0,1.2),xlabel='$v$',ylabel='$F$',
           xticks=[0,1],xticklabels=['0',r'$v_0$'],yticks=[1],yticklabels=[r'$F_0$'])
    ax.set_box_aspect(1); ax.set_anchor('N'); ax.spines[['top','right']].set_visible(False)
    axc=fig.add_subplot(gs[:,2]);tag(axc,'C')
    for i,g in enumerate(('WT','ODA')):
        ax=fig.add_subplot(gs[i,1]);v,f,F0,slope,v0=velocity(data[g]); vr=np.linspace(0,35,200)
        ax.plot(v,f,'.',c=COL[g],ms=1.7,ls='none')
        for j in np.linspace(0,len(F0)-1,13,dtype=int): ax.plot(vr,F0[j]-slope[j]*vr,c='.78',lw=.65)
        ax.plot(vr,np.min(F0[:,None]-slope[:,None]*vr,axis=0),'k',lw=1.6)
        ax.text(.77,.79,g,transform=ax.transAxes,color=COL[g],fontweight='bold')
        ax.set(xlim=(0,33),ylim=(0,25),ylabel=r'$|F_{\rm head}|$ [pN]')
        if i: ax.set_xlabel(r'$v_{\rm eff}$ [$\mu$m/s]')
        else: tag(ax,'B')
        axc.plot(v0,F0,c=COL[g],label=g); axc.axhline(np.percentile(f,95),c=COL[g],ls=':',lw=.9)
        axc.axvline(min(v0),c=COL[g],ls=':',lw=.9,alpha=.6)
    axc.set(xlim=(0,50),ylim=(0,50),xlabel=r'$v_0$ [$\mu$m/s]',ylabel=r'$F_0$ [pN]')
    axc.legend(frameon=False)
    axd=fig.add_subplot(gs[:,3]); tag(axd,'D')
    inset=axd.inset_axes([.67,.58,.30,.38])
    for i,g in enumerate(FREQ):
        local,total=atp(data[g],g)
        axd.plot(np.linspace(0,1,len(local)),local,c=COL[g],lw=1.7,label=g)
        vals=[atp(x,g)[1] for x in ind[g]]
        inset.plot(i+np.linspace(-.13,.13,len(vals)),vals,'.',c=COL[g],ms=3.5,alpha=.55)
        inset.plot(i,total,'*',c=COL[g],ms=10)
    axd.axhline(1,c='.5',lw=.8,ls=':'); axd.set(xlim=(0,1),xlabel='$s/L$',ylabel='ATP equivalents / head / beat')
    axd.legend(frameon=False,loc='upper left',fontsize=12)
    inset.set(xlim=(-.4,1.4),xticks=[0,1],xticklabels=['WT','ODA'],ylabel='ATP / head / beat')
    inset.tick_params(labelsize=9); inset.yaxis.label.set_size(10)
    for panel in fig.axes:
        for label in panel.texts:
            if label.get_text() in ('A','B','C','D'): label.set_fontsize(16)
    save('Fig4',fig)


def individuals(g):
    p=DATA/f'phase_averaged_beat_patterns_individual_axonemes/{g}/KCl_000mM'
    return sorted(x.name for x in p.iterdir() if x.is_dir() and len(x.name)==16)


def fig_s23(g):
    ids=individuals(g); fig,axes=plt.subplots(5,3,figsize=(9.5,12.2))
    for ax,ident in zip(axes.flat,ids[:15]):
        p=Path(f'phase_averaged_beat_patterns_individual_axonemes/{g}/KCl_000mM/{ident}')
        gamma=csv(p.parent/'phase_aligned'/ident/'periodic_beat.csv')[:,1:]
        mean=csv(p/'mean_shape.csv')[:,1]
        x,y=xy(gamma)
        for k in range(0,len(gamma),2): ax.plot(x[k],y[k],c=plt.cm.hsv(k/len(gamma)),lw=.7)
        mx,my=xy(mean[None,:]); ax.plot(mx[0],my[0],'k',lw=1.5)
        ax.set(xlim=(-4,10),ylim=(-1.7,11)); ax.set_aspect('equal'); ax.axis('off')
        ax.plot([0,2],[-.6,-.6],'k',lw=2)
    for ax in axes.flat[15:]:ax.axis('off')
    fig.text(.5,.973,f'{g}  •  {len(ids)} axonemes (15 shown)',ha='center',fontsize=16,color=COL[g])
    fig.text(.05,.075,r'Scale bar: 2 $\mu$m',fontsize=12)
    strip=fig.add_axes([.23,.068,.54,.018]); strip.imshow(np.arange(256)[None,:],cmap='hsv',aspect='auto',extent=(0,2*np.pi,0,1))
    strip.set(yticks=[],xticks=[0,np.pi,2*np.pi],xticklabels=['0',r'$\pi$',r'$2\pi$'],xlabel=r'oscillator phase $\varphi$')
    fig.subplots_adjust(top=.94,bottom=.15,left=.05,right=.95,wspace=.10,hspace=.13)
    save('FigS2' if g=='WT' else 'FigS3',fig)


def fig_s4(ind):
    fig,ax=plt.subplots(figsize=(4.0,3.5))
    for i,g in enumerate(('WT','ODA')):
        values=np.array([bound(x) for x in ind[g]])
        ax.plot(i+np.linspace(-.13,.13,len(values)),values,'.',c=COL[g],ms=5,alpha=.55)
        ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),color=COL[g],marker='o',lw=1.7,capsize=6,ms=6)
    ax.set(xlim=(-.5,1.5),ylim=(0,32),xticks=[0,1],xticklabels=['WT','ODA'],ylabel=r'$F_{\rm head}^{95\%}$ [pN]')
    save('FigS4',fig)


def fig_s5(d):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.7))
    for i,g in enumerate(('WT','ODA')): scatter(axes[i],d[g],g,histogram=i==1);tag(axes[i],'AB'[i])
    fig.subplots_adjust(wspace=.54,left=.09,right=.86,top=.84,bottom=.19)
    save('FigS5',fig)


def fig_s6(wt,ideal):
    fig=plt.figure(figsize=(11.4,7.8));gs=fig.add_gridspec(2,2,height_ratios=[.75,1],hspace=.65,wspace=.65,
                                              left=.09,right=.89,bottom=.11,top=.91)
    amps=[];phases=[]
    for d in (wt,ideal):
        gamma=d['gamma'];phi=2*np.pi*np.arange(len(gamma))/len(gamma)
        coeff=2*np.mean(gamma*np.exp(-1j*phi[:,None]),axis=0)
        amps.append(abs(coeff));phases.append(-np.unwrap(np.angle(coeff)))
    s=np.linspace(0,1,len(amps[0])); ax=fig.add_subplot(gs[0,0]);tag(ax,'A')
    ax.plot(s,amps[0],c='.65',lw=2,label='measured');ax.plot(s,amps[1],'k',lw=1.8,label='idealized')
    ax.legend(frameon=False);ax.set(xlabel='$s/L$',ylabel=r'$a(s)$ [rad]')
    ax=fig.add_subplot(gs[0,1])
    for ph,c,lw in zip(phases,('.65','k'),(2,1.8)):
        ph[0]=2*ph[1]-ph[2]; ax.plot(s,ph-ph[1]-1.45,c=c,lw=lw)
    ax.set(xlabel='$s/L$',ylabel=r'$\Phi(s)-\Phi(0)$ [rad]')
    ax=fig.add_subplot(gs[1,0]);scatter(ax,ideal,'WT',histogram=True);tag(ax,'B')
    ax=fig.add_subplot(gs[1,1]);im=heat(ax,ideal,-5,5);tag(ax,'C')
    fig.colorbar(im,ax=ax,label=r'$F_{\rm head}$ [pN]',shrink=.83)
    save('FigS6',fig)


def striegler():
    p=Path('other_data/Supplementary Data_Geyer')
    r=np.stack([np.loadtxt(DATA/p/f'{z}_arclength_phase.txt') for z in 'xyz'],axis=-1).transpose(1,0,2)
    out={}
    for mode in ('planar_xy','signed_curvature_3D'):
        if mode=='planar_xy': gamma=np.arctan2(np.diff(r[:,:,1],axis=1),np.diff(r[:,:,0],axis=1))
        else:
            rows=[]
            for pos in r:
                ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
                t=np.stack([np.gradient(pos[:,j],ss) for j in range(3)],axis=-1);t/=np.linalg.norm(t,axis=1)[:,None]
                dt=np.stack([np.gradient(t[:,j],ss) for j in range(3)],axis=-1)
                k=np.linalg.norm(dt,axis=1)*np.sign(t[:,0]*dt[:,1]-t[:,1]*dt[:,0])
                rows.append(np.r_[0,np.cumsum(np.diff(ss)*(k[:-1]+k[1:])/2)][:-1])
            gamma=np.array(rows)
        h=csv(p/'precomputed_hydrodynamic_force_moment'/f'hydrodynamic_torque_free_{mode}.csv',header=False)[:,1:]
        out[mode]=calc(gamma,h,1/.014,ds=.4156)
    return out


def main():
    d={g:load(g) for g in FREQ}
    ind={g:[load(g,ident=i) for i in individuals(g)] for g in FREQ}
    fig2(d['WT']); fig3(d); fig4(d,ind)
    for g in FREQ: fig_s23(g)
    fig_s4(ind)
    dyn={g:load(g,'dynamic') for g in FREQ};fig_s5(dyn)
    ideal={g:load(g,'ideal') for g in FREQ};fig_s6(d['WT'],ideal['WT'])
    third=striegler()
    lines=['AXONEME FORCE ANALYSIS — bundled processed data',
           'Equation: F_head = -(B*gamma_ss - H)/(a*rho_DMT), H = -stored CSV;',
           'trust: (B*gamma_ss-H)*gamma_t < 0 and (B*gamma_ss-H)*gamma < 0.',
           'B=840 pN um², K=2000 pN/um², a=0.2 um, L=10 um.',
           'Reporting absolute force percentiles within region-of-trust.',
           f'Fig4D: all positive work from m_m=-B*gamma_ss+K*a²*gamma+H (b=0), divided by N*{E_ATP:.3f} pN um/ATP (100 zJ);',
           'Uncertainty: vary B by ±280 pN um² and viscosity by ±100% separately, holding waveform fixed and recomputing trust.',
           'These are parameter sensitivities, not confidence intervals. Relative sensitivity to a and rho is -1 at fixed H.',
           'ATP-equivalent energetic LOWER bound (100% efficiency), not measured ATP consumption.',
           'Fig4D follows thesis Sec. 10.4 (positive-work domain, b=0); no region-of-trust restriction.','']
    for g in FREQ:
        f=d[g]; gind=np.array([bound(x) for x in ind[g]]); v,ff,F0,sl,v0=velocity(f)
        hydro=np.nanmean(abs(f['hydro'])/(abs(f['bend']-f['slide'])+abs(f['hydro'])+1e-12))
        nominal=bound(f)
        bvals=np.array([parameter_bound(f,g,b=B+x*DB) for x in (-1,1)])
        evals=np.array([parameter_bound(f,g,eta_scale=x) for x in np.linspace(1-DETA,1+DETA,201)])
        basal=[]
        for alpha in (0,.5,1):
            gam=f['gamma']-alpha*f['gamma'].mean(axis=1,keepdims=True)
            good=(f['passive']*np.gradient(gam,axis=0)*f['n']*f['f0']<0)&(f['passive']*gam<0)
            basal.append(np.percentile(abs(f['force'][good]),95))
        stiffness,forces=sweep(g); k=np.argmin(abs(forces[:,0]-5))
        lines += [f'{g}: {len(gind)} axonemes, mean beat frequency {f["f0"]:.2f} Hz',
                  f'  group: trust {f["trust"].sum()}/{f["trust"].size} ({100*f["trust"].mean():.2f}%), 95% {bound(f):.3f} pN, 98% {bound(f,98):.3f} pN',
                  f'  B={B-DB:.0f}/{B+DB:.0f}: 95th percentile {bvals[0]:.3f}/{bvals[1]:.3f} pN; relative changes {(bvals[0]/nominal-1)*100:+.1f}%/{(bvals[1]/nominal-1)*100:+.1f}%',
                  f'  eta/eta0=0..2 (±100%, 201 samples): 95th percentile range {evals.min():.3f}–{evals.max():.3f} pN; max absolute relative change {100*max(abs(evals/nominal-1)):.2f}%',
                  f'  ATP-equivalent lower bound (all positive motor work): {atp(f,g)[1]:.3f} per head/beat; spatial peak {atp(f,g)[0].max():.3f}',
                  f'  dynamic-only (FigS5): 95% {bound(dyn[g]):.3f} pN',
                  f'  idealized (FigS6 uses WT): 95% {bound(ideal[g]):.3f} pN',
                  f'  individual: mean±sample SD of 95% bounds = {gind.mean():.3f} ± {gind.std(ddof=1):.3f} pN',
                  f'  hydrodynamic relative fraction (script convention): {100*hydro:.3f}%',
                  f'  basal sliding alpha 0/.5/1: 95% bounds {basal[0]:.3f}/{basal[1]:.3f}/{basal[2]:.3f} pN',
                  f'  stiffness giving approximately 5 pN at 95%: B≈{stiffness[k]:.1f} pN um²',
                  f'  force–velocity: minimum interpolated F0={F0.min():.3f} pN, min v0={v0.min():.3f} um/s, {len(v)} trusted (v,F) samples',
                  '  individual 95% bounds [ID: pN]:']
        lines += [f'    {i}: {x:.3f}' for i,x in zip(individuals(g),gind)] + ['']
    lines += [f'Striegler 3D reference (n=17), planar projection: {bound(third["planar_xy"]):.3f} pN',
              f'Striegler 3D reference (n=17), signed curvature: {bound(third["signed_curvature_3D"]):.3f} pN',
              '', 'Manuscript comparison (95th %): group WT 10.2, ODA 18.2; dynamic WT 9.6, ODA 18.0;',
              'idealized WT 3.1, ODA 6.3; 3D planar 7.1, 3D signed 6.1 pN.',
              'Figure panels are programmatically recreated, not pixel-identical publisher artwork.']
    txt='\n'.join(lines)+'\n';(ROOT/'analysis_report.txt').write_text(txt,encoding='utf-8');print(txt)
    print(f'Saved {len(list((ROOT/"output_png").glob("Fig*.png")))} PNGs and {len(list((ROOT/"output_pdf").glob("Fig*.pdf")))} PDFs.')


if __name__=='__main__': main()
