"""
Ch05: Monte Carlo and TD Visualizations
Figures:
  - mc_vs_td_bias_variance.png  : MC vs TD bias/variance comparison
  - nstep_return.png            : n-step return interpolation
  - td_lambda_trace.gif         : TD(λ) eligibility trace animation
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../docs/asserts/ch05_mc_td')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: MC vs TD Bias/Variance ─────────────────────────────────────────
def plot_mc_vs_td():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MC vs TD: Bias-Variance Tradeoff', fontsize=13, fontweight='bold')

    # Left: conceptual bias/variance axes
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    methods = [
        ('REINFORCE\n(MC full)', 1.0, 8.5, '#4C8EBF'),
        ('TD(0)',                8.5, 1.5, '#E8994C'),
        ('TD(λ) λ=0.8',         3.5, 5.0, '#5BAD6F'),
        ('TD(λ) λ=0.5',         5.5, 3.5, '#BF4C4C'),
        ('n-step n=4',          4.5, 4.5, '#9B59B6'),
    ]
    for name, bias, var, c in methods:
        ax.scatter([bias], [var], color=c, s=180, zorder=5)
        ax.text(bias + 0.3, var + 0.3, name, fontsize=8.5, color=c, fontweight='bold')

    ax.set_xlabel('Bias  (estimator bias)', fontsize=11)
    ax.set_ylabel('Variance  (sample noise)', fontsize=11)
    ax.set_title('Conceptual Bias-Variance Position', fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(8, 9, 'Ideal:\nlow bias\nlow var', ha='center', fontsize=8,
            color='gray', style='italic')
    ax.axvline(5, color='gray', lw=0.8, linestyle='--', alpha=0.4)
    ax.axhline(5, color='gray', lw=0.8, linestyle='--', alpha=0.4)

    # Right: return estimate comparison on a sample trajectory
    ax2 = axes[1]
    T = 12
    rewards = np.array([0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 3, 0], dtype=float)
    gamma = 0.9
    t_plot = np.arange(T)

    # MC return from t=0
    mc_return = np.zeros(T)
    g = 0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g
        mc_return[t] = g

    # TD(0) bootstrapped estimates (simulated)
    td_return = mc_return + np.random.default_rng(42).normal(0, 0.4, T)

    ax2.fill_between(t_plot, mc_return - 0.5, mc_return + 0.5,
                     alpha=0.2, color='#4C8EBF', label='MC uncertainty band')
    ax2.plot(t_plot, mc_return, color='#4C8EBF', lw=2.5, marker='o', label='MC return G_t')
    ax2.fill_between(t_plot, td_return - 0.15, td_return + 0.15,
                     alpha=0.25, color='#E8994C')
    ax2.plot(t_plot, td_return, color='#E8994C', lw=2.5, marker='s', linestyle='--',
             label='TD estimate (low var, biased)')
    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('Value estimate')
    ax2.set_title('Return Estimates Along Trajectory\n(MC high variance, TD low variance)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'mc_vs_td_bias_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: n-step Return ───────────────────────────────────────────────────
def plot_nstep_return():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1, 5.5)
    ax.axis('off')
    ax.set_title('n-Step Return: Interpolating TD(0) and Monte Carlo\n'
                 'G_t^(n) = r_t + γ·r_{t+1} + ... + γ^(n-1)·r_{t+n-1} + γ^n·V(s_{t+n})',
                 fontsize=11, fontweight='bold')

    colors = ['#4C8EBF', '#5BAD6F', '#E8994C', '#BF4C4C', '#9B59B6']
    ns = [1, 2, 4, 8, '∞ (MC)']
    labels = ['TD(0)  n=1', 'n=2', 'n=4', 'n=8', 'Monte Carlo  n=∞']

    # Draw trajectory
    T = 10
    xs = np.arange(T) + 0.5
    for x in xs:
        ax.scatter([x], [3.5], color='#888888', s=60, zorder=3)
        if x < T - 0.5:
            ax.annotate('', xy=(x + 1, 3.5), xytext=(x, 3.5),
                        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2))
    ax.text(-0.2, 3.5, 's_t', ha='right', va='center', fontsize=10, color='#555555')

    # Bootstrap arrows for different n
    row_ys = [4.5, 4.1, 3.9, 3.7, 3.5]   # slightly offset for clarity; shown below instead
    for i, (n, lbl, c) in enumerate(zip(ns, labels, colors)):
        n_val = T if n == '∞ (MC)' else n
        y = 2.5 - i * 0.55
        # Span bracket
        ax.plot([0.5, 0.5 + n_val], [y, y], color=c, lw=3.5, solid_capstyle='round')
        ax.scatter([0.5], [y], color=c, s=60, zorder=5)
        ax.scatter([0.5 + n_val], [y], color=c, s=90, marker='D', zorder=5)
        ax.text(0.5 + n_val + 0.25, y, f'V(s_{{t+{n_val}}}) bootstrap', va='center',
                fontsize=8.5, color=c)
        ax.text(-0.2, y, lbl, ha='right', va='center', fontsize=9, color=c, fontweight='bold')

    ax.text(5.5, -0.6,
            'n=1 (TD): uses immediate reward + V bootstrap  |  n=∞ (MC): uses full trajectory return',
            ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    path = os.path.join(OUT, 'nstep_return.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: TD(λ) Eligibility Trace Animation ──────────────────────────────
def plot_td_lambda_anim():
    T = 12
    gammas = [0.9]
    lambdas = [0.3, 0.7, 0.95]
    colors = ['#4C8EBF', '#E8994C', '#5BAD6F']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.5, T + 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Time step  t')
    ax.set_ylabel('Eligibility trace  e(s)')
    ax.set_title('TD(λ) Eligibility Trace Decay\ne(s) ← γ·λ·e(s)   at each step; +1 at visited state',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    lines = [ax.plot([], [], color=c, lw=2.5, label=f'λ={lam}')[0]
             for lam, c in zip(lambdas, colors)]
    ax.legend(fontsize=10)

    rng = np.random.default_rng(7)
    visit_times = sorted(rng.choice(T, size=4, replace=False))
    vline = ax.axvline(0, color='gray', lw=1.2, linestyle='--', alpha=0.6)
    visit_marks = [ax.axvline(v, color='#BF4C4C', lw=1.0, linestyle=':', alpha=0.0)
                   for v in visit_times]

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')

    def compute_trace(lam, up_to_t):
        e = 0.0
        trace = []
        for t in range(up_to_t + 1):
            e = 0.9 * lam * e
            if t in visit_times:
                e += 1.0
            trace.append(e)
        return np.array(trace)

    FRAMES = T + 3

    def update(frame):
        t = min(frame, T - 1)
        for line, lam in zip(lines, lambdas):
            tr = compute_trace(lam, t)
            line.set_data(np.arange(len(tr)), tr)
        vline.set_xdata([t, t])
        for vm, vt in zip(visit_marks, visit_times):
            vm.set_alpha(0.5 if vt <= t else 0.0)
        time_text.set_text(f't = {t}' + ('  (visited!)' if t in visit_times else ''))
        return lines + [vline, time_text] + visit_marks

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=400, blit=True)
    path = os.path.join(OUT, 'td_lambda_trace.gif')
    ani.save(path, writer='pillow', fps=2.5)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_mc_vs_td()
    plot_nstep_return()
    plot_td_lambda_anim()
    print("Ch05 done.")
