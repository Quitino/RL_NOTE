"""
Ch10: PPO Visualizations
Figures:
  - ppo_clip_objective.png   : PPO clip objective shape
  - trpo_vs_ppo.png          : TRPO trust region vs PPO clip boundary
  - ppo_training_loop.png    : PPO training loop flowchart
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.patches import Arc

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch10_ppo')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: PPO Clip Objective ──────────────────────────────────────────────
def plot_ppo_clip():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PPO-Clip Objective\n'
                 'L^CLIP = E[min(r_t(θ)·A_t,  clip(r_t, 1−ε, 1+ε)·A_t)]',
                 fontsize=12, fontweight='bold')

    r = np.linspace(0.2, 2.5, 300)
    eps = 0.2

    for ax, A_sign, sign_label in zip(axes, [1, -1], ['Positive  A > 0', 'Negative  A < 0']):
        A = A_sign * 1.0
        surr  = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        objective = np.minimum(surr, clipped)

        ax.plot(r, surr,      color='#4C8EBF', lw=2, linestyle='--', label='r·A  (unclipped)')
        ax.plot(r, clipped,   color='#E8994C', lw=2, linestyle=':',  label='clip(r,1±ε)·A')
        ax.plot(r, objective, color='#BF4C4C', lw=3,                 label='min(·,·)  (PPO objective)')
        ax.axvline(1.0,       color='gray',   lw=1.2, linestyle='--', alpha=0.6)
        ax.axvline(1 - eps,   color='#5BAD6F', lw=1.0, linestyle=':')
        ax.axvline(1 + eps,   color='#5BAD6F', lw=1.0, linestyle=':',
                   label=f'clip bounds  [1-ε, 1+ε]  ε={eps}')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel('Probability ratio  r = π_new / π_old', fontsize=10)
        ax.set_ylabel('Objective value')
        ax.set_title(sign_label)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, 'ppo_clip_objective.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: TRPO vs PPO ─────────────────────────────────────────────────────
def plot_trpo_vs_ppo():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('TRPO (Trust Region) vs PPO-Clip: Constraining Policy Updates',
                 fontsize=12, fontweight='bold')

    for ax, title, color, constraint_type in zip(
            axes,
            ['TRPO: KL Divergence Constraint\nKL(π_old || π_new) ≤ δ',
             'PPO-Clip: Ratio Clipping\n|r−1| ≤ ε  (implicit constraint)'],
            ['#4C8EBF', '#E8994C'],
            ['kl', 'clip']):

        theta = np.linspace(0, 2 * np.pi, 300)
        # Objective surface contours (schematic ellipses)
        for lvl, alpha in [(0.3, 0.15), (0.5, 0.2), (0.8, 0.25), (1.2, 0.3)]:
            ax.add_patch(Ellipse((0, 0), 4 * lvl, 3 * lvl, color='#888888', alpha=alpha, fill=True))

        # Current policy
        ax.scatter([0], [0], color='black', s=120, zorder=6, label='θ_old')

        if constraint_type == 'kl':
            # Trust region circle
            circle = plt.Circle((0, 0), 1.0, color=color, fill=False, lw=2.5,
                                 linestyle='--', label='KL ≤ δ  (trust region)')
            ax.add_patch(circle)
            # Optimal inside trust region
            ax.scatter([0.6], [0.4], color=color, s=150, zorder=7, marker='*', label='θ* (constrained)')
            ax.annotate('', xy=(0.6, 0.4), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
        else:
            # Clip region box
            eps = 0.2
            rect = plt.Rectangle((-1, -1), 2, 2, color=color, fill=False, lw=2.5,
                                  linestyle='--', label='|r−1| ≤ ε  (clip box)')
            ax.add_patch(rect)
            ax.scatter([0.5], [0.35], color=color, s=150, zorder=7, marker='*', label='θ* (clipped)')
            ax.annotate('', xy=(0.5, 0.35), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.2)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUT, 'trpo_vs_ppo.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: PPO Training Loop ───────────────────────────────────────────────
def plot_ppo_loop():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('PPO Training Loop Flowchart', fontsize=13, fontweight='bold', pad=10)

    def box(cx, cy, w, h, txt, color, fontsize=9.5):
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333', lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold')

    def arr(x0, y0, x1, y1, lbl='', c='#444'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.0))
        if lbl:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx + 0.25, my, lbl, fontsize=8, color=c)

    box(5, 9.2, 5.5, 0.9, 'Initialize  π_θ,  V_φ  (random weights)', '#4C8EBF')
    box(5, 7.8, 5.5, 0.9, 'Collect N rollouts with  π_old = π_θ', '#4C8EBF')
    box(5, 6.4, 5.5, 0.9, 'Compute returns G_t  and  advantages A_t^GAE', '#9B59B6')
    box(5, 5.0, 5.5, 0.9, 'Normalize advantages:  A ← (A − mean)/std', '#9B59B6')
    box(5, 3.6, 6.5, 0.9,
        'For K epochs over mini-batches:\n'
        '  Compute L^CLIP + c₁·L^VF − c₂·H\n'
        '  Update θ, φ  with Adam', '#E8994C')
    box(5, 2.2, 5.5, 0.9, 'Check KL(π_old, π_new) ≤ KL_target?', '#BF4C4C')
    box(5, 0.9, 5.5, 0.7, 'Loop back for next iteration', '#5BAD6F')

    arr(5, 8.75, 5, 8.25, '')
    arr(5, 7.35, 5, 6.85, '')
    arr(5, 5.95, 5, 5.45, '')
    arr(5, 4.55, 5, 4.05, '')
    arr(5, 3.15, 5, 2.65, '')
    arr(5, 1.75, 5, 1.25, '')

    # Early stop annotation
    ax.text(7.8, 2.2, 'If KL too large:\nearly stop K epochs',
            ha='center', fontsize=8, color='#BF4C4C',
            bbox=dict(fc='#FFF0F0', ec='#BF4C4C', boxstyle='round,pad=0.3'))
    ax.annotate('', xy=(7.0, 2.2), xytext=(5.0 + 3.25 * 0.5, 2.2),
                arrowprops=dict(arrowstyle='->', color='#BF4C4C', lw=1.5))

    # Loop-back arc
    ax.annotate('', xy=(2.0, 7.8), xytext=(2.0, 0.9),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=1.8,
                                connectionstyle='arc3,rad=-0.4'))

    plt.tight_layout()
    path = os.path.join(OUT, 'ppo_training_loop.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_ppo_clip()
    plot_trpo_vs_ppo()
    plot_ppo_loop()
    print("Ch10 done.")
