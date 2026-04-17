"""
Ch12: Sim-to-Real Visualizations
Figures:
  - domain_randomization.png  : Domain randomization parameter distributions
  - sim2real_pipeline.png     : Sim-to-real transfer pipeline flowchart
  - reward_breakdown.png      : Reward function component breakdown
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm, uniform

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch12_sim_to_real')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Domain Randomization ───────────────────────────────────────────
def plot_domain_randomization():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Domain Randomization: Sampling Physical Parameters During Training\n'
                 'Train on a distribution of environments → generalize to real world',
                 fontsize=12, fontweight='bold')

    params = [
        ('Friction Coefficient μ', 0.6, 0.2, 'normal',   '#4C8EBF', (0.0, 1.5)),
        ('Ground Stiffness (kN/m)', 500, 150,  'normal',  '#E8994C', (100, 900)),
        ('Motor Torque Scale',    1.0, 0.15,   'normal',  '#5BAD6F', (0.5, 1.5)),
        ('Link Mass Offset (kg)', 0.0, 0.3,    'normal',  '#9B59B6', (-1.0, 1.0)),
        ('Initial Pose Noise',    0.0, 0.08,   'normal',  '#BF4C4C', (-0.3, 0.3)),
        ('Control Latency (ms)',  5.0, 3.0,    'uniform', '#1ABC9C', (0, 15)),
    ]

    for ax, (name, mu, sig, dist, c, xlim) in zip(axes.flat, params):
        x = np.linspace(xlim[0], xlim[1], 300)
        if dist == 'normal':
            y = norm.pdf(x, mu, sig)
        else:
            y = uniform.pdf(x, xlim[0], xlim[1] - xlim[0])

        ax.fill_between(x, y, alpha=0.4, color=c)
        ax.plot(x, y, color=c, lw=2.5)
        ax.axvline(mu if dist == 'normal' else (xlim[0] + xlim[1]) / 2,
                   color='black', lw=1.5, linestyle='--', alpha=0.7, label='Nominal')
        # Shade real-world likely region
        real_lo = mu - sig * 0.5 if dist == 'normal' else (xlim[0] + xlim[1]) * 0.45
        real_hi = mu + sig * 0.5 if dist == 'normal' else (xlim[0] + xlim[1]) * 0.55
        ax.axvspan(real_lo, real_hi, alpha=0.12, color='#F0C000', label='Real world range')
        ax.set_title(name, fontsize=9.5, fontweight='bold', color=c)
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUT, 'domain_randomization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Sim-to-Real Pipeline ───────────────────────────────────────────
def plot_sim2real_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Sim-to-Real Transfer Pipeline for Humanoid Locomotion',
                 fontsize=13, fontweight='bold', pad=10)

    def box(cx, cy, w, h, txt, color, fontsize=9):
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333', lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold')

    def arr(x0, y0, x1, y1, lbl='', c='#444'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.0))
        if lbl:
            ax.text((x0+x1)/2, (y0+y1)/2 + 0.2, lbl, ha='center', fontsize=8, color=c)

    # Simulation box
    box(2.5, 5.0, 3.8, 3.5, 'Physics Simulator\n(MuJoCo / IsaacGym / Genesis)\n\n'
        '• High-speed parallel rollouts\n• Domain randomization\n• 4096+ envs in parallel',
        '#4C8EBF', 8.5)

    # Policy
    box(6.5, 5.2, 2.8, 1.0, 'Policy  π_θ\n(PPO / SAC)', '#E8994C')
    box(6.5, 3.5, 2.8, 1.5, 'Observation\nNormalization\n+ Noise Injection', '#9B59B6')

    # Adaptation
    box(10.0, 5.2, 2.8, 1.0, 'Sim2Real\nAdapter\n(RMA / EST)', '#BF4C4C')

    # Real robot
    box(10.0, 3.0, 2.8, 1.5, 'Real Robot\n(Unitree G1 / H1)\nOnboard inference', '#5BAD6F')

    arr(4.4, 5.0, 5.5, 5.2, 'state s_t')
    arr(6.5, 4.7, 6.5, 4.25, 'actions')
    arr(7.9, 5.2, 8.6, 5.2, 'policy')
    arr(10.0, 4.7, 10.0, 3.75, 'adapted obs')

    # Feedback loop
    ax.annotate('', xy=(2.5, 3.3), xytext=(8.6, 3.0),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=1.8,
                                connectionstyle='arc3,rad=0.3'))
    ax.text(5.5, 2.0, 'Real data used for\nsystem identification / fine-tuning', ha='center',
            fontsize=8.5, color='#5BAD6F', style='italic')

    # Gap annotation
    ax.text(6.5, 6.7,
            'Sim-to-Real Gap: friction, motor dynamics, sensing noise, unmodeled contacts',
            ha='center', fontsize=8.5, color='#333',
            bbox=dict(fc='#FFF5E6', ec='#E8994C', boxstyle='round,pad=0.3', lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, 'sim2real_pipeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Reward Function Breakdown ───────────────────────────────────────
def plot_reward_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Humanoid Locomotion Reward Function Design\n'
                 'Typical decomposition: R_total = Σ_i w_i · r_i',
                 fontsize=12, fontweight='bold')

    # Components
    components = [
        ('Velocity\nTracking',   2.0, '#4C8EBF'),
        ('Joint Torque\nPenalty', -0.5, '#BF4C4C'),
        ('Foot Contact\nSymmetry', 0.8, '#5BAD6F'),
        ('Body Height\nRegulation', 0.6, '#E8994C'),
        ('Smoothness\nPenalty',   -0.3, '#9B59B6'),
        ('Fall\nPenalty',        -1.5, '#E74C3C'),
        ('Energy\nEfficiency',    0.4, '#1ABC9C'),
    ]

    labels = [c[0] for c in components]
    weights = [c[1] for c in components]
    colors  = [c[2] for c in components]

    # Left: bar chart of weights
    ax = axes[0]
    bars = ax.barh(labels, weights, color=colors, edgecolor='white', height=0.65)
    ax.axvline(0, color='black', lw=1.5)
    ax.set_xlabel('Reward weight  w_i')
    ax.set_title('Reward Component Weights\n(positive = bonus, negative = penalty)')
    for bar, w in zip(bars, weights):
        ax.text(w + 0.05 * np.sign(w), bar.get_y() + bar.get_height() / 2,
                f'{w:+.1f}', va='center', fontsize=9, fontweight='bold',
                color='#1a5c1a' if w > 0 else '#8B0000')
    ax.grid(axis='x', alpha=0.3)

    # Right: pie chart of magnitude contribution
    ax2 = axes[1]
    magnitudes = [abs(w) for w in weights]
    wedge_colors = [c[2] for c in components]
    wedges, texts, autotexts = ax2.pie(magnitudes, labels=labels, colors=wedge_colors,
                                        autopct='%1.0f%%', startangle=90,
                                        pctdistance=0.75, labeldistance=1.1)
    for t in texts:
        t.set_fontsize(8.5)
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color('white')
        at.set_fontweight('bold')
    ax2.set_title('Relative Magnitude of Each\nReward Component')

    plt.tight_layout()
    path = os.path.join(OUT, 'reward_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_domain_randomization()
    plot_sim2real_pipeline()
    plot_reward_breakdown()
    print("Ch12 done.")
