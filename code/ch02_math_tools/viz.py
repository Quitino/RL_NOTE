"""
Ch02: Math Tools Visualizations
Figures:
  - markov_chain.png      : Markov chain state transition diagram
  - kl_divergence.png     : KL divergence between two Gaussians
  - bias_variance.png     : Bias-variance tradeoff curve
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from scipy.stats import norm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch02_math_tools')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Markov Chain ────────────────────────────────────────────────────
def plot_markov_chain():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')
    ax.set_title('Markov Chain: State Transition Diagram\n'
                 'P(s_{t+1} | s_t, s_{t-1}, ...) = P(s_{t+1} | s_t)',
                 fontsize=12, fontweight='bold')

    colors = ['#4C8EBF', '#E8994C', '#5BAD6F', '#BF4C4C', '#9B59B6']
    states = [f's_{i}' for i in range(5)]
    xs = [1, 2.8, 4.6, 6.4, 8.2]
    y = 1.0

    # Draw state circles
    for i, (x, s) in enumerate(zip(xs, states)):
        circle = plt.Circle((x, y), 0.45, color=colors[i], zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, s, ha='center', va='center', fontsize=12,
                color='white', fontweight='bold', zorder=4)

    # Transition arrows between consecutive states
    probs = [0.7, 0.6, 0.8, 0.5]
    for i in range(4):
        ax.annotate('', xy=(xs[i+1] - 0.46, y), xytext=(xs[i] + 0.46, y),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0))
        ax.text((xs[i] + xs[i+1]) / 2, y + 0.62,
                f'p={probs[i]}', ha='center', va='bottom', fontsize=9, color='#333333')

    # Self-loops
    loop_probs = [0.3, 0.4, 0.2, 0.5, 0.9]
    for i, (x, p) in enumerate(zip(xs, loop_probs)):
        ax.annotate('', xy=(x + 0.3, y + 0.44), xytext=(x - 0.3, y + 0.44),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5,
                                   connectionstyle='arc3,rad=-1.2'))
        ax.text(x, y + 1.0, f'p={p}', ha='center', va='bottom', fontsize=8, color=colors[i])

    # Markov property annotation
    ax.text(4.6, -0.9, 'Markov Property: future depends only on current state,  NOT history',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, 'markov_chain.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: KL Divergence ───────────────────────────────────────────────────
def plot_kl_divergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('KL Divergence Between Two Distributions\n'
                 'KL(P || Q) = E_P[ log(P/Q) ] ≥ 0', fontsize=12, fontweight='bold')

    x = np.linspace(-5, 8, 500)

    pairs = [
        dict(mu_p=0, s_p=1.0, mu_q=2, s_q=1.2, label='Moderate KL'),
        dict(mu_p=0, s_p=0.5, mu_q=3, s_q=1.5, label='Large KL'),
    ]

    for ax, cfg in zip(axes, pairs):
        p = norm.pdf(x, cfg['mu_p'], cfg['s_p'])
        q = norm.pdf(x, cfg['mu_q'], cfg['s_q'])
        ax.fill_between(x, p, alpha=0.35, color='#4C8EBF', label=f"P: N({cfg['mu_p']}, {cfg['s_p']}²)")
        ax.fill_between(x, q, alpha=0.35, color='#E8994C', label=f"Q: N({cfg['mu_q']}, {cfg['s_q']}²)")
        ax.plot(x, p, color='#4C8EBF', lw=2)
        ax.plot(x, q, color='#E8994C', lw=2)

        # KL computation
        eps = 1e-10
        kl = np.trapz(p * np.log((p + eps) / (q + eps)), x)
        ax.set_title(f'{cfg["label"]}\nKL(P||Q) ≈ {kl:.3f}', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'kl_divergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Bias-Variance Tradeoff ─────────────────────────────────────────
def plot_bias_variance():
    fig, ax = plt.subplots(figsize=(8, 5))

    complexity = np.linspace(0.5, 10, 200)
    bias2   = 4.5 / complexity**1.3
    variance = 0.08 * complexity**1.8
    total   = bias2 + variance + 0.3   # irreducible noise

    ax.plot(complexity, bias2,   color='#4C8EBF', lw=2.5, label='Bias²', linestyle='--')
    ax.plot(complexity, variance, color='#E8994C', lw=2.5, label='Variance', linestyle='-.')
    ax.plot(complexity, total,   color='#BF4C4C', lw=3.0, label='Total Error = Bias² + Variance + Noise')

    # optimal point
    opt_idx = np.argmin(total)
    ax.axvline(complexity[opt_idx], color='#5BAD6F', lw=1.5, linestyle=':', alpha=0.8)
    ax.scatter([complexity[opt_idx]], [total[opt_idx]], color='#5BAD6F', s=120, zorder=5)
    ax.text(complexity[opt_idx] + 0.2, total[opt_idx] + 0.15, 'Optimal\ncomplexity',
            fontsize=9, color='#5BAD6F', va='bottom')

    # RL relevance note
    ax.axvspan(0.5, complexity[opt_idx] * 0.6, alpha=0.06, color='#4C8EBF')
    ax.axvspan(complexity[opt_idx] * 1.4, 10, alpha=0.06, color='#E8994C')
    ax.text(1.5, ax.get_ylim()[1] * 0.92, 'Underfitting\n(high bias)', ha='center',
            fontsize=8, color='#4C8EBF')
    ax.text(8.5, ax.get_ylim()[1] * 0.92, 'Overfitting\n(high variance)', ha='center',
            fontsize=8, color='#E8994C')

    ax.set_xlabel('Model Complexity / Function Approximator Capacity', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Bias-Variance Tradeoff\n'
                 'Relevant to RL: TD methods (low variance, biased) vs MC (unbiased, high variance)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 10)
    ax.set_ylim(0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'bias_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_markov_chain()
    plot_kl_divergence()
    plot_bias_variance()
    print("Ch02 done.")
