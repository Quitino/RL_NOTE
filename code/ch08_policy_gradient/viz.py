"""
Ch08: Policy Gradient Visualizations
Figures:
  - pg_theorem_flow.png        : Policy gradient theorem derivation flow
  - reinforce_variance.png     : REINFORCE high variance illustration
  - baseline_variance.gif      : Baseline subtraction variance reduction (animated)
  - pg_vs_value_comparison.png : Policy gradient vs value-based update comparison
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

OUT = os.path.join(os.path.dirname(__file__), '../../docs/asserts/ch08_policy_gradient')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Policy Gradient Theorem Flow ───────────────────────────────────
def plot_pg_theorem_flow():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Policy Gradient Theorem: Derivation Flow',
                 fontsize=13, fontweight='bold', pad=10)

    def box(cx, cy, w, h, txt, color, fontsize=9.5):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333',
                                    lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold')

    def arrow(x0, y0, x1, y1, lbl=''):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.8))
        if lbl:
            ax.text((x0+x1)/2 + 0.2, (y0+y1)/2, lbl, fontsize=8, color='#444')

    box(5, 7.3, 7.5, 0.9, 'Objective:  J(θ) = E_{τ~π_θ} [G_0]  = E[Σ_t r_t]', '#4C8EBF', 10)
    arrow(5, 6.85, 5, 6.25, 'differentiate')
    box(5, 5.9, 7.5, 0.9, '∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ', '#4C8EBF', 10)
    arrow(5, 5.45, 5, 4.85, 'log-derivative trick\n∇ log p_θ = ∇p_θ / p_θ')
    box(5, 4.45, 8.5, 0.9, '= E_{τ~π_θ} [ (Σ_t ∇_θ log π_θ(a_t|s_t)) · R(τ) ]', '#9B59B6', 10)
    arrow(5, 4.0, 5, 3.4, 'causality + baseline')
    box(5, 3.0, 8.5, 0.9, '≈ E [ Σ_t  ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]', '#5BAD6F', 10)
    arrow(5, 2.55, 5, 1.95, 'sample gradient')
    box(5, 1.6, 7.5, 0.9, 'θ ← θ + α · ∇_θ J(θ)   (gradient ascent)', '#E8994C', 10)

    # Side annotation
    ax.text(0.3, 3.0, 'REINFORCE\nestimator', ha='center', va='center',
            fontsize=9, color='#5BAD6F', fontweight='bold',
            bbox=dict(fc='#F0FFF0', ec='#5BAD6F', boxstyle='round,pad=0.3'))
    ax.text(0.3, 4.45, 'Policy Gradient\nTheorem', ha='center', va='center',
            fontsize=9, color='#9B59B6', fontweight='bold',
            bbox=dict(fc='#F5F0FF', ec='#9B59B6', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    path = os.path.join(OUT, 'pg_theorem_flow.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: REINFORCE High Variance ────────────────────────────────────────
def plot_reinforce_variance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('REINFORCE: High Variance of Monte Carlo Return Estimates',
                 fontsize=12, fontweight='bold')

    rng = np.random.default_rng(42)
    T = 20
    episodes = 8

    # Left: multiple return trajectories from same policy
    ax = axes[0]
    for ep in range(episodes):
        rewards = rng.normal(0.3, 1.5, T)
        returns = np.array([sum(0.9**(k - t) * rewards[k] for k in range(t, T)) for t in range(T)])
        ax.plot(np.arange(T), returns, alpha=0.6, lw=1.5)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('Return G_t')
    ax.set_title(f'G_t Estimates Across {episodes} Episodes\n(same policy, different trajectories)')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')

    # Right: gradient estimate variance over training
    ax2 = axes[1]
    iters = np.arange(1, 201)
    n_samples = [1, 4, 16, 64]
    colors = ['#BF4C4C', '#E8994C', '#5BAD6F', '#4C8EBF']
    for n, c in zip(n_samples, colors):
        # Variance decreases as 1/n with more trajectories
        std = 5.0 / np.sqrt(n * iters / 5)
        ax2.semilogy(iters, std, color=c, lw=2.0, label=f'{n} trajectories/update')
    ax2.set_xlabel('Training iteration')
    ax2.set_ylabel('Gradient estimate std (log scale)')
    ax2.set_title('Gradient Variance vs Batch Size\n(more trajectories → lower variance)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join(OUT, 'reinforce_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Baseline Variance Reduction (animated) ─────────────────────────
def plot_baseline_anim():
    rng = np.random.default_rng(7)
    FRAMES = 20
    batch_size = 30
    true_mean = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Baseline Subtraction Reduces Gradient Variance\n'
                 'With baseline b: (G_t − b) has same expectation, lower variance',
                 fontsize=11, fontweight='bold')

    ax1, ax2 = axes
    bins = np.linspace(-8, 12, 30)

    returns_no_base = rng.normal(true_mean, 3.0, (FRAMES, batch_size))
    returns_base    = returns_no_base - true_mean + rng.normal(0, 0.8, (FRAMES, batch_size))

    # Static histogram bars (will update data)
    _, _, bar1 = ax1.hist(returns_no_base[0], bins=bins, color='#BF4C4C', alpha=0.7, edgecolor='white')
    ax1.axvline(true_mean, color='black', lw=2, label=f'Mean={true_mean}')
    ax1.set_title('Without Baseline\nG_t  (high variance)')
    ax1.set_xlabel('Return estimate')
    ax1.legend()
    ax1.set_xlim(-8, 12)
    ax1.grid(alpha=0.3)
    std1_text = ax1.text(0.97, 0.95, '', transform=ax1.transAxes, ha='right', va='top', fontsize=10)

    _, _, bar2 = ax2.hist(returns_base[0], bins=bins, color='#5BAD6F', alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='black', lw=2, label='Mean=0 (centered)')
    ax2.set_title('With Baseline  b = V(s)\nG_t − b(s_t)  (low variance)')
    ax2.set_xlabel('Centered return estimate')
    ax2.legend()
    ax2.set_xlim(-8, 12)
    ax2.grid(alpha=0.3)
    std2_text = ax2.text(0.97, 0.95, '', transform=ax2.transAxes, ha='right', va='top', fontsize=10)

    def update_hist(bars, ax, data):
        n, edges = np.histogram(data, bins=bins)
        for rect, h in zip(bars, n):
            rect.set_height(h)

    def update(frame):
        f = frame % FRAMES
        d1 = returns_no_base[f]
        d2 = returns_base[f]
        update_hist(bar1, ax1, d1)
        update_hist(bar2, ax2, d2)
        std1_text.set_text(f'std = {d1.std():.2f}')
        std2_text.set_text(f'std = {d2.std():.2f}')
        return list(bar1) + list(bar2) + [std1_text, std2_text]

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=300, blit=False)
    path = os.path.join(OUT, 'baseline_variance.gif')
    ani.save(path, writer='pillow', fps=3)
    plt.close()
    print(f"Saved: {path}")


# ── Figure 4: Policy Gradient vs Value Learning Comparison ───────────────────
def plot_pg_vs_value_comparison():
    _, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Policy Gradient vs Value-Based Methods: Core Differences',
                 fontsize=13, fontweight='bold')

    def box(cx, cy, w, h, txt, color, fontsize=9.5):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333',
                                    lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', multialignment='center')

    # Column headers
    box(3.2, 7.4, 5.5, 0.8, 'Value-Based (Q-Learning / DQN)', '#4C8EBF', 11)
    box(9.8, 7.4, 5.5, 0.8, 'Policy Gradient (REINFORCE / PPO)', '#5BAD6F', 11)

    # Row labels
    row_labels = ['学习对象', '优化目标', '更新方程', '动作空间', '策略类型', '探索机制']
    row_ys = [6.3, 5.3, 4.3, 3.3, 2.3, 1.3]
    left_vals = [
        'Q_θ(s,a)  — 动作价值函数',
        'min  ½(r + γ·max Q_{θ\'} − Q_θ)²',
        'θ ← θ − α · ∂J/∂θ\n(最小化TD误差)',
        '离散（枚举取 argmax）',
        '确定性（ε-greedy 取 max）',
        'ε-greedy 外部显式探索',
    ]
    right_vals = [
        'π_θ(a|s)  — 策略函数',
        'max  E_π[Σ A^π_θ(s,a)]',
        'θ ← θ + α · ∇_θ log π_θ(a|s) · A\n(最大化期望优势)',
        '连续 + 离散均适用',
        '随机（输出分布，内置随机性）',
        '策略熵自然提供探索',
    ]

    for lbl, y, lv, rv in zip(row_labels, row_ys, left_vals, right_vals):
        # Row label
        ax.add_patch(FancyBboxPatch((0.1, y - 0.38), 1.1, 0.76,
                                    boxstyle='round,pad=0.05', fc='#555555', ec='white', lw=0.8))
        ax.text(0.65, y, lbl, ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        # Left (value-based)
        ax.add_patch(FancyBboxPatch((1.35, y - 0.38), 5.5, 0.76,
                                    boxstyle='square,pad=0', fc='#EBF5FB', ec='#AED6F1', lw=0.8))
        ax.text(4.1, y, lv, ha='center', va='center', fontsize=8.8, color='#1A5276',
                multialignment='center')
        # Right (policy gradient)
        ax.add_patch(FancyBboxPatch((7.15, y - 0.38), 5.5, 0.76,
                                    boxstyle='square,pad=0', fc='#EAFAF1', ec='#A9DFBF', lw=0.8))
        ax.text(9.9, y, rv, ha='center', va='center', fontsize=8.8, color='#1E8449',
                multialignment='center')

    # Bottom note
    ax.text(6.5, 0.5,
            'Actor-Critic = Value-Based (Critic 估计 V/A) + Policy Gradient (Actor 优化 π)',
            ha='center', fontsize=10, color='#333',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1.2))

    plt.tight_layout()
    path = os.path.join(OUT, 'pg_vs_value_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_pg_theorem_flow()
    plot_reinforce_variance()
    plot_baseline_anim()
    plot_pg_vs_value_comparison()
    print("Ch08 done.")
