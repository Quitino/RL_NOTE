"""
Ch11: Algorithm Landscape Visualizations
Figures:
  - rl_taxonomy.png          : RL algorithm taxonomy tree
  - sac_entropy.png          : SAC entropy-regularized objective
  - sample_efficiency.png    : Sample efficiency vs stability for major algorithms
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch11_algorithms')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: RL Taxonomy Tree ────────────────────────────────────────────────
def plot_rl_taxonomy():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Reinforcement Learning Algorithm Taxonomy', fontsize=14, fontweight='bold', pad=10)

    def box(cx, cy, w, h, txt, color, fontsize=9):
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                    boxstyle='round,pad=0.12', fc=color, ec='#333',
                                    lw=1.3, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold')

    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color='#555555', lw=1.5)

    # Root
    box(7, 8.3, 4.5, 0.85, 'Reinforcement Learning', '#333333', 11)
    # Level 1
    box(3.5, 6.8, 4.5, 0.8, 'Model-Free', '#4C8EBF', 10)
    box(10.5, 6.8, 4.5, 0.8, 'Model-Based', '#9B59B6', 10)
    line(7, 7.88, 3.5, 7.2); line(7, 7.88, 10.5, 7.2)

    # Model-Free branches
    box(2.0, 5.3, 3.5, 0.8, 'Value-Based', '#4C8EBF')
    box(5.5, 5.3, 3.5, 0.8, 'Policy-Based', '#5BAD6F')
    line(3.5, 6.4, 2.0, 5.7); line(3.5, 6.4, 5.5, 5.7)

    # Value-based algorithms
    box(1.2, 3.8, 2.8, 0.75, 'DQN\nDDQN  Dueling', '#4C8EBF')
    box(1.2, 2.8, 2.8, 0.75, 'C51  Rainbow\nIQN  QR-DQN', '#4C8EBF')
    line(2.0, 4.9, 1.2, 4.18); line(2.0, 4.9, 1.2, 3.18)

    # Policy-based
    box(4.5, 3.8, 2.8, 0.75, 'REINFORCE\nA2C  A3C', '#5BAD6F')
    box(6.5, 3.8, 3.2, 0.75, 'TRPO  PPO\nDDPG  TD3  SAC', '#E8994C')
    line(5.5, 4.9, 4.5, 4.18); line(5.5, 4.9, 6.5, 4.18)
    ax.text(5.7, 3.25, 'Actor-Critic', ha='center', fontsize=8, color='#E8994C', style='italic')

    # Model-Based branches
    box(9.5, 5.3, 3.2, 0.8, 'Learn Model', '#9B59B6')
    box(12.0, 5.3, 2.8, 0.8, 'Use Known\nModel', '#BF4C4C')
    line(10.5, 6.4, 9.5, 5.7); line(10.5, 6.4, 12.0, 5.7)

    box(9.5, 3.8, 3.2, 0.75, 'Dyna  MBPO\nDreamerV3  PETS', '#9B59B6')
    box(12.0, 3.8, 2.8, 0.75, 'AlphaZero\nMuZero  MPC', '#BF4C4C')
    line(9.5, 4.9, 9.5, 4.18); line(12.0, 4.9, 12.0, 4.18)

    # Off-Policy note
    ax.add_patch(FancyBboxPatch((4.0, 0.3), 6.0, 0.85,
                                boxstyle='round,pad=0.1', fc='#FFFDE7', ec='#F0C000', lw=1.5))
    ax.text(7.0, 0.73, 'Off-Policy: DQN, DDPG, TD3, SAC, Rainbow  |  On-Policy: PPO, A2C, TRPO, REINFORCE',
            ha='center', va='center', fontsize=8.5, color='#333')

    plt.tight_layout()
    path = os.path.join(OUT, 'rl_taxonomy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: SAC Entropy Objective ──────────────────────────────────────────
def plot_sac_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Soft Actor-Critic (SAC): Maximum Entropy Reinforcement Learning\n'
                 'π* = argmax_π E[Σ_t r_t + α·H(π(·|s_t))]',
                 fontsize=11, fontweight='bold')

    # Left: entropy of different action distributions
    ax = axes[0]
    actions = np.linspace(-3, 3, 200)
    sigmas = [0.3, 0.8, 1.5, 2.5]
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(sigmas)))
    for sigma, c in zip(sigmas, colors):
        density = np.exp(-0.5 * (actions / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        H = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        ax.plot(actions, density, color=c, lw=2.5, label=f'σ={sigma}  H={H:.2f}')
    ax.fill_between(actions, 0, alpha=0.0)
    ax.set_title('Policy Entropy H(π)\nhigh entropy = more exploration')
    ax.set_xlabel('Action  a')
    ax.set_ylabel('Policy density  π(a|s)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: temperature α effect on reward-entropy tradeoff
    ax2 = axes[1]
    alpha_range = np.linspace(0, 2, 100)
    reward_weight = 1.0 / (1 + alpha_range)
    entropy_weight = alpha_range / (1 + alpha_range)
    ax2.fill_between(alpha_range, reward_weight, color='#4C8EBF', alpha=0.5, label='Reward weight')
    ax2.fill_between(alpha_range, reward_weight, 1.0, color='#E8994C', alpha=0.5, label='Entropy weight')
    ax2.plot(alpha_range, reward_weight, color='#4C8EBF', lw=2.5)
    ax2.plot(alpha_range, 1 - reward_weight, color='#E8994C', lw=2.5)
    ax2.axvline(0.2, color='#5BAD6F', lw=1.5, linestyle='--', label='Typical α=0.2')
    ax2.set_xlabel('Temperature  α  (entropy coefficient)')
    ax2.set_ylabel('Relative weight in objective')
    ax2.set_title('α Controls Exploration-Exploitation\n'
                  'α→0: pure reward maximization,  α→∞: uniform policy')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'sac_entropy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Sample Efficiency vs Stability ──────────────────────────────────
def plot_sample_efficiency():
    fig, ax = plt.subplots(figsize=(10, 6.5))

    algorithms = {
        'REINFORCE':   (0.1, 0.2, '#BF4C4C'),
        'A2C':         (0.2, 0.45, '#E8994C'),
        'PPO':         (0.4, 0.82, '#5BAD6F'),
        'TRPO':        (0.35, 0.75, '#4C8EBF'),
        'DDPG':        (0.65, 0.45, '#9B59B6'),
        'TD3':         (0.70, 0.65, '#1ABC9C'),
        'SAC':         (0.75, 0.80, '#E74C3C'),
        'DQN':         (0.50, 0.55, '#2ECC71'),
        'Rainbow':     (0.68, 0.62, '#F39C12'),
        'DreamerV3':   (0.78, 0.72, '#8E44AD'),
        'MBPO':        (0.85, 0.60, '#2980B9'),
    }

    for name, (se, stab, c) in algorithms.items():
        ax.scatter([se], [stab], color=c, s=220, zorder=5, edgecolors='white', lw=1)
        ax.text(se + 0.015, stab + 0.018, name, fontsize=9.5, color=c, fontweight='bold')

    ax.set_xlabel('Sample Efficiency\n(higher → learns from fewer environment steps)', fontsize=11)
    ax.set_ylabel('Training Stability\n(higher → less hyperparameter sensitivity)', fontsize=11)
    ax.set_title('RL Algorithms: Sample Efficiency vs Stability\n'
                 '(Approximate schematic — actual performance is task-dependent)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    # Quadrant annotations
    ax.axvline(0.5, color='gray', lw=0.8, linestyle='--', alpha=0.4)
    ax.axhline(0.5, color='gray', lw=0.8, linestyle='--', alpha=0.4)
    for tx, ty, txt in [(0.18, 0.92, 'Stable\nbut slow'),
                        (0.78, 0.92, 'Sweet spot\n(target zone)'),
                        (0.78, 0.08, 'Fast but\nunstable')]:
        ax.text(tx, ty, txt, ha='center', fontsize=8, color='#777',
                style='italic', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(OUT, 'sample_efficiency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_rl_taxonomy()
    plot_sac_entropy()
    plot_sample_efficiency()
    print("Ch11 done.")
