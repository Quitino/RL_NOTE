"""
Ch03: MDP Visualizations
Figures:
  - mdp_structure.png      : MDP 5-tuple diagram
  - discount_return.png    : Discounted return vs gamma curves
  - v_vs_q.png             : State-value V vs Q-value relationship
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch03_mdp')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: MDP Structure ───────────────────────────────────────────────────
def plot_mdp_structure():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('MDP: Markov Decision Process  M = (S, A, P, R, γ)',
                 fontsize=13, fontweight='bold', pad=10)

    def box(ax, x, y, w, h, label, sub, color):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333333', lw=1.8, alpha=0.92))
        ax.text(x, y + 0.15, label, ha='center', va='center', fontsize=12,
                color='white', fontweight='bold')
        ax.text(x, y - 0.35, sub, ha='center', va='center', fontsize=8, color='#EEEEEE')

    box(ax, 2.0, 4.5, 2.8, 1.4, 'State  s',  'S = state space (continuous/discrete)', '#4C8EBF')
    box(ax, 5.5, 5.5, 2.8, 1.0, 'Action  a', 'A = action space',                      '#E8994C')
    box(ax, 9.0, 4.5, 2.8, 1.4, "Next State  s'", "P(s'|s,a) = transition model",     '#5BAD6F')
    box(ax, 5.5, 2.5, 2.8, 1.0, 'Reward  r',      'R(s,a,s\') = immediate feedback',  '#BF4C4C')
    box(ax, 5.5, 0.9, 2.8, 0.9, 'Discount  γ ∈ [0,1)', 'controls future weight',      '#9B59B6')

    # Arrows
    kw = dict(arrowstyle='->', lw=2.0, color='#555555')
    ax.annotate('', xy=(5.5 - 1.4, 5.5), xytext=(2.0 + 1.4, 4.5 + 0.3), arrowprops=kw)
    ax.annotate('', xy=(9.0 - 1.4, 4.5), xytext=(5.5 + 1.4, 5.5 - 0.3), arrowprops=kw)
    ax.annotate('', xy=(5.5, 2.5 + 0.5), xytext=(5.5, 5.5 - 0.5), arrowprops=kw)
    # feedback loop s' -> s
    ax.annotate('', xy=(2.0 + 1.4, 4.5 - 0.3), xytext=(9.0 - 1.4, 4.5 - 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='#3a7a4a',
                                connectionstyle='arc3,rad=0.4'))
    ax.text(5.5, 3.55, "s' becomes next s", ha='center', fontsize=8,
            color='#3a7a4a', style='italic')

    # Return formula
    ax.text(5.5, 6.7,
            r'Return:  G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_k  γ^k · r_{t+k}',
            ha='center', fontsize=10, color='#333333',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, 'mdp_structure.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Discounted Return vs Gamma ─────────────────────────────────────
def plot_discount_return():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: reward weights over time
    ax = axes[0]
    steps = np.arange(0, 30)
    gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gammas)))
    for g, c in zip(gammas, colors):
        ax.plot(steps, g**steps, color=c, lw=2.0, label=f'γ = {g}')
    ax.set_xlabel('Time step  k')
    ax.set_ylabel('Discount weight  γ^k')
    ax.set_title('Future Reward Discount Weights\nγ^k for various γ', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Right: effective horizon  1/(1-gamma)
    ax2 = axes[1]
    g_range = np.linspace(0.5, 0.995, 200)
    horizon = 1.0 / (1.0 - g_range)
    ax2.plot(g_range, horizon, color='#4C8EBF', lw=2.5)
    ax2.fill_between(g_range, horizon, alpha=0.15, color='#4C8EBF')
    for gv in [0.9, 0.95, 0.99]:
        h = 1 / (1 - gv)
        ax2.scatter([gv], [h], color='#E8994C', s=80, zorder=5)
        ax2.text(gv + 0.003, h + 5, f'γ={gv}\n→ H≈{h:.0f}', fontsize=8, color='#E8994C')
    ax2.set_xlabel('Discount factor  γ')
    ax2.set_ylabel('Effective horizon  1/(1−γ)')
    ax2.set_title('Effective Planning Horizon\nH ≈ 1/(1−γ)', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.5, 1.0)

    plt.tight_layout()
    path = os.path.join(OUT, 'discount_return.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: V vs Q Relationship ────────────────────────────────────────────
def plot_v_vs_q():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('State-Value V(s) vs Action-Value Q(s,a) Relationship',
                 fontsize=12, fontweight='bold', pad=10)

    def box(cx, cy, w, h, txt, color, alpha=0.9):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333333',
                                    lw=1.5, alpha=alpha))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    # State node
    box(1.5, 3.0, 2.0, 1.0, 'State  s', '#4C8EBF')

    # Action nodes
    action_colors = ['#E8994C', '#5BAD6F', '#BF4C4C']
    actions = ['a₁', 'a₂', 'a₃']
    action_ys = [4.5, 3.0, 1.5]
    for a, ay, ac in zip(actions, action_ys, action_colors):
        box(4.5, ay, 1.6, 0.8, f'Q(s, {a})', ac)
        ax.annotate('', xy=(4.5 - 0.8, ay), xytext=(1.5 + 1.0, 3.0),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='#555555'))
        ax.text(3.0, (ay + 3.0) / 2 + 0.1, a, ha='center', fontsize=9, color='#555555')

    # V = sum over actions
    box(7.5, 3.0, 2.5, 1.1,
        'V(s) = Σ_a π(a|s)·Q(s,a)', '#9B59B6')
    for ay in action_ys:
        ax.annotate('', xy=(7.5 - 1.25, 3.0), xytext=(4.5 + 0.8, ay),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#9B59B6', alpha=0.7))

    # Bellman backup symbol
    ax.text(5.0, 0.5,
            'V(s) = max_a Q(s,a)   [under greedy policy π*]',
            ha='center', fontsize=10, color='#333333',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, 'v_vs_q.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_mdp_structure()
    plot_discount_return()
    plot_v_vs_q()
    print("Ch03 done.")
