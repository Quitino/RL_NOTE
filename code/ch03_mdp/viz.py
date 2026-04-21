"""
Ch03: MDP Visualizations
Figures:
  - mdp_structure.png      : MDP 5-tuple diagram
  - discount_return.png    : Discounted return vs gamma curves
  - v_vs_q.png             : State-value V vs Q-value relationship
  - v_q_backup_tree.png    : Backup tree: V(s) = Σ π(a|s)·Q(s,a), Q→next states
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../docs/asserts/ch03_mdp')
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
    actions = ['a1', 'a2', 'a3']
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


# ── Figure 4: V-Q Backup Tree ─────────────────────────────────────────────────
def plot_v_q_backup_tree():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('状态价值 = 按策略对动作价值的加权平均\n'
                 'V(s) = Σ_a  π(a|s) · Q(s,a)',
                 fontsize=13, fontweight='bold', pad=10)

    # ── helpers ──────────────────────────────────────────────────────────────
    def circle(cx, cy, r, color, ec='#333', lw=2.0, alpha=0.92):
        ax.add_patch(plt.Circle((cx, cy), r, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=3))

    def square(cx, cy, w, h, color, ec='#333', lw=1.8, alpha=0.92):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.12', fc=color, ec=ec,
                                    lw=lw, alpha=alpha, zorder=3))

    def arrow(x0, y0, x1, y1, lw=1.8, color='#555', label='', lx=0, ly=0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)
        if label:
            mx = (x0 + x1) / 2 + lx
            my = (y0 + y1) / 2 + ly
            ax.text(mx, my, label, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold',
                    bbox=dict(fc='white', ec='none', pad=1))

    # ── layout constants ──────────────────────────────────────────────────────
    V_X, V_Y = 6.0, 7.0          # V(s) node
    Q_Y = 4.8                    # Q(s,a) row
    S_Y = 2.2                    # next-state row
    Q_XS = [2.5, 6.0, 9.5]      # Q node x positions
    pi_weights = ['π(a1|s)=0.5', 'π(a2|s)=0.3', 'π(a3|s)=0.2']
    pi_lx = [-0.55, 0.0, 0.55]
    Q_colors = ['#E8994C', '#5BAD6F', '#BF4C4C']
    q_vals = ['Q(s,a1)\n= 8.0', 'Q(s,a2)\n= 5.0', 'Q(s,a3)\n= 2.0']

    # ── V(s) node ─────────────────────────────────────────────────────────────
    circle(V_X, V_Y, 0.62, '#4C8EBF')
    ax.text(V_X, V_Y + 0.12, 'V(s)', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold', zorder=4)
    ax.text(V_X, V_Y - 0.28, '= 5.9', ha='center', va='center',
            fontsize=8.5, color='#DDEEFF', zorder=4)

    # ── arrows V → Q with π weights ──────────────────────────────────────────
    for qx, lbl, lx, qc in zip(Q_XS, pi_weights, pi_lx, Q_colors):
        arrow(V_X, V_Y - 0.62, qx, Q_Y + 0.45,
              lw=2.2, color=qc, label=lbl, lx=lx, ly=0.25)

    # ── Q(s,a) nodes ─────────────────────────────────────────────────────────
    for qx, qv, qc in zip(Q_XS, q_vals, Q_colors):
        square(qx, Q_Y, 1.9, 0.85, qc)
        ax.text(qx, Q_Y, qv, ha='center', va='center',
                fontsize=9.5, color='white', fontweight='bold', zorder=4)

    # ── next-state circles per Q node ────────────────────────────────────────
    next_state_groups = [
        ([1.2, 2.5, 3.8], ["s'11", "s'12", "s'13"], [0.4, 0.4, 0.2]),
        ([5.0, 6.0, 7.0], ["s'21", "s'22", "s'23"], [0.5, 0.3, 0.2]),
        ([8.4, 9.5, 10.6], ["s'31", "s'32", "s'33"], [0.6, 0.3, 0.1]),
    ]
    s_color = '#7F8C8D'
    for qx, (sxs, lbls, probs) in zip(Q_XS, next_state_groups):
        for sx, slbl, prob in zip(sxs, lbls, probs):
            circle(sx, S_Y, 0.38, s_color, lw=1.4)
            ax.text(sx, S_Y, slbl, ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold', zorder=4)
            arrow(qx, Q_Y - 0.43, sx, S_Y + 0.38,
                  lw=1.4, color='#888888',
                  label=f'P={prob}', lx=0, ly=0.18)

    # ── formula bar at bottom ─────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.3, 0.25), 11.4, 1.3,
                                boxstyle='round,pad=0.15', fc='#1A252F',
                                ec='#4C8EBF', lw=2.2, alpha=0.95, zorder=3))
    ax.text(6.0, 1.1,
            'V(s)  =  π(a1|s)·Q(s,a1) + π(a2|s)·Q(s,a2) + π(a3|s)·Q(s,a3)',
            ha='center', va='center', fontsize=10.5,
            color='#AED6F1', fontweight='bold', zorder=4)
    ax.text(6.0, 0.58,
            '= 0.5 × 8.0  +  0.3 × 5.0  +  0.2 × 2.0  =  4.0 + 1.5 + 0.4  =  5.9',
            ha='center', va='center', fontsize=10,
            color='#F0FFF0', zorder=4)

    plt.tight_layout()
    path = os.path.join(OUT, 'v_q_backup_tree.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


    plot_mdp_structure()
    plot_discount_return()
    plot_v_vs_q()
    plot_v_q_backup_tree()
    print("Ch03 done.")
