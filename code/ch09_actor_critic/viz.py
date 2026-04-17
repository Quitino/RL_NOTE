"""
Ch09: Actor-Critic Visualizations
Figures:
  - actor_critic_arch.png   : Actor-Critic two-network architecture
  - gae_lambda.png          : GAE lambda interpolation diagram
  - advantage_fn.png        : Advantage A(s,a) = Q(s,a) - V(s) visualization
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch09_actor_critic')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Actor-Critic Architecture ──────────────────────────────────────
def plot_actor_critic_arch():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Actor-Critic Architecture\n'
                 'Actor (policy π_θ) + Critic (value V_φ) trained jointly',
                 fontsize=12, fontweight='bold', pad=8)

    def box(cx, cy, w, h, txt, color):
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333', lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=9.5,
                color='white', fontweight='bold')

    def arr(x0, y0, x1, y1, lbl='', c='#444'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.0))
        if lbl:
            ax.text((x0+x1)/2+0.1, (y0+y1)/2+0.15, lbl, fontsize=8.5, color=c)

    # Environment
    box(6, 6.3, 3.0, 0.9, 'Environment', '#888888')
    # State node
    box(6, 4.9, 2.8, 0.8, 'State  s_t', '#4C8EBF')
    # Actor
    box(3, 3.2, 3.2, 1.0, 'Actor  π_θ(a|s)\n(policy network)', '#E8994C')
    # Critic
    box(9, 3.2, 3.2, 1.0, 'Critic  V_φ(s)\n(value network)', '#9B59B6')
    # Action
    box(3, 1.6, 2.4, 0.8, 'Action  a_t', '#E8994C')
    # Advantage
    box(9, 1.6, 3.2, 0.8, 'Advantage  A_t\n= r + γV(s\')-V(s)', '#5BAD6F')

    arr(6, 5.9, 6, 5.3, '')                   # env -> state
    arr(6, 4.5, 3, 3.7, 'observe s', '#444')  # state -> actor
    arr(6, 4.5, 9, 3.7, 'observe s', '#444')  # state -> critic
    arr(3, 2.7, 3, 2.0, '')                    # actor -> action
    arr(9, 2.7, 9, 2.0, '')                    # critic -> advantage
    arr(3, 1.2, 6, 5.9, '', '#E8994C')         # action -> env (loop back) -- schematic
    # Update arrows
    arr(9, 1.2, 3, 2.7, 'A_t updates actor', '#5BAD6F')
    arr(9, 1.2, 9, 2.7, 'TD error updates critic', '#5BAD6F')

    # Gradient labels
    ax.text(2.0, 2.3,
            'Actor grad:\n∇_θ log π_θ(a|s) · A_t',
            ha='center', fontsize=8, color='#E8994C',
            bbox=dict(fc='#FFF5E6', ec='#E8994C', boxstyle='round,pad=0.3'))
    ax.text(10.2, 0.4,
            'Critic loss:\n(A_t)² = (r+γV(s\')-V(s))²',
            ha='center', fontsize=8, color='#9B59B6',
            bbox=dict(fc='#F5F0FF', ec='#9B59B6', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    path = os.path.join(OUT, 'actor_critic_arch.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: GAE Lambda Interpolation ───────────────────────────────────────
def plot_gae_lambda():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Generalized Advantage Estimation (GAE)\n'
                 'A^GAE(λ) = Σ_l (γλ)^l · δ_{t+l}   where  δ_t = r_t + γV(s_{t+1}) - V(s_t)',
                 fontsize=11, fontweight='bold')

    # Left: lambda blending visualization
    ax = axes[0]
    lambdas = np.linspace(0, 1, 100)

    # Schematic: bias and variance as function of lambda
    bias     = lambdas ** 0.5                    # increases with lambda (more MC)
    variance = (1 - lambdas) ** 0.8              # decreases with lambda
    tradeoff = bias**2 + variance**2             # total (schematic)
    opt_lam  = lambdas[np.argmin(tradeoff)]

    ax.plot(lambdas, bias,     color='#4C8EBF', lw=2, label='Bias (↑ with λ)')
    ax.plot(lambdas, variance, color='#E8994C', lw=2, label='Variance (↓ with λ)')
    ax.plot(lambdas, tradeoff, color='#BF4C4C', lw=2.5, linestyle='--', label='Total error')
    ax.axvline(opt_lam, color='#5BAD6F', lw=1.5, linestyle=':', label=f'Optimal λ ≈ {opt_lam:.2f}')
    ax.scatter([0], [variance[0]], s=100, color='#E8994C', zorder=5)
    ax.scatter([1], [bias[-1]],    s=100, color='#4C8EBF', zorder=5)
    ax.text(0.05, variance[0] + 0.05, 'TD(0)\nλ=0', fontsize=8.5, color='#E8994C')
    ax.text(0.88, bias[-1] + 0.05,   'MC\nλ=1',   fontsize=8.5, color='#4C8EBF')
    ax.set_xlabel('λ')
    ax.set_ylabel('Error (schematic)')
    ax.set_title('GAE λ: Bias-Variance Tradeoff')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0)

    # Right: δ weights for different lambda
    ax2 = axes[1]
    T = 15
    t_arr = np.arange(T)
    gamma = 0.99
    for lam, c in zip([0.0, 0.5, 0.9, 1.0], ['#4C8EBF', '#5BAD6F', '#E8994C', '#BF4C4C']):
        weights = (gamma * lam) ** t_arr
        ax2.plot(t_arr, weights, color=c, lw=2.5, marker='o', markersize=4,
                 label=f'λ={lam}')
    ax2.set_xlabel('Lookahead steps l')
    ax2.set_ylabel('Weight  (γλ)^l  of δ_{t+l}')
    ax2.set_title('TD Residual Weights in GAE Sum')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(OUT, 'gae_lambda.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Advantage Function Visualization ───────────────────────────────
def plot_advantage_fn():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Advantage Function  A(s,a) = Q(s,a) − V(s)\n'
                 '"How much better is action a than the average action at state s?"',
                 fontsize=11, fontweight='bold')

    n_actions = 5
    action_labels = [f'a_{i}' for i in range(n_actions)]
    rng = np.random.default_rng(3)

    Q = rng.normal(2.0, 1.5, n_actions)
    V = np.mean(Q)                        # schematic
    A = Q - V

    colors_q = ['#4C8EBF'] * n_actions
    colors_a = ['#5BAD6F' if a >= 0 else '#BF4C4C' for a in A]

    # Q values
    axes[0].bar(action_labels, Q, color=colors_q, edgecolor='white')
    axes[0].axhline(V, color='#E8994C', lw=2.5, linestyle='--', label=f'V(s) = {V:.2f}')
    axes[0].set_title('Q(s, a) values')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # V value
    axes[1].barh(['V(s)'], [V], color='#E8994C', height=0.5)
    axes[1].set_title('V(s) = E_π[Q(s,a)]')
    axes[1].set_xlabel('Value')
    axes[1].grid(axis='x', alpha=0.3)

    # Advantage
    axes[2].bar(action_labels, A, color=colors_a, edgecolor='white')
    axes[2].axhline(0, color='black', lw=1.5)
    axes[2].set_title('A(s, a) = Q(s,a) − V(s)\n(green=above avg, red=below avg)')
    axes[2].set_ylabel('Advantage')
    axes[2].grid(axis='y', alpha=0.3)
    for i, a_val in enumerate(A):
        axes[2].text(i, a_val + 0.05 * np.sign(a_val), f'{a_val:.2f}',
                     ha='center', fontsize=8.5, fontweight='bold',
                     color='#1a5c1a' if a_val >= 0 else '#8B0000')

    plt.tight_layout()
    path = os.path.join(OUT, 'advantage_fn.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_actor_critic_arch()
    plot_gae_lambda()
    plot_advantage_fn()
    print("Ch09 done.")
