"""
Ch06: Q-Learning and Sarsa Visualizations
Figures:
  - epsilon_greedy.png       : Epsilon-greedy exploration decay schedule
  - on_off_policy.png        : On-policy vs off-policy data flow diagram
  - q_vs_sarsa.gif           : Q-Learning vs Sarsa update animation
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch06_qlearning')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Epsilon-Greedy Schedule ────────────────────────────────────────
def plot_epsilon_greedy():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('ε-greedy Exploration Strategy', fontsize=13, fontweight='bold')

    steps = np.arange(0, 1001)

    # Left: different decay strategies
    ax = axes[0]
    eps_linear = np.maximum(0.05, 1.0 - steps / 500)
    eps_exp    = 0.05 + 0.95 * np.exp(-steps / 200)
    eps_const  = np.full_like(steps, 0.1, dtype=float)

    ax.plot(steps, eps_linear, color='#4C8EBF', lw=2.5, label='Linear decay')
    ax.plot(steps, eps_exp,    color='#E8994C', lw=2.5, label='Exponential decay')
    ax.plot(steps, eps_const,  color='#5BAD6F', lw=2.0, linestyle='--', label='Constant ε=0.1')
    ax.axhline(0.05, color='gray', lw=1.0, linestyle=':', label='ε_min = 0.05')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Exploration rate  ε')
    ax.set_title('ε Decay Schedule')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Right: action selection illustration at different ε
    ax2 = axes[1]
    epsilons = [0.9, 0.5, 0.1]
    n_actions = 4
    q_values = np.array([2.1, 3.5, 1.2, 0.8])   # Q values for 4 actions
    greedy_a = np.argmax(q_values)
    bar_colors = ['#E8994C'] * n_actions
    bar_colors[greedy_a] = '#5BAD6F'

    y_offsets = [0.7, 0.4, 0.1]
    heights = [0.22] * 3
    for eps, y0, h in zip(epsilons, y_offsets, heights):
        probs = np.full(n_actions, eps / n_actions)
        probs[greedy_a] += 1 - eps
        for a, p in enumerate(probs):
            c = '#5BAD6F' if a == greedy_a else '#4C8EBF'
            ax2.barh(y0, p, height=h * 0.9, left=sum(probs[:a]),
                     color=c, edgecolor='white', lw=0.8)
        ax2.text(-0.02, y0 + h * 0.45, f'ε={eps}', ha='right', va='center', fontsize=9, color='#333')

    ax2.axvline(1.0, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlim(-0.15, 1.1)
    ax2.set_yticks([])
    ax2.set_xlabel('Probability')
    ax2.set_title('Action Selection Probability\n(green = greedy, blue = random)')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'epsilon_greedy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: On-Policy vs Off-Policy ────────────────────────────────────────
def plot_on_off_policy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('On-Policy (Sarsa) vs Off-Policy (Q-Learning)', fontsize=13, fontweight='bold')

    def draw_diagram(ax, title, color, boxes, arrows, note):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=8)
        for (x, y, w, h, txt) in boxes:
            ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle='round,pad=0.15', fc=color, ec='#333333',
                                        lw=1.5, alpha=0.85))
            ax.text(x, y, txt, ha='center', va='center', fontsize=9.5,
                    color='white', fontweight='bold')
        for (x0, y0, x1, y1, lbl) in arrows:
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8))
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.2, my, lbl, fontsize=8.5, color='#444444')
        ax.text(5.0, 0.5, note, ha='center', fontsize=8.5, style='italic',
                color='#555555', bbox=dict(fc='#FFFDE7', ec='#F0C000',
                                           boxstyle='round,pad=0.3', lw=0.8))

    # Sarsa (on-policy)
    draw_diagram(axes[0], 'Sarsa  (On-Policy)', '#4C8EBF',
                 boxes=[
                     (5, 6.2, 4.0, 0.9, 'Behavior Policy π_ε (ε-greedy)'),
                     (5, 4.8, 3.5, 0.9, 'Sample (s,a,r,s\',a\')'),
                     (5, 3.4, 3.5, 0.9, 'Update Q with a\'=π(s\')'),
                     (5, 2.0, 3.5, 0.9, 'Target Policy = π_ε (same!)'),
                 ],
                 arrows=[
                     (5, 5.75, 5, 5.25, ''),
                     (5, 4.35, 5, 3.85, ''),
                     (5, 2.95, 5, 2.45, ''),
                 ],
                 note='Update uses action sampled from same policy → on-policy')

    # Q-Learning (off-policy)
    draw_diagram(axes[1], 'Q-Learning  (Off-Policy)', '#E8994C',
                 boxes=[
                     (5, 6.2, 4.0, 0.9, 'Behavior Policy π_ε (ε-greedy)'),
                     (5, 4.8, 3.5, 0.9, 'Sample (s, a, r, s\')'),
                     (5, 3.4, 3.5, 0.9, 'Update Q with max_a Q(s\',a)'),
                     (5, 2.0, 3.5, 0.9, 'Target Policy = greedy (different!)'),
                 ],
                 arrows=[
                     (5, 5.75, 5, 5.25, ''),
                     (5, 4.35, 5, 3.85, ''),
                     (5, 2.95, 5, 2.45, ''),
                 ],
                 note='Update uses max_a (greedy) → off-policy: data from any π works')

    plt.tight_layout()
    path = os.path.join(OUT, 'on_off_policy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Q-Learning vs Sarsa on Grid World (animated) ───────────────────
def plot_q_vs_sarsa_anim():
    GRID = 4
    rng = np.random.default_rng(0)

    # Simulate simplified Q and Sarsa value convergence
    def simulate(q_method, episodes=60):
        Q = np.zeros((GRID * GRID, 4))
        gamma, alpha, eps = 0.9, 0.3, 0.3
        goal = GRID * GRID - 1
        history = [Q.copy()]
        for ep in range(episodes):
            s = 0
            a = rng.integers(4)
            for _ in range(20):
                r = 1.0 if s == goal else -0.01
                ns = min(GRID * GRID - 1, s + 1) if a in (1, 2) else max(0, s - 1)
                na = rng.integers(4)
                if q_method == 'qlearn':
                    Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
                else:
                    Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
                s, a = ns, na
                if s == goal:
                    break
            if ep % 5 == 0:
                history.append(Q.copy())
        return history

    hist_q    = simulate('qlearn')
    hist_sarsa = simulate('sarsa')
    FRAMES = min(len(hist_q), len(hist_sarsa), 12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Q-Learning vs Sarsa: Value Convergence on Grid World', fontweight='bold')

    ims = []
    for ax, hist, title in zip(axes, [hist_q, hist_sarsa], ['Q-Learning', 'Sarsa']):
        V = np.max(hist[0], axis=1).reshape(GRID, GRID)
        im = ax.imshow(V, cmap='YlOrRd', vmin=-0.5, vmax=1.0)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
        ims.append(im)

    ep_text = fig.text(0.5, 0.01, 'Episode 0', ha='center', fontsize=11, style='italic')

    def update(frame):
        f = min(frame, FRAMES - 1)
        for im, hist in zip(ims, [hist_q, hist_sarsa]):
            V = np.max(hist[f], axis=1).reshape(GRID, GRID)
            im.set_data(V)
        ep_text.set_text(f'Episode {f * 5}')
        return ims + [ep_text]

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=500, blit=False)
    path = os.path.join(OUT, 'q_vs_sarsa.gif')
    ani.save(path, writer='pillow', fps=2)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_epsilon_greedy()
    plot_on_off_policy()
    plot_q_vs_sarsa_anim()
    print("Ch06 done.")
