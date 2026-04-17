"""
Ch06: Q-Learning and Sarsa Visualizations
Figures:
  - epsilon_greedy.png       : Epsilon-greedy exploration decay schedule
  - on_off_policy.png        : On-policy vs off-policy data flow diagram
  - q_vs_sarsa.gif           : Q-Learning vs Sarsa update animation
  - q_table_structure.png    : Q-table states×actions matrix with update loop
  - on_off_policy_table.png  : On-Policy vs Off-Policy comparison table
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

OUT = os.path.join(os.path.dirname(__file__), '../../docs/asserts/ch06_qlearning')
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


# ── Figure 4: Q-Table Structure ──────────────────────────────────────────────
def plot_q_table_structure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Q-Learning: Q-Table Structure and Update Mechanism', fontsize=13, fontweight='bold')

    # Left: Q-table grid
    ax = axes[0]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.0, 5.5)
    ax.axis('off')
    ax.set_title('Q-Table: Q(s, a) for all state-action pairs', fontsize=10, fontweight='bold')

    n_states = 4
    n_actions = 3
    action_labels = ['a₀', 'a₁', 'a₂']
    state_labels = ['S₀', 'S₁', 'S₂', 'S₃']
    # Sample Q values
    q_vals = np.array([[0.2, 0.8, 0.1],
                        [0.5, 0.3, 0.9],
                        [0.7, 0.4, 0.2],
                        [0.1, 0.6, 0.4]])
    cmap = plt.cm.YlOrRd

    # Header row
    for j, albl in enumerate([''] + action_labels):
        x = j
        ax.add_patch(FancyBboxPatch((x, n_states), 0.9, 0.6,
                                    boxstyle='square,pad=0.0', fc='#4C8EBF', ec='white', lw=1.5))
        ax.text(x + 0.45, n_states + 0.3, albl, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    for i in range(n_states):
        y = n_states - 1 - i
        # State label column
        ax.add_patch(FancyBboxPatch((0, y), 0.9, 0.85,
                                    boxstyle='square,pad=0.0', fc='#9B59B6', ec='white', lw=1.5))
        ax.text(0.45, y + 0.42, state_labels[i], ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        for j in range(n_actions):
            v = q_vals[i, j]
            c = cmap(v)
            ax.add_patch(FancyBboxPatch((j + 1, y), 0.9, 0.85,
                                        boxstyle='square,pad=0.0', fc=c, ec='white', lw=1.5))
            ax.text(j + 1.45, y + 0.42, f'{v:.1f}', ha='center', va='center', fontsize=10)

    # Highlight best action for S1
    best_j = np.argmax(q_vals[1])
    ax.add_patch(FancyBboxPatch((best_j + 1, n_states - 2), 0.9, 0.85,
                                boxstyle='square,pad=0.0', fc='none', ec='#E8994C', lw=3))
    ax.text(best_j + 1.45, n_states - 2 - 0.35, 'max', ha='center', va='center',
            fontsize=8.5, color='#E8994C', fontweight='bold')

    ax.text(2.5, -0.6, 'Policy: π(s) = argmax_a Q(s, a)', ha='center', fontsize=9.5,
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.3'))

    # Right: Q-update loop diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7)
    ax2.axis('off')
    ax2.set_title('Q-Learning Update Rule', fontsize=10, fontweight='bold')

    def box2(cx, cy, w, h, txt, color):
        ax2.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                     boxstyle='round,pad=0.15', fc=color, ec='#333', lw=1.5, alpha=0.9))
        ax2.text(cx, cy, txt, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    def arr2(x0, y0, x1, y1, lbl=''):
        ax2.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', color='#444', lw=2.0))
        if lbl:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax2.text(mx + 0.15, my, lbl, fontsize=8, color='#555')

    box2(5, 6.3, 6.5, 0.85, 'Observe state s, choose action a (ε-greedy)', '#4C8EBF')
    arr2(5, 5.88, 5, 5.28, '')
    box2(5, 4.9, 6.5, 0.75, 'Execute a → get reward r, next state s\'', '#9B59B6')
    arr2(5, 4.52, 5, 3.92, '')
    box2(5, 3.55, 7.5, 0.75, 'Compute TD target:  y = r + γ · max_{a\'} Q(s\', a\')', '#5BAD6F')
    arr2(5, 3.17, 5, 2.57, '')
    box2(5, 2.2, 7.5, 0.75, 'Update:  Q(s,a) ← Q(s,a) + α·(y − Q(s,a))', '#E8994C')
    arr2(5, 1.82, 5, 1.22, '')
    box2(5, 0.85, 4.5, 0.65, 's ← s\'  (advance to next state)', '#4C8EBF')

    # TD error bracket
    ax2.annotate('', xy=(8.8, 2.2), xytext=(8.8, 3.55),
                 arrowprops=dict(arrowstyle='<->', color='#BF4C4C', lw=1.8))
    ax2.text(9.2, 2.87, 'TD\nerror', ha='left', fontsize=8.5, color='#BF4C4C', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT, 'q_table_structure.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 5: On/Off-Policy Comparison Table ──────────────────────────────────
def plot_on_off_policy_table():
    _, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('On-Policy vs Off-Policy: Comprehensive Comparison', fontsize=13, fontweight='bold')

    headers = ['Property', 'On-Policy (同策略)', 'Off-Policy (异策略)']
    rows = [
        ['定义', '行为策略 = 目标策略', '行为策略 ≠ 目标策略'],
        ['数据使用', '只能用当前策略采集的数据', '可用历史/任意策略数据（经验回放）'],
        ['数据一致性', '好，无分布偏移', '需要重要性采样修正偏差'],
        ['样本效率', '低（数据用完即弃）', '高（数据可复用）'],
        ['收敛性', '理论收敛性好', '需处理IS偏差，实现更复杂'],
        ['适用场景', '仿真充足，高成本交互', '真机实验，成本高的环境'],
        ['代表算法', 'SARSA, PPO, TRPO', 'Q-Learning, DQN, DDPG, SAC'],
    ]

    col_widths = [3.5, 4.5, 4.5]
    col_xs = [0.3, 4.1, 8.8]
    row_h = 0.72
    header_y = 6.2

    # Header
    header_colors = ['#555555', '#4C8EBF', '#E8994C']
    for hdr, cx, cw, hc in zip(headers, col_xs, col_widths, header_colors):
        ax.add_patch(FancyBboxPatch((cx, header_y - 0.35), cw, 0.7,
                                    boxstyle='round,pad=0.08', fc=hc, ec='white', lw=1))
        ax.text(cx + cw/2, header_y, hdr, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # Rows
    for ri, row in enumerate(rows):
        y = header_y - (ri + 1) * row_h - 0.1
        bg = '#F8F9FA' if ri % 2 == 0 else '#FFFFFF'
        for ci, (cell, cx, cw) in enumerate(zip(row, col_xs, col_widths)):
            ax.add_patch(FancyBboxPatch((cx, y - 0.32), cw, 0.64,
                                        boxstyle='square,pad=0.0', fc=bg,
                                        ec='#DDDDDD', lw=0.8))
            fc = '#333333' if ci == 0 else ('#4C8EBF' if ci == 1 else '#E8994C')
            fw = 'bold' if ci == 0 else 'normal'
            ax.text(cx + cw/2, y, cell, ha='center', va='center',
                    fontsize=8.8, color=fc, fontweight=fw)

    plt.tight_layout()
    path = os.path.join(OUT, 'on_off_policy_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_epsilon_greedy()
    plot_on_off_policy()
    plot_q_vs_sarsa_anim()
    plot_q_table_structure()
    plot_on_off_policy_table()
    print("Ch06 done.")
