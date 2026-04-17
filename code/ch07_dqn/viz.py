"""
Ch07: DQN Visualizations
Figures:
  - dqn_architecture.png      : Neural Q-network architecture
  - replay_buffer.png         : Experience replay buffer FIFO diagram
  - target_network_update.png : Target network hard-update schedule
  - dqn_stability.png         : Training stability (with vs without target net)
  - dqn_networks.png          : DQN dual-network + replay buffer overview
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

OUT = os.path.join(os.path.dirname(__file__), '../../docs/asserts/ch07_dqn')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: DQN Architecture ───────────────────────────────────────────────
def plot_dqn_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Deep Q-Network (DQN) Architecture\n'
                 'Q_θ(s, a)  approximates  Q*(s, a)',
                 fontsize=12, fontweight='bold', pad=8)

    layers = [
        (1.0, 'Input\nState s\n(48-dim)', '#4C8EBF', 4),
        (3.2, 'FC Layer 1\n256 units\nReLU', '#9B59B6', 3),
        (5.6, 'FC Layer 2\n256 units\nReLU', '#9B59B6', 3),
        (8.0, 'FC Layer 3\n128 units\nReLU', '#9B59B6', 3),
        (10.5, 'Output\nQ(s, a₁)...\nQ(s, aₙ)', '#5BAD6F', 4),
    ]

    prev_x, prev_h = None, None
    for x, label, color, h in layers:
        ax.add_patch(FancyBboxPatch((x - 0.55, 3 - h / 2), 1.1, h,
                                    boxstyle='round,pad=0.1', fc=color, ec='#333',
                                    lw=1.5, alpha=0.9))
        ax.text(x, 3, label, ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold')
        if prev_x is not None:
            ax.annotate('', xy=(x - 0.55, 3), xytext=(prev_x + 0.55, 3),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0))
        prev_x, prev_h = x, h

    # Loss formula
    ax.text(6.0, 0.6,
            'Loss = E[(r + γ·max_{a\'} Q_{θ⁻}(s\',a\') − Q_θ(s,a))²]\n'
            '       TD target uses frozen  θ⁻  (target network)',
            ha='center', fontsize=9.5, color='#333',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, 'dqn_architecture.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Experience Replay Buffer ───────────────────────────────────────
def plot_replay_buffer():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Experience Replay Buffer  (capacity N)\n'
                 'Breaks temporal correlation → i.i.d. mini-batches for training',
                 fontsize=11, fontweight='bold')

    N = 10
    slot_w = 0.9
    buf_x0 = 1.0
    # Draw buffer slots
    for i in range(N):
        x = buf_x0 + i * slot_w
        age = N - i
        alpha = 0.3 + 0.6 * (i / N)
        ax.add_patch(FancyBboxPatch((x, 2.0), slot_w * 0.88, 1.2,
                                    boxstyle='square,pad=0.05', fc='#4C8EBF',
                                    ec='#2c5f8a', lw=1.0, alpha=alpha))
        ax.text(x + slot_w * 0.44, 2.6, f'(s,a,r,s\')\n{i+1}', ha='center', va='center',
                fontsize=7, color='white')
    ax.text(buf_x0 + 0, 1.6, 'Oldest', ha='center', fontsize=8.5, color='#777')
    ax.text(buf_x0 + (N - 1) * slot_w, 1.6, 'Newest', ha='center', fontsize=8.5, color='#777')

    # New experience arrow
    ax.annotate('', xy=(buf_x0 + (N - 1) * slot_w + slot_w * 0.44, 3.2 + 0.3),
                xytext=(buf_x0 + (N - 1) * slot_w + slot_w * 0.44, 4.0),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=2.5))
    ax.text(buf_x0 + (N - 1) * slot_w + slot_w * 0.44, 4.3,
            'New experience\n(s_t, a_t, r_t, s_{t+1})', ha='center', fontsize=8.5, color='#5BAD6F')

    # Random sample arrows
    sample_idxs = [1, 4, 7]
    for si in sample_idxs:
        x = buf_x0 + si * slot_w + slot_w * 0.44
        ax.annotate('', xy=(x, 0.8), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#E8994C', lw=1.8))
    ax.text(buf_x0 + 4 * slot_w, 0.45, 'Random mini-batch  (breaks correlation)',
            ha='center', fontsize=9.5, color='#E8994C', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT, 'replay_buffer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Target Network Update Schedule ──────────────────────────────────
def plot_target_network():
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(0, 1001)
    C = 100   # update every C steps

    ax.plot(steps, np.sin(steps * 0.008) * 0.4 + 0.5 + steps * 0.0003,
            color='#4C8EBF', lw=2.0, label='Online network θ (updates every step)')

    # Target network: step function updated every C steps
    target_vals = []
    online_ref = np.sin(steps * 0.008) * 0.4 + 0.5 + steps * 0.0003
    for t in steps:
        snap = (t // C) * C
        target_vals.append(online_ref[snap])
    ax.step(steps, target_vals, color='#E8994C', lw=2.5, where='post',
            label=f'Target network θ⁻ (hard copy every {C} steps)')

    for t in range(0, 1001, C):
        ax.axvline(t, color='#5BAD6F', lw=0.9, linestyle='--', alpha=0.6)

    ax.set_xlabel('Training steps')
    ax.set_ylabel('Network parameter magnitude (schematic)')
    ax.set_title('Target Network θ⁻: Hard Update Every C Steps\n'
                 'Stabilizes TD targets → prevents moving-target divergence',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, 'target_network_update.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 4: Training Stability Comparison ───────────────────────────────────
def plot_dqn_stability():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('DQN Stability: Effect of Target Network + Replay Buffer',
                 fontsize=12, fontweight='bold')

    rng = np.random.default_rng(1)
    ep = np.arange(200)

    # Without target net: unstable oscillating loss
    loss_unstable = (10 * np.exp(-ep / 80)
                     + 2 * np.sin(ep * 0.5)
                     + rng.normal(0, 2.0, 200)
                     + np.maximum(0, (ep - 80) * 0.02))

    # With target net: smooth convergence
    loss_stable = (8 * np.exp(-ep / 60)
                   + rng.normal(0, 0.3, 200)
                   + 0.2)

    axes[0].plot(ep, np.maximum(0, loss_unstable), color='#BF4C4C', lw=1.8,
                 label='No target network')
    axes[0].set_title('Without Target Network\n(divergence / oscillation)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('TD Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, np.maximum(0, loss_stable), color='#5BAD6F', lw=1.8,
                 label='With target network')
    axes[1].set_title('With Target Network\n(stable convergence)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('TD Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'dqn_stability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 5: DQN Dual-Network + Replay Buffer Overview ──────────────────────
def plot_dqn_networks():
    _, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('DQN Architecture: Dual Networks + Experience Replay\n'
                 'Two key innovations that stabilize deep Q-learning',
                 fontsize=12, fontweight='bold', pad=8)

    def box(cx, cy, w, h, txt, color, fontsize=9.5):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=color, ec='#333',
                                    lw=1.5, alpha=0.9))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', multialignment='center')

    def arr(x0, y0, x1, y1, lbl='', lw=2.0, color='#444444'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        if lbl:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx + 0.2, my, lbl, fontsize=8, color=color)

    # Environment
    box(1.5, 4.0, 2.2, 1.0, 'Environment', '#555555', 10)
    # Online network
    box(5.5, 5.8, 3.0, 1.1, 'Online Network\nQ_θ(s, a)', '#4C8EBF', 10)
    # Target network
    box(5.5, 2.2, 3.0, 1.1, 'Target Network\nQ_{θ⁻}(s\', a\')', '#E8994C', 10)
    # Replay buffer
    box(9.5, 4.0, 2.8, 2.8, 'Replay\nBuffer D\n(s,a,r,s\')\n× N', '#9B59B6', 10)
    # Loss
    box(5.5, 4.0, 2.5, 1.1, 'MSE Loss\nL = (y − Q_θ)²', '#BF4C4C', 10)

    # Arrows
    arr(2.6, 4.0, 3.85, 5.4, 'state s', color='#4C8EBF')   # env → online
    arr(2.6, 4.0, 3.85, 2.6, 'state s\'', color='#E8994C')  # env → target
    arr(7.0, 5.8, 8.1, 4.7, 'Q_θ(s,a)', color='#4C8EBF')   # online → loss
    arr(7.0, 2.2, 8.1, 3.3, 'max Q_{θ⁻}', color='#E8994C') # target → loss
    arr(8.1, 4.0, 6.75, 4.0, '', color='#BF4C4C')           # loss → online (backprop)
    ax.text(7.1, 4.25, 'backprop\nupdate θ', ha='center', fontsize=8, color='#BF4C4C')

    # Replay buffer arrows
    arr(2.6, 4.4, 8.1, 4.7, '(s,a,r,s\') stored', color='#9B59B6', lw=1.5)
    arr(8.1, 3.5, 6.75, 5.5, 'random\nmini-batch', color='#9B59B6', lw=1.5)

    # Target update
    ax.annotate('', xy=(5.5, 3.0), xytext=(5.5, 5.25),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=1.8, linestyle='dashed',
                                connectionstyle='arc3,rad=0.4'))
    ax.text(3.7, 4.0, 'θ⁻ ← θ\nevery C steps', ha='center', fontsize=8.5,
            color='#5BAD6F', fontweight='bold',
            bbox=dict(fc='#F0FFF0', ec='#5BAD6F', boxstyle='round,pad=0.25'))

    # TD target formula
    ax.text(6.5, 0.7, 'TD target:  y = r  +  γ · max_{a\'} Q_{θ⁻}(s\', a\')',
            ha='center', fontsize=10, color='#333',
            bbox=dict(fc='#FFFDE7', ec='#F0C000', boxstyle='round,pad=0.4', lw=1.2))

    plt.tight_layout()
    path = os.path.join(OUT, 'dqn_networks.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_dqn_architecture()
    plot_replay_buffer()
    plot_target_network()
    plot_dqn_stability()
    plot_dqn_networks()
    print("Ch07 done.")
