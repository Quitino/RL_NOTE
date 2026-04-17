"""
Ch01: RL Introduction Visualizations
Figures:
  - ml_paradigms.png   : Three ML paradigms side-by-side
  - agent_env_loop.gif : Agent-Environment interaction loop (animated)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.animation as animation

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

OUT = os.path.join(os.path.dirname(__file__), '../../asserts/ch01_rl_intro')
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Three ML Paradigms ──────────────────────────────────────────────
def plot_ml_paradigms():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Three Machine Learning Paradigms', fontsize=14, fontweight='bold', y=1.01)

    configs = [
        {
            'title': 'Supervised Learning',
            'color': '#4C8EBF',
            'data_label': 'Labeled Dataset\n(input, label) pairs',
            'signal_label': 'Exact Error\n(ground-truth labels)',
            'flow': ['Input', 'Model', 'Prediction', 'Loss (label)'],
            'note': 'Static, i.i.d. data\nf: X → Y',
        },
        {
            'title': 'Unsupervised Learning',
            'color': '#E8994C',
            'data_label': 'Unlabeled Dataset\n(input only)',
            'signal_label': 'No External Signal\n(self-supervised)',
            'flow': ['Input', 'Model', 'Structure/Repr.', 'Internal Loss'],
            'note': 'Static data\nDiscover hidden structure',
        },
        {
            'title': 'Reinforcement Learning',
            'color': '#5BAD6F',
            'data_label': 'Online Interaction\n(agent-generated)',
            'signal_label': 'Delayed Reward\n(sparse scalar)',
            'flow': ['State s', 'Policy π', 'Action a', 'Reward r (delayed)'],
            'note': 'Sequential decisions\nExploration required',
        },
    ]

    for ax, cfg in zip(axes, configs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(cfg['title'], fontsize=12, fontweight='bold', color=cfg['color'], pad=8)

        # flow boxes
        box_h = 0.13
        ys = [0.80, 0.62, 0.44, 0.26]
        for i, (label, y) in enumerate(zip(cfg['flow'], ys)):
            fc = cfg['color'] if i in (0, 2) else '#F0F0F0'
            ec = cfg['color']
            tc = 'white' if i in (0, 2) else '#333333'
            ax.add_patch(FancyBboxPatch((0.15, y - box_h / 2), 0.70, box_h,
                                        boxstyle='round,pad=0.02', fc=fc, ec=ec, lw=1.5))
            ax.text(0.50, y, label, ha='center', va='center', fontsize=9, color=tc, fontweight='bold')
            if i < len(ys) - 1:
                ax.annotate('', xy=(0.50, ys[i + 1] + box_h / 2),
                            xytext=(0.50, y - box_h / 2),
                            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

        # data & signal labels
        ax.text(0.50, 0.10, f"Data: {cfg['data_label']}", ha='center', va='center',
                fontsize=8, style='italic', color='#555555',
                bbox=dict(fc='#FAFAFA', ec='#CCCCCC', boxstyle='round,pad=0.3', lw=0.8))

    plt.tight_layout()
    path = os.path.join(OUT, 'ml_paradigms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Agent-Environment Loop (animated GIF) ──────────────────────────
def plot_agent_env_loop():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#F8F8F8')
    fig.patch.set_facecolor('#F8F8F8')

    # Static boxes
    agent_box = FancyBboxPatch((0.5, 1.5), 3.0, 2.5,
                               boxstyle='round,pad=0.2', fc='#4C8EBF', ec='#2c5f8a', lw=2)
    env_box   = FancyBboxPatch((6.5, 1.5), 3.0, 2.5,
                               boxstyle='round,pad=0.2', fc='#5BAD6F', ec='#3a7a4a', lw=2)
    ax.add_patch(agent_box)
    ax.add_patch(env_box)
    ax.text(2.0, 2.75, 'AGENT\n(Policy π)', ha='center', va='center',
            fontsize=13, color='white', fontweight='bold')
    ax.text(8.0, 2.75, 'ENVIRONMENT', ha='center', va='center',
            fontsize=13, color='white', fontweight='bold')

    ax.set_title('Agent-Environment Interaction Loop', fontsize=13, fontweight='bold', pad=12)

    FRAMES = 20

    # Use Line2D arrows (animatable) instead of annotate
    line_action, = ax.plot([3.5, 6.5], [3.3, 3.3], color='#E8994C', lw=2.5,
                           marker='>', markersize=8, markevery=[1])
    line_state,  = ax.plot([6.5, 3.5], [2.2, 2.2], color='#BF4C4C', lw=2.5,
                           marker='>', markersize=8, markevery=[1])
    line_reward, = ax.plot([6.5, 3.5], [1.8, 1.8], color='#9B59B6', lw=2.5,
                           marker='>', markersize=8, markevery=[1])

    lbl_action = ax.text(5.0, 3.65, 'Action  a_t', ha='center', va='bottom',
                         fontsize=11, color='#E8994C', fontweight='bold')
    lbl_state  = ax.text(5.0, 2.05, "State  s_{t+1}", ha='center', va='top',
                         fontsize=11, color='#BF4C4C', fontweight='bold')
    lbl_reward = ax.text(5.0, 1.65, "Reward  r_{t+1}", ha='center', va='top',
                         fontsize=11, color='#9B59B6', fontweight='bold')

    step_text = ax.text(5.0, 5.5, 'Step t = 0', ha='center', va='top',
                        fontsize=12, color='#333333', style='italic')

    def update(frame):
        t = frame % FRAMES
        alpha_action = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * t / FRAMES))
        alpha_state  = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * t / FRAMES + np.pi))
        alpha_reward = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * t / FRAMES + np.pi * 1.2))
        for obj, a in [(line_action, alpha_action), (lbl_action, alpha_action),
                       (line_state, alpha_state), (lbl_state, alpha_state),
                       (line_reward, alpha_reward), (lbl_reward, alpha_reward)]:
            obj.set_alpha(a)
        step_text.set_text(f'Step  t = {frame}')
        return line_action, line_state, line_reward, lbl_action, lbl_state, lbl_reward, step_text

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=150, blit=True)

    path = os.path.join(OUT, 'agent_env_loop.gif')
    ani.save(path, writer='pillow', fps=8)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_ml_paradigms()
    plot_agent_env_loop()
    print("Ch01 done.")
