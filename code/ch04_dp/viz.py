"""
Ch04: Dynamic Programming Visualizations  (consolidated wrapper)
Runs all three original DP visualization scripts with output
redirected to asserts/ch04_dp/.

Figures generated (in asserts/ch04_dp/):
  bellman_backup_static.png / bellman_backup_anim.gif
  policy_iteration_static.png / policy_iteration_anim.gif
  value_iteration_static.png / value_iteration_anim.gif
  value_iteration_qvals.png / vi_pi_comparison.png
  bellman_recursion_loop.png
"""
import os
import types
import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, '..', '..')
CODE_ROOT = os.path.join(REPO_ROOT, 'code')
OUT_DIR = os.path.join(REPO_ROOT, 'docs', 'asserts', 'ch04_dp')
os.makedirs(OUT_DIR, exist_ok=True)


def _run_script(script_path: str):
    """Load a script as a module, overriding its OUT_DIR before execution."""
    spec = importlib.util.spec_from_file_location('_tmp_mod', script_path)
    mod = types.ModuleType(spec.name)
    mod.__file__ = script_path
    with open(script_path) as f:
        source = f.read()
    source = source.replace(
        'OUT_DIR = os.path.join(SCRIPT_DIR, "..", "asserts")',
        f'OUT_DIR = r"{OUT_DIR}"',
    )
    exec(compile(source, script_path, 'exec'), mod.__dict__)


def plot_bellman_recursion_loop():
    """V-Q mutual recursion cycle + EKF analogy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('#F8F9FA')

    # ── helpers ─────────────────────────────────────────────────────────────
    def circle(ax, cx, cy, r, fc, ec='#333', lw=2.0, zorder=3):
        ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw,
                                alpha=0.93, zorder=zorder))

    def box(ax, cx, cy, w, h, fc, ec='#333', lw=1.8, zorder=3):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.15', fc=fc, ec=ec,
                                    lw=lw, alpha=0.93, zorder=zorder))

    def arr(ax, x0, y0, x1, y1, color, lw=2.2, rad=0.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                   connectionstyle=f'arc3,rad={rad}'), zorder=2)

    def label(ax, x, y, txt, color, fs=9, bold=False, bg=None, bgec=None):
        kw = dict(ha='center', va='center', fontsize=fs, color=color,
                  fontweight='bold' if bold else 'normal', zorder=5)
        if bg:
            kw['bbox'] = dict(fc=bg, ec=bgec or bg, boxstyle='round,pad=0.3',
                              lw=1.2, alpha=0.92)
        ax.text(x, y, txt, **kw)

    # ═══════════════════════════════════════════════════════════════════════
    # LEFT PANEL — Bellman V-Q closed loop
    # ═══════════════════════════════════════════════════════════════════════
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 9); ax1.axis('off')
    ax1.set_title('Bellman 方程的递推闭环\nV 与 Q 相互定义，两侧含同一未知量',
                  fontsize=11, fontweight='bold', pad=8)

    # Nodes
    circle(ax1, 2.2, 6.5, 0.82, '#2E86C1')          # V(s)
    label(ax1, 2.2, 6.65, 'V(s)', 'white', fs=12, bold=True)
    label(ax1, 2.2, 6.2, '当前状态价值', '#D6EAF8', fs=8)

    box(ax1, 7.8, 6.5, 2.3, 1.05, '#E67E22')         # Q(s,a)
    label(ax1, 7.8, 6.65, 'Q(s, a)', 'white', fs=12, bold=True)
    label(ax1, 7.8, 6.2, '动作价值', '#FDEBD0', fs=8)

    circle(ax1, 7.8, 3.5, 0.82, '#1E8449')           # V(s')
    label(ax1, 7.8, 3.65, "V(s')", 'white', fs=12, bold=True)
    label(ax1, 7.8, 3.2, '下一状态价值', '#D5F5E3', fs=8)

    # Arrow V(s) → Q(s,a)
    arr(ax1, 3.02, 6.5, 6.69, 6.5, '#2E86C1', rad=-0.15)
    label(ax1, 5.0, 7.3, 'Sigma_a  pi(a|s) * Q(s,a)', '#2E86C1', fs=9, bold=True,
          bg='#EBF5FB', bgec='#2E86C1')
    label(ax1, 5.0, 6.85, '按策略对所有动作加权求和', '#2E86C1', fs=8)

    # Arrow Q(s,a) → V(s')
    arr(ax1, 7.8, 5.97, 7.8, 4.32, '#E67E22', rad=0.0)
    label(ax1, 9.45, 5.15, "Sigma_{s'} P(s'|s,a)", '#E67E22', fs=9, bold=True,
          bg='#FEF9E7', bgec='#E67E22')
    label(ax1, 9.45, 4.72, '* [R + gamma*V(s\')]', '#E67E22', fs=9, bold=True,
          bg='#FEF9E7', bgec='#E67E22')
    label(ax1, 9.45, 4.3, '环境转移 + 折扣未来', '#E67E22', fs=8)

    # Arrow V(s') → back to V(s)  — the "loop-closing" arrow
    arr(ax1, 6.99, 3.5, 3.02, 6.17, '#1E8449', rad=0.3)
    label(ax1, 4.1, 4.1, "V(s') 与 V(s) 是同一函数 V^pi", '#1E8449', fs=9, bold=True,
          bg='#EAFAF1', bgec='#1E8449')
    label(ax1, 4.1, 3.65, '代入后方程两侧均含 V^pi → 自洽闭环', '#1E8449', fs=8)

    # Fixed-point label in center
    label(ax1, 5.0, 5.15, '固定点迭代', '#7D3C98', fs=10, bold=True)
    label(ax1, 5.0, 4.72, 'V_{k+1}(s) <- T[V_k]  直到  V_{k+1} = V_k', '#7D3C98', fs=8.5)

    # Formula box
    ax1.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 1.6,
                  boxstyle='round,pad=0.15', fc='#1A252F', ec='#2E86C1', lw=2.0, alpha=0.95))
    label(ax1, 5.0, 1.22,
          "V^pi(s) = Sigma_a pi(a|s) Sigma_{s'} P(s'|s,a) [R + gamma * V^pi(s')]",
          '#AED6F1', fs=10, bold=True)
    label(ax1, 5.0, 0.68,
          '右侧出现 V^pi(s\') — 与左侧 V^pi(s) 是同一函数 → 方程两侧含未知量 = "闭环"',
          '#F0FFF0', fs=8.5)

    # ═══════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — EKF comparison
    # ═══════════════════════════════════════════════════════════════════════
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 9); ax2.axis('off')
    ax2.set_title('类比：EKF 预测-更新闭环\n（相似结构，不同物理意义）',
                  fontsize=11, fontweight='bold', pad=8)

    # EKF loop diagram (top half)
    # x_{k-1}, predict, x^-_k, update, x_k, loop back
    ekf_y = 6.8
    nodes = [
        (1.3, ekf_y, 'x_hat_{k-1}', '上一时刻\n估计', '#2E86C1'),
        (4.5, ekf_y, 'x_hat^-_k',  '预测状态\n（先验）', '#8E44AD'),
        (7.8, ekf_y, 'x_hat_k',    '更新状态\n（后验）', '#1E8449'),
    ]
    for cx, cy, title, sub, fc in nodes:
        circle(ax2, cx, cy, 0.72, fc)
        label(ax2, cx, cy + 0.18, title, 'white', fs=9, bold=True)
        label(ax2, cx, cy - 0.25, sub, 'white', fs=7)

    # Arrows between EKF nodes
    arr(ax2, 2.02, ekf_y, 3.78, ekf_y, '#8E44AD', lw=2.2)
    label(ax2, 3.16, ekf_y + 0.75, '预测步', '#8E44AD', fs=9, bold=True,
          bg='#F5EEF8', bgec='#8E44AD')
    label(ax2, 3.16, ekf_y + 0.32, "x^-_k = f(x_{k-1}, u_k)", '#8E44AD', fs=8)

    arr(ax2, 5.22, ekf_y, 7.08, ekf_y, '#1E8449', lw=2.2)
    label(ax2, 6.5, ekf_y + 0.75, '更新步', '#1E8449', fs=9, bold=True,
          bg='#EAFAF1', bgec='#1E8449')
    label(ax2, 6.5, ekf_y + 0.32, "K*(z_k - h(x^-_k))", '#1E8449', fs=8)

    # Loop-back arrow x_k → x_{k-1} (next step)
    ax2.annotate('', xy=(1.3, ekf_y - 0.72), xytext=(7.8, ekf_y - 0.72),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.0,
                                 connectionstyle='arc3,rad=0.0'), zorder=2)
    label(ax2, 4.55, ekf_y - 1.05, 'k -> k+1  时序循环（有新测量 z_k 驱动）',
          '#E74C3C', fs=8.5, bold=True, bg='#FDEDEC', bgec='#E74C3C')

    # Comparison table (bottom half)
    table_top = 5.0
    col_xs = [0.1, 3.3, 6.7]
    col_ws = [3.1, 3.3, 3.1]
    rows = [
        ('对比维度',          'Bellman 递推闭环',           'EKF 预测-更新闭环'),
        ('循环变量',          '状态 s -> s\' -> s\'\'...',  '时刻 k -> k+1 -> k+2...'),
        ('更新对象',          '价值函数 V(s) 或 Q(s,a)',    '状态估计 x_hat_k, 协方差 P_k'),
        ('驱动信号',          '奖励 R + 折扣 gamma',        '新测量 z_k'),
        ('收敛目标',          '不动点 V^pi（或 V*）',       '后验均值收敛到真值'),
        ('相同本质',          '迭代求解含未知量的自洽方程 = 不动点迭代',
                             '迭代求解含未知量的自洽方程 = 不动点迭代'),
    ]
    row_h = 0.72
    header_fc = ['#1A252F', '#154360', '#145A32']
    body_fc   = ['#FDFEFE', '#EBF5FB', '#EAFAF1']
    same_fc   = ['#4A235A', '#4A235A', '#4A235A']

    for i, row in enumerate(rows):
        y = table_top - i * row_h
        is_header = (i == 0)
        is_same = (i == len(rows) - 1)
        for j, (text, cx, cw) in enumerate(zip(row, col_xs, col_ws)):
            if is_header:
                fc = header_fc[j]; tc = 'white'; fw = 'bold'
            elif is_same:
                fc = same_fc[j]; tc = '#F8F9FA'; fw = 'bold'
            else:
                fc = body_fc[j]; tc = '#1A252F'; fw = 'normal'
            ax2.add_patch(FancyBboxPatch((cx, y - row_h * 0.46), cw, row_h * 0.88,
                          boxstyle='square,pad=0.0', fc=fc, ec='#BDC3C7', lw=0.8,
                          alpha=0.92, zorder=3))
            ax2.text(cx + cw / 2, y + 0.02, text, ha='center', va='center',
                     fontsize=8 if not is_header else 8.5, color=tc,
                     fontweight=fw, zorder=4, wrap=True)

    # Bottom note
    label(ax2, 5.0, 0.55,
          '结论：两者精神相似（方程两侧均含未知量，迭代到不动点），但 EKF 是时序驱动，Bellman 是空间自洽。',
          '#2C3E50', fs=8.5)

    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, 'bellman_recursion_loop.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    plot_bellman_recursion_loop()
    scripts = [
        os.path.join(CODE_ROOT, 'bellman_backup_viz.py'),
        os.path.join(CODE_ROOT, 'policy_iteration_viz.py'),
        os.path.join(CODE_ROOT, 'value_iteration_viz.py'),
    ]
    for s in scripts:
        print(f"\n--- Running {os.path.basename(s)} ---")
        _run_script(s)
    print("\nCh04 done. Assets in docs/asserts/ch04_dp/")
