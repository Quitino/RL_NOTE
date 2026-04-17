"""
Value Iteration – Animated Grid World Visualization
====================================================
Shows Bellman Optimality operator applied iteratively:
  V_{k+1}(s) = max_a  sum_{s'} P(s'|s,a) [ R + gamma * V_k(s') ]

Uses the same 5x5 grid world as policy_iteration_viz.py.

Outputs
-------
  asserts/value_iteration_static.png   – key frames (every ~5 sweeps)
  asserts/value_iteration_anim.gif     – animated convergence
  asserts/value_iteration_qvals.png    – Q(s,a) heatmap at convergence
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "asserts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── same grid as policy_iteration_viz ─────────────────────────────────────────
NROW, NCOL = 5, 5
GOAL  = (0, 4)
TRAP  = (3, 2)
WALLS = {(1, 1), (1, 3), (3, 1)}
GAMMA = 0.95
THETA = 1e-4
MAX_ITER = 300

ACTIONS  = [(-1,0),(1,0),(0,-1),(0,1)]
ACT_SYM  = ["↑","↓","←","→"]
ACT_NAME = ["Up","Down","Left","Right"]

C_BG   = "#F0F4F8"
C_WALL = "#2C3E50"
C_GOAL = "#27AE60"
C_TRAP = "#E74C3C"
C_TEXT = "#2C3E50"
CMAP   = "RdYlGn"

def get_reward(s):
    if s == GOAL: return  1.0
    if s == TRAP: return -1.0
    return -0.02

def is_terminal(s):
    return s in (GOAL, TRAP)

def is_valid(s):
    r, c = s
    return 0 <= r < NROW and 0 <= c < NCOL and s not in WALLS

def transition(s, a):
    if is_terminal(s): return s
    nr, nc = s[0]+ACTIONS[a][0], s[1]+ACTIONS[a][1]
    ns = (nr, nc)
    return ns if is_valid(ns) else s

ALL_STATES = [(r,c) for r in range(NROW) for c in range(NCOL)
              if (r,c) not in WALLS]

# ── value iteration core ───────────────────────────────────────────────────────
def run_value_iteration():
    """Return list of (iteration, V_snapshot, delta)."""
    V = {s: 0.0 for s in ALL_STATES}
    V[GOAL] = 1.0; V[TRAP] = -1.0
    history = [(0, V.copy(), float("inf"))]

    for k in range(1, MAX_ITER+1):
        new_V = V.copy()
        delta = 0.0
        for s in ALL_STATES:
            if is_terminal(s):
                continue
            q_vals = []
            for a in range(4):
                ns  = transition(s, a)
                q   = get_reward(ns) + GAMMA * V[ns]
                q_vals.append(q)
            best = max(q_vals)
            delta = max(delta, abs(best - V[s]))
            new_V[s] = best
        V = new_V
        history.append((k, V.copy(), delta))
        if delta < THETA:
            break

    return history


def extract_policy(V):
    policy = {}
    for s in ALL_STATES:
        if is_terminal(s):
            policy[s] = 0
            continue
        best_a, best_v = 0, -1e9
        for a in range(4):
            ns = transition(s, a)
            v  = get_reward(ns) + GAMMA * V[ns]
            if v > best_v:
                best_v, best_a = v, a
        policy[s] = best_a
    return policy


def compute_q_table(V):
    """Return Q[s][a] for all non-wall states."""
    Q = {}
    for s in ALL_STATES:
        Q[s] = []
        for a in range(4):
            ns = transition(s, a)
            Q[s].append(get_reward(ns) + GAMMA * V[ns])
    return Q


# ── drawing ────────────────────────────────────────────────────────────────────
def v_to_grid(V):
    g = np.full((NROW, NCOL), np.nan)
    for (r,c), v in V.items():
        g[r,c] = v
    return g


def draw_grid(ax, V, policy=None, title="", phase_label="", delta=None, iteration=None):
    grid = v_to_grid(V)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(CMAP)

    ax.set_xlim(-0.5, NCOL-0.5)
    ax.set_ylim(-0.5, NROW-0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", color=C_TEXT, pad=4)

    for r in range(NROW):
        for c in range(NCOL):
            s = (r, c)
            if s in WALLS:
                col = C_WALL
            elif s == GOAL:
                col = C_GOAL
            elif s == TRAP:
                col = C_TRAP
            else:
                v   = grid[r, c]
                col = cmap(norm(v)) if not np.isnan(v) else "#CCCCCC"

            rect = plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                  color=col, ec="white", lw=1.5, zorder=1)
            ax.add_patch(rect)

            if s not in WALLS:
                v   = grid[r, c]
                txt = f"{v:.3f}" if not np.isnan(v) else "?"
                tcol = "white" if abs(v) > 0.55 else C_TEXT
                ax.text(c, r+0.3, txt, ha="center", va="center",
                        fontsize=6.5, color=tcol, zorder=3)

            if s == GOAL:
                ax.text(c, r-0.22, "GOAL", ha="center", fontsize=6,
                        color="white", fontweight="bold", zorder=3)
            elif s == TRAP:
                ax.text(c, r-0.22, "TRAP", ha="center", fontsize=6,
                        color="white", fontweight="bold", zorder=3)
            elif s in WALLS:
                ax.text(c, r, "WALL", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold", zorder=3)

    if policy:
        arr_dx = [0, 0, -0.3, 0.3]
        arr_dy = [-0.3, 0.3, 0, 0]
        for s, a in policy.items():
            r, c = s
            if is_terminal(s) or s in WALLS:
                continue
            ax.annotate("", xy=(c+arr_dx[a], r+arr_dy[a]),
                        xytext=(c, r),
                        arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.12",
                                        color="black", lw=1.2),
                        zorder=4)

    if phase_label:
        ax.text(NCOL/2-0.5, NROW-0.05, phase_label,
                ha="center", va="top", fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=C_TEXT, ec="none"),
                zorder=5)
    if iteration is not None:
        ax.text(-0.4, -0.4, f"k={iteration}", fontsize=8,
                color=C_TEXT, fontweight="bold", zorder=5)
    if delta is not None:
        ax.text(NCOL-0.55, -0.35, f"Δ={delta:.5f}", ha="right",
                fontsize=6.5, color="#555", zorder=5)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label("V*(s)", fontsize=6.5)


# ── static key-frames ─────────────────────────────────────────────────────────
def save_static(history):
    total = len(history)
    picks = [0, min(3, total-1), min(8, total-1),
             min(20, total-1), total-1]
    picks = sorted(set(picks))

    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 5.5), facecolor=C_BG)
    if n == 1: axes = [axes]
    fig.suptitle("Value Iteration  –  V*(s) Convergence  (key frames)",
                 fontsize=12, fontweight="bold", color=C_TEXT)

    for ax, idx in zip(axes, picks):
        k, V, delta = history[idx]
        pol = extract_policy(V) if idx == total-1 else None
        lbl = f"k={k}  Δ={delta:.5f}"
        draw_grid(ax, V, policy=pol,
                  title=f"Iteration k={k}", phase_label=lbl,
                  delta=delta, iteration=k)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "value_iteration_static.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Q-value heatmap at convergence ────────────────────────────────────────────
def save_qvals(history):
    _, V_opt, _ = history[-1]
    Q = compute_q_table(V_opt)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor=C_BG)
    fig.suptitle("Q*(s, a) at Convergence  –  One panel per action",
                 fontsize=12, fontweight="bold", color=C_TEXT)

    all_q = [q for qs in Q.values() for q in qs]
    vmin, vmax = min(all_q), max(all_q)
    norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap2 = plt.get_cmap("coolwarm")

    for ai, ax in enumerate(axes.flat):
        grid_q = np.full((NROW, NCOL), np.nan)
        for s, qs in Q.items():
            r, c = s
            grid_q[r, c] = qs[ai]

        ax.set_xlim(-0.5, NCOL-0.5)
        ax.set_ylim(-0.5, NROW-0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"Q*(s, a={ACT_NAME[ai]}  {ACT_SYM[ai]})",
                     fontsize=10, fontweight="bold", color=C_TEXT)

        for r in range(NROW):
            for c in range(NCOL):
                s = (r, c)
                q = grid_q[r, c]
                if s in WALLS:
                    col = C_WALL
                elif s == GOAL:
                    col = C_GOAL
                elif s == TRAP:
                    col = C_TRAP
                elif np.isnan(q):
                    col = "#CCCCCC"
                else:
                    col = cmap2(norm(q))
                rect = plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                     color=col, ec="white", lw=1.5, zorder=1)
                ax.add_patch(rect)
                if not np.isnan(q) and s not in WALLS:
                    tcol = "white" if abs(q-0.5*(vmin+vmax)) > 0.3*(vmax-vmin) else C_TEXT
                    ax.text(c, r, f"{q:.3f}", ha="center", va="center",
                            fontsize=6.5, color=tcol, zorder=3)
                if s == GOAL:
                    ax.text(c, r+0.32, "GOAL", ha="center", fontsize=5.5,
                            color="white", zorder=3)
                elif s == TRAP:
                    ax.text(c, r+0.32, "TRAP", ha="center", fontsize=5.5,
                            color="white", zorder=3)
                elif s in WALLS:
                    ax.text(c, r, "WALL", ha="center", va="center",
                            fontsize=5.5, color="white", zorder=3)

        # mark best action cells
        for s in ALL_STATES:
            if is_terminal(s): continue
            r, c = s
            best_a = int(np.argmax(Q[s]))
            if best_a == ai:
                ax.add_patch(plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                            fill=False, ec="#FFD700", lw=2.5, zorder=4))

        sm = ScalarMappable(cmap=cmap2, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=5.5)
        cbar.set_label("Q*(s,a)", fontsize=6.5)

    fig.text(0.5, 0.01,
             "Gold border = state where this action is optimal  (argmax_a Q*(s,a))",
             ha="center", fontsize=8, color="#555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(OUT_DIR, "value_iteration_qvals.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── animation ─────────────────────────────────────────────────────────────────
def save_animation(history):
    # subsample: first 10 dense, then sparse
    total = len(history)
    dense = list(range(min(10, total)))
    sparse = list(range(10, total, max(1, total//25)))
    idxs  = sorted(set(dense + sparse + [total-1]))

    fig = plt.figure(figsize=(13, 5.5), facecolor=C_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[3, 3, 2.2], wspace=0.35)
    ax_now  = fig.add_subplot(gs[0])   # current V_k
    ax_diff = fig.add_subplot(gs[1])   # |V_k - V_{k-1}|
    ax_conv = fig.add_subplot(gs[2])   # convergence curve

    fig.suptitle(
        "Value Iteration  –  Animated  |  "
        r"$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[r + \gamma V_k(s')]$",
        fontsize=10, color=C_TEXT, fontweight="bold")

    # pre-compute diff grids
    diff_grids = {}
    for i, idx in enumerate(idxs):
        k, V, delta = history[idx]
        if i == 0:
            diff_grids[idx] = {s: 0.0 for s in ALL_STATES}
        else:
            prev_idx = idxs[i-1]
            _, V_prev, _ = history[prev_idx]
            diff_grids[idx] = {s: abs(V[s] - V_prev[s]) for s in ALL_STATES}

    def update(fi):
        idx = idxs[fi]
        k, V, delta = history[idx]
        pol = extract_policy(V)

        ax_now.cla(); ax_diff.cla(); ax_conv.cla()

        # left: V_k with greedy policy arrows
        draw_grid(ax_now, V, policy=pol,
                  title=f"$V_k(s)$  iteration k={k}",
                  phase_label=f"VALUE ITERATION  k={k}  Δ={delta:.5f}",
                  delta=delta, iteration=k)

        # middle: change map |V_k - V_prev|
        diff = diff_grids[idx]
        diff_norm = Normalize(vmin=0, vmax=max(max(diff.values()), 1e-6))
        diff_cmap = plt.get_cmap("hot_r")
        ax_diff.set_xlim(-0.5, NCOL-0.5)
        ax_diff.set_ylim(-0.5, NROW-0.5)
        ax_diff.set_aspect("equal")
        ax_diff.invert_yaxis()
        ax_diff.axis("off")
        ax_diff.set_title(f"|ΔV|  (change from prev frame)  k={k}",
                          fontsize=9, fontweight="bold", color=C_TEXT)

        for r in range(NROW):
            for c in range(NCOL):
                s = (r, c)
                if s in WALLS:
                    col = C_WALL
                elif s == GOAL:
                    col = C_GOAL
                elif s == TRAP:
                    col = C_TRAP
                else:
                    dv  = diff.get(s, 0.0)
                    col = diff_cmap(diff_norm(dv))
                rect = plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                     color=col, ec="white", lw=1.5, zorder=1)
                ax_diff.add_patch(rect)
                if s not in WALLS:
                    dv = diff.get(s, 0.0)
                    ax_diff.text(c, r, f"{dv:.3f}", ha="center", va="center",
                                 fontsize=6, color=C_TEXT, zorder=3)
                if s == GOAL:
                    ax_diff.text(c, r+0.32, "GOAL", ha="center", fontsize=5.5,
                                 color="white", zorder=3)
                elif s == TRAP:
                    ax_diff.text(c, r+0.32, "TRAP", ha="center", fontsize=5.5,
                                 color="white", zorder=3)
                elif s in WALLS:
                    ax_diff.text(c, r, "WALL", ha="center", va="center",
                                 fontsize=5.5, color="white", zorder=3)

        sm2 = ScalarMappable(cmap=diff_cmap, norm=diff_norm)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax_diff, fraction=0.035, pad=0.02)
        cbar2.ax.tick_params(labelsize=5.5)
        cbar2.set_label("|ΔV|", fontsize=6.5)

        # right: convergence curve (log delta)
        ax_conv.set_facecolor("#FAFAFA")
        ax_conv.set_title("Convergence  log(Δ)", fontsize=9,
                          fontweight="bold", color=C_TEXT)
        ax_conv.set_xlabel("Iteration k", fontsize=7)
        ax_conv.set_ylabel("max |ΔV|  (log scale)", fontsize=7)
        ax_conv.set_yscale("log")
        ax_conv.tick_params(labelsize=6)
        ax_conv.grid(True, alpha=0.3)
        ax_conv.axhline(THETA, color="gray", lw=0.8, ls="--",
                        label=f"θ={THETA}")

        ks     = [h[0] for h in history]
        deltas = [max(h[2], 1e-7) for h in history]
        ax_conv.plot(ks, deltas, color="#3498DB", lw=1.5, zorder=2)
        ax_conv.scatter([k], [max(delta, 1e-7)],
                        color=C_TRAP, s=60, zorder=5, label=f"now k={k}")
        ax_conv.legend(fontsize=6.5)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(idxs),
                                   interval=250, blit=False, repeat=True)
    out = os.path.join(OUT_DIR, "value_iteration_anim.gif")
    anim.save(out, writer="pillow", fps=3, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── comparison chart: VI vs PI convergence ────────────────────────────────────
def save_comparison(vi_history):
    """Plot VI convergence curve + annotate key moments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=C_BG)
    fig.suptitle("Value Iteration vs Policy Iteration  –  Conceptual Comparison",
                 fontsize=11, fontweight="bold", color=C_TEXT)

    # left: VI convergence
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ks     = [h[0] for h in vi_history]
    deltas = [max(h[2], 1e-7) for h in vi_history]
    ax.semilogy(ks, deltas, "-o", color="#3498DB", ms=3, lw=1.5, label="VI delta")
    ax.axhline(THETA, color="gray", lw=0.8, ls="--", label=f"θ={THETA}")
    ax.set_xlabel("Iteration k", fontsize=9)
    ax.set_ylabel("max |ΔV|  (log scale)", fontsize=9)
    ax.set_title("Value Iteration – Convergence", fontsize=10,
                 fontweight="bold", color=C_TEXT)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # annotate convergence point
    conv_k = vi_history[-1][0]
    ax.annotate(f"Converged\n@ k={conv_k}",
                xy=(conv_k, THETA*2), xytext=(conv_k*0.6, THETA*50),
                arrowprops=dict(arrowstyle="->", color=C_TRAP, lw=1.2),
                fontsize=8, color=C_TRAP)

    # right: conceptual comparison table
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor("#FAFAFA")
    ax2.set_title("Algorithm Comparison", fontsize=10,
                  fontweight="bold", color=C_TEXT)
    table_data = [
        ["Property",          "Policy Iteration",    "Value Iteration"],
        ["Inner loop",        "Full eval (many swps)","No inner loop"],
        ["Update rule",       "Bellman Expectation",  "Bellman Optimality"],
        ["Per-iter cost",     "Higher (many sweeps)", "Lower (1 sweep)"],
        ["Iters to converge", "Fewer",                "More"],
        ["Intermediate π",    "Always valid policy",  "Only at end"],
        ["Complexity / iter", "O(|S|²|A|)",           "O(|S|²|A|)"],
        ["Convergence",       "Finite (exact)",       "Asymptotic"],
    ]
    col_widths = [0.28, 0.36, 0.36]
    row_h = 0.11
    y0 = 0.92
    row_colors = ["#2C3E50"] + ["#EAF2FF", "#F5F5F5"] * 10

    for ri, row in enumerate(table_data):
        y = y0 - ri * row_h
        x = 0.0
        for ci, cell in enumerate(row):
            bg = row_colors[ri] if ri > 0 else "#2C3E50"
            fc = "white" if ri == 0 else C_TEXT
            ax2.add_patch(plt.Rectangle([x, y-row_h*0.9], col_widths[ci], row_h*0.85,
                                        color=bg, ec="white", lw=0.5,
                                        transform=ax2.transAxes, zorder=1))
            ax2.text(x + col_widths[ci]/2, y - row_h*0.45, cell,
                     ha="center", va="center", fontsize=7.5,
                     color=fc, fontweight="bold" if ri == 0 else "normal",
                     transform=ax2.transAxes, zorder=2)
            x += col_widths[ci]

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "vi_pi_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[value_iteration_viz]  Running VI...")
    history = run_value_iteration()
    print(f"  VI converged in {history[-1][0]} iterations  "
          f"(final Δ={history[-1][2]:.6f})")
    save_static(history)
    save_qvals(history)
    save_animation(history)
    save_comparison(history)
    print("[value_iteration_viz]  Done.")
