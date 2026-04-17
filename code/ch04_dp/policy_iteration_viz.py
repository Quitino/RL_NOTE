"""
Policy Iteration – Animated Grid World Visualization
=====================================================
Demonstrates the two-phase cycle of Policy Iteration:
  Phase 1 – Policy Evaluation  : iteratively compute V^pi until convergence
  Phase 2 – Policy Improvement : one-step greedy update of policy arrows

Grid world (5 x 5):
  G = Goal  (+1 reward, terminal)
  T = Trap  (-1 reward, terminal)
  W = Wall  (blocked)
  S = Start

Outputs
-------
  asserts/policy_iteration_static.png   – key frames (evaluation + improvement)
  asserts/policy_iteration_anim.gif     – full animated run
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

# ── grid setup ────────────────────────────────────────────────────────────────
NROW, NCOL = 5, 5
GOAL  = (0, 4)   # top-right
TRAP  = (3, 2)   # middle
WALLS = {(1, 1), (1, 3), (3, 1)}
GAMMA = 0.95
THETA = 1e-4     # convergence threshold

# rewards
def get_reward(s):
    if s == GOAL: return  1.0
    if s == TRAP: return -1.0
    return -0.02   # small step cost

def is_terminal(s):
    return s in (GOAL, TRAP)

def is_valid(s):
    r, c = s
    return 0 <= r < NROW and 0 <= c < NCOL and s not in WALLS

# actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACT_SYM = ["↑","↓","←","→"]
ACT_NAME = ["Up","Down","Left","Right"]

def transition(s, a):
    """Deterministic transition (returns next state)."""
    if is_terminal(s):
        return s
    nr, nc = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    ns = (nr, nc)
    if not is_valid(ns):
        return s   # stay
    return ns

# all non-wall states
ALL_STATES = [(r, c) for r in range(NROW) for c in range(NCOL)
              if (r, c) not in WALLS]

# ── DP core ───────────────────────────────────────────────────────────────────
def policy_evaluation_step(V, policy):
    """One full sweep of policy evaluation; return new V and max delta."""
    new_V = V.copy()
    delta = 0.0
    for s in ALL_STATES:
        if is_terminal(s):
            continue
        a = policy[s]
        ns = transition(s, a)
        r  = get_reward(ns)
        v  = r + GAMMA * V[ns]
        delta = max(delta, abs(v - new_V[s]))
        new_V[s] = v
    return new_V, delta


def policy_improvement(V, policy):
    """One-step greedy improvement; return new policy + stable flag."""
    new_policy = dict(policy)
    stable = True
    for s in ALL_STATES:
        if is_terminal(s):
            continue
        old_a = policy[s]
        best_a, best_v = old_a, -1e9
        for a in range(4):
            ns = transition(s, a)
            v  = get_reward(ns) + GAMMA * V[ns]
            if v > best_v:
                best_v, best_a = v, a
        new_policy[s] = best_a
        if best_a != old_a:
            stable = False
    return new_policy, stable


def run_policy_iteration():
    """Run full policy iteration; record every evaluation sweep and improvement."""
    # initialise
    V = {s: 0.0 for s in ALL_STATES}
    V[GOAL] = 1.0; V[TRAP] = -1.0
    policy = {s: 0 for s in ALL_STATES}   # all Up initially

    history = []  # list of (phase, sweep_idx, V_snapshot, policy_snapshot, delta)
    pi_idx = 0

    while True:
        # ---- evaluation phase ----
        sweep = 0
        while True:
            V, delta = policy_evaluation_step(V, policy)
            history.append(("eval", pi_idx, sweep, V.copy(), dict(policy), delta))
            sweep += 1
            if delta < THETA or sweep > 200:
                break

        # ---- improvement phase ----
        policy, stable = policy_improvement(V, policy)
        history.append(("improve", pi_idx, 0, V.copy(), dict(policy), 0.0))
        pi_idx += 1
        if stable:
            break

    return history


# ── drawing helpers ────────────────────────────────────────────────────────────
C_BG    = "#F0F4F8"
C_WALL  = "#2C3E50"
C_GOAL  = "#27AE60"
C_TRAP  = "#E74C3C"
C_START = "#3498DB"
C_TEXT  = "#2C3E50"
CMAP    = "RdYlGn"

def v_to_grid(V):
    grid = np.zeros((NROW, NCOL))
    for (r, c), v in V.items():
        grid[r, c] = v
    return grid


def draw_grid(ax, V, policy, title="", phase_label="", delta=None):
    """Draw value-function heat-map + policy arrows."""
    grid = v_to_grid(V)
    vmin, vmax = -1.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    ax.set_xlim(-0.5, NCOL-0.5)
    ax.set_ylim(-0.5, NROW-0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", color=C_TEXT, pad=4)

    # cells
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
                col = cmap(norm(grid[r, c]))

            rect = plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                  color=col, ec="white", lw=1.5, zorder=1)
            ax.add_patch(rect)

            # value text
            if s not in WALLS:
                txt = f"{grid[r,c]:.2f}"
                txt_col = "white" if abs(grid[r,c]) > 0.5 else C_TEXT
                ax.text(c, r+0.28, txt, ha="center", va="center",
                        fontsize=7, color=txt_col, zorder=3)

            # cell label
            if s == GOAL:
                ax.text(c, r-0.22, "GOAL", ha="center", fontsize=6.5,
                        color="white", fontweight="bold", zorder=3)
            elif s == TRAP:
                ax.text(c, r-0.22, "TRAP", ha="center", fontsize=6.5,
                        color="white", fontweight="bold", zorder=3)
            elif s in WALLS:
                ax.text(c, r, "WALL", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=3)

    # policy arrows
    arr_dx = [0, 0, -0.3, 0.3]
    arr_dy = [-0.3, 0.3, 0, 0]
    for s, a in policy.items():
        r, c = s
        if is_terminal(s) or s in WALLS:
            continue
        ax.annotate("", xy=(c + arr_dx[a], r + arr_dy[a]),
                    xytext=(c, r),
                    arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.12",
                                    color="black", lw=1.2),
                    zorder=4)

    # phase label
    if phase_label:
        ax.text(NCOL/2 - 0.5, NROW - 0.05, phase_label,
                ha="center", va="top", fontsize=7.5,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="#2C3E50", ec="none"),
                zorder=5)
    if delta is not None:
        ax.text(NCOL - 0.55, 0.3, f"Δ={delta:.4f}",
                ha="right", fontsize=7, color="#888", zorder=5)

    # colour bar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("V(s)", fontsize=7)


# ── static key-frames ─────────────────────────────────────────────────────────
def save_static(history):
    # pick: first eval sweep, last eval sweep before first improvement, after improvement, final
    eval_frames   = [h for h in history if h[0] == "eval"    and h[1] == 0]
    improv_frames = [h for h in history if h[0] == "improve"]

    candidates = [
        (eval_frames[0],                        "PI_0  Eval sweep 0  (init)"),
        (eval_frames[min(5, len(eval_frames)-1)],"PI_0  Eval sweep 5"),
        (eval_frames[-1],                        "PI_0  Eval converged"),
        (improv_frames[0],                       "PI_0  Policy Improved"),
    ]
    if len(improv_frames) > 1:
        last_eval = [h for h in history if h[0] == "eval" and h[1] == len(improv_frames)-1]
        if last_eval:
            candidates.append((last_eval[-1], "Final Policy (converged)"))

    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5.5), facecolor=C_BG)
    if n == 1:
        axes = [axes]
    fig.suptitle("Policy Iteration  –  Key Frames", fontsize=13,
                 fontweight="bold", color=C_TEXT, y=1.01)

    for ax, (frame, label) in zip(axes, candidates):
        phase, pi_i, sw_i, V, pol, delta = frame
        phase_str = (f"EVAL  iter={pi_i}  sweep={sw_i}  Δ={delta:.4f}"
                     if phase == "eval" else f"IMPROVE  iter={pi_i}")
        draw_grid(ax, V, pol, title=label, phase_label=phase_str, delta=delta)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "policy_iteration_static.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── animation ─────────────────────────────────────────────────────────────────
def save_animation(history):
    # subsample to keep GIF manageable
    def pick_frames(hist, max_frames=60):
        eval_groups = {}
        for h in hist:
            if h[0] == "eval":
                key = h[1]
                eval_groups.setdefault(key, []).append(h)
        sampled = []
        for key in sorted(eval_groups):
            grp = eval_groups[key]
            # take first 3, every-other middle, last 2
            picks = (grp[:3]
                     + grp[3:-2:max(1, len(grp)//8)]
                     + grp[-2:])
            sampled.extend(picks)
        for h in hist:
            if h[0] == "improve":
                sampled.append(h)
        # sort by original order
        order = {id(h): i for i, h in enumerate(hist)}
        sampled.sort(key=lambda h: order.get(id(h), 9999))
        # dedupe preserving order
        seen = set()
        out = []
        for h in sampled:
            hid = id(h)
            if hid not in seen:
                seen.add(hid)
                out.append(h)
        return out[:max_frames]

    frames = pick_frames(history)

    # convergence data
    eval_deltas = [(h[1], h[2], h[5]) for h in history if h[0] == "eval"]

    fig = plt.figure(figsize=(13, 5.5), facecolor=C_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[3, 3, 2], wspace=0.35)
    ax_grid  = fig.add_subplot(gs[0])   # current V + policy
    ax_opt   = fig.add_subplot(gs[1])   # "optimal so far" (last improve)
    ax_conv  = fig.add_subplot(gs[2])   # convergence curve

    fig.suptitle("Policy Iteration  –  Animated\n"
                 "Left: current state  |  Middle: latest policy  |  Right: delta curve",
                 fontsize=10, color=C_TEXT, fontweight="bold")

    # track last improvement frame
    last_improve = {
        "V": {s: 0.0 for s in ALL_STATES},
        "policy": {s: 0 for s in ALL_STATES},
        "pi_idx": 0,
    }

    def update(frame_idx):
        phase, pi_i, sw_i, V, pol, delta = frames[frame_idx]
        nonlocal last_improve

        if phase == "improve":
            last_improve = {"V": V, "policy": pol, "pi_idx": pi_i}

        for ax in [ax_grid, ax_opt]:
            ax.cla()

        # left: current frame
        phase_str = (f"EVAL  PI-iter={pi_i}  sweep={sw_i}"
                     if phase == "eval" else f"IMPROVE  PI-iter={pi_i}")
        draw_grid(ax_grid, V, pol,
                  title=f"Current  ({phase.upper()}  pi_iter={pi_i})",
                  phase_label=phase_str, delta=delta)

        # middle: latest stable policy
        draw_grid(ax_opt, last_improve["V"], last_improve["policy"],
                  title=f"Latest Policy  (PI iter {last_improve['pi_idx']})",
                  phase_label=f"STABLE after improve {last_improve['pi_idx']}")

        # right: delta curve
        ax_conv.cla()
        ax_conv.set_facecolor("#FAFAFA")
        ax_conv.set_title("Evaluation Delta", fontsize=8.5,
                          fontweight="bold", color=C_TEXT)
        ax_conv.set_xlabel("Cumulative sweep", fontsize=7)
        ax_conv.set_ylabel("max |ΔV|", fontsize=7)
        ax_conv.axhline(THETA, color="gray", lw=0.8, ls="--",
                        label=f"θ={THETA}")
        ax_conv.set_yscale("log")
        ax_conv.tick_params(labelsize=6)

        cum = 0
        for g_pi_i, g_sw_i, g_delta in eval_deltas:
            cum += 1
            col = plt.get_cmap("tab10")(g_pi_i % 10)
            ax_conv.scatter(cum, max(g_delta, 1e-6), color=col,
                            s=8, zorder=3)
            if g_pi_i <= last_improve["pi_idx"]:
                ax_conv.scatter(cum, max(g_delta, 1e-6),
                                color=col, s=20, zorder=4,
                                marker="o", edgecolors="black", lw=0.5)

        # PI iteration boundaries (vertical lines)
        boundary = 0
        prev_pi = -1
        for g_pi_i, g_sw_i, g_delta in eval_deltas:
            boundary += 1
            if g_pi_i != prev_pi and prev_pi >= 0:
                ax_conv.axvline(boundary, color="#AAAAAA", lw=0.7, ls=":")
            prev_pi = g_pi_i

        ax_conv.legend(fontsize=6)
        ax_conv.grid(True, alpha=0.3)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=350, blit=False, repeat=True)
    out = os.path.join(OUT_DIR, "policy_iteration_anim.gif")
    anim.save(out, writer="pillow", fps=2.5, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[policy_iteration_viz]  Running PI...")
    history = run_policy_iteration()
    n_eval_frames = sum(1 for h in history if h[0] == "eval")
    n_improve     = sum(1 for h in history if h[0] == "improve")
    print(f"  PI complete: {n_improve} improvement steps, "
          f"{n_eval_frames} eval sweeps total")
    save_static(history)
    save_animation(history)
    print("[policy_iteration_viz]  Done.")
