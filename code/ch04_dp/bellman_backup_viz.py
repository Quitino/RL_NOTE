"""
Bellman Backup Diagrams – Animated Visualization
=================================================
Shows all four Bellman equations as animated backup trees:
  1. Bellman Expectation  – V^pi(s)
  2. Bellman Expectation  – Q^pi(s,a)
  3. Bellman Optimality   – V*(s)
  4. Bellman Optimality   – Q*(s,a)

Outputs
-------
  asserts/bellman_backup_static.png   – 4-panel static reference
  asserts/bellman_backup_anim.gif     – animated version
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "asserts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
C_STATE  = "#4A90D9"   # blue  – state node
C_ACTION = "#E8A838"   # amber – action node
C_EDGE   = "#555555"
C_ACTIVE = "#E74C3C"   # red   – currently lit edge / node
C_MATH   = "#2C3E50"
C_BG     = "#F8F9FA"
C_PANEL  = "#FFFFFF"

NODE_R_S = 0.07   # radius of state node
NODE_R_A = 0.045  # radius of action node


# ── helpers ───────────────────────────────────────────────────────────────────
def draw_circle(ax, xy, r, color, zorder=4, lw=1.5, ec="white", label=None):
    c = Circle(xy, r, color=color, zorder=zorder, lw=lw, ec=ec)
    ax.add_patch(c)
    if label:
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=zorder+1)
    return c


def draw_arrow(ax, p0, p1, color=C_EDGE, lw=1.2, zorder=3, head=0.012):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=f"->,head_width={head*6},head_length={head*4}",
                                color=color, lw=lw),
                zorder=zorder)


def draw_label(ax, xy, txt, fontsize=7.5, color=C_MATH, va="center", ha="left"):
    ax.text(xy[0], xy[1], txt, fontsize=fontsize, color=color, va=va, ha=ha,
            fontfamily="monospace")


def draw_dashed_line(ax, p0, p1, color=C_EDGE, lw=0.8):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color=color, lw=lw, zorder=2)


# ── per-panel drawing functions ───────────────────────────────────────────────
def panel_V_expect(ax, highlight=None):
    """V^pi(s) backup tree."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(C_PANEL)

    # root state node
    s = (0.5, 0.82)
    draw_circle(ax, s, NODE_R_S, C_STATE, label="s")

    # three action branches
    actions = [(0.22, 0.55), (0.50, 0.55), (0.78, 0.55)]
    a_labels = ["a1", "a2", "a3"]
    probs_a  = [0.3, 0.4, 0.3]

    for i, (a_xy, a_lbl, pa) in enumerate(zip(actions, a_labels, probs_a)):
        edge_col = C_ACTIVE if highlight in ("actions", "all") else C_EDGE
        draw_arrow(ax, s, a_xy, color=edge_col)
        ax.text((s[0]+a_xy[0])/2 - 0.06, (s[1]+a_xy[1])/2 + 0.02,
                f"pi={pa}", fontsize=6.5, color=C_MATH, ha="right")
        draw_circle(ax, a_xy, NODE_R_A, C_ACTION, label=a_lbl)

        # two successor state branches per action
        s_primes = [(a_xy[0]-0.09, 0.22), (a_xy[0]+0.09, 0.22)]
        ps_primes = [0.7, 0.3]
        for j, (sp_xy, pp) in enumerate(zip(s_primes, ps_primes)):
            e_col = C_ACTIVE if highlight == "all" else C_EDGE
            draw_arrow(ax, a_xy, sp_xy, color=e_col)
            ax.text((a_xy[0]+sp_xy[0])/2 + 0.02, (a_xy[1]+sp_xy[1])/2,
                    f"P={pp}", fontsize=5.8, color=C_MATH)
            draw_circle(ax, sp_xy, NODE_R_S, C_STATE, label="s'")
            if highlight == "all":
                ax.text(sp_xy[0], sp_xy[1]-0.09, "r+γV(s')",
                        fontsize=6, color=C_ACTIVE, ha="center")

    # formula box
    formula = (r"$V^\pi(s) = \sum_a \pi(a|s)$" + "\n"
               r"$\times\sum_{s'} P(s'|s,a)[r + \gamma V^\pi(s')]$")
    ax.text(0.5, 0.06, formula, ha="center", va="bottom", fontsize=8.5,
            color=C_MATH, bbox=dict(boxstyle="round,pad=0.3", fc="#EAF2FF", ec=C_STATE, lw=1))

    # legend patches
    ax.legend(handles=[
        mpatches.Patch(color=C_STATE,  label="State node  (circle)"),
        mpatches.Patch(color=C_ACTION, label="Action node (circle)"),
    ], fontsize=6.5, loc="upper right", framealpha=0.8)

    ax.set_title("(1)  Bellman Expectation  –  $V^\\pi(s)$",
                 fontsize=10, fontweight="bold", color=C_MATH, pad=6)


def panel_Q_expect(ax, highlight=None):
    """Q^pi(s,a) backup tree."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(C_PANEL)

    # root action node
    a_root = (0.5, 0.82)
    draw_circle(ax, a_root, NODE_R_A, C_ACTION, label="s,a")

    # two successor state branches
    s_primes = [(0.28, 0.55), (0.72, 0.55)]
    ps = [0.6, 0.4]
    for sp_xy, pp in zip(s_primes, ps):
        e_col = C_ACTIVE if highlight in ("states", "all") else C_EDGE
        draw_arrow(ax, a_root, sp_xy, color=e_col)
        ax.text((a_root[0]+sp_xy[0])/2 + 0.03, (a_root[1]+sp_xy[1])/2 + 0.02,
                f"P={pp}", fontsize=6.5, color=C_MATH)
        draw_circle(ax, sp_xy, NODE_R_S, C_STATE, label="s'")

        # action branches from s'
        a_primes = [(sp_xy[0]-0.07, 0.28), (sp_xy[0]+0.07, 0.28)]
        pis = [0.5, 0.5]
        for ap_xy, pi_v in zip(a_primes, pis):
            e_col2 = C_ACTIVE if highlight == "all" else C_EDGE
            draw_arrow(ax, sp_xy, ap_xy, color=e_col2)
            ax.text((sp_xy[0]+ap_xy[0])/2 - 0.05, (sp_xy[1]+ap_xy[1])/2,
                    f"pi={pi_v}", fontsize=5.8, color=C_MATH)
            draw_circle(ax, ap_xy, NODE_R_A, C_ACTION, label="a'")
            if highlight == "all":
                ax.text(ap_xy[0], ap_xy[1]-0.08, "Q(s',a')",
                        fontsize=6, color=C_ACTIVE, ha="center")

    formula = (r"$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)$" + "\n"
               r"$\times [r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$")
    ax.text(0.5, 0.06, formula, ha="center", va="bottom", fontsize=8.5,
            color=C_MATH, bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E7", ec=C_ACTION, lw=1))
    ax.set_title("(2)  Bellman Expectation  –  $Q^\\pi(s,a)$",
                 fontsize=10, fontweight="bold", color=C_MATH, pad=6)


def panel_V_opt(ax, highlight=None):
    """V*(s) backup tree – max over actions."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(C_PANEL)

    s = (0.5, 0.82)
    draw_circle(ax, s, NODE_R_S, C_STATE, label="s")

    actions = [(0.22, 0.55), (0.50, 0.55), (0.78, 0.55)]

    for i, a_xy in enumerate(actions):
        # highlight the "max" action in red when active
        is_best = (i == 1)  # middle action is "best"
        edge_col = C_ACTIVE if (highlight in ("max", "all") and is_best) else C_EDGE
        a_col    = C_ACTIVE if (highlight in ("max", "all") and is_best) else C_ACTION
        draw_arrow(ax, s, a_xy, color=edge_col)
        draw_circle(ax, a_xy, NODE_R_A, a_col, label=f"a{i+1}")
        if highlight in ("max", "all") and is_best:
            ax.text(a_xy[0], a_xy[1]+0.10, "max", fontsize=8,
                    color=C_ACTIVE, ha="center", fontweight="bold")

        s_primes = [(a_xy[0]-0.09, 0.22), (a_xy[0]+0.09, 0.22)]
        ps = [0.7, 0.3]
        for sp_xy, pp in zip(s_primes, ps):
            e_col = C_ACTIVE if (highlight == "all" and is_best) else C_EDGE
            draw_arrow(ax, a_xy, sp_xy, color=e_col)
            ax.text((a_xy[0]+sp_xy[0])/2 + 0.02, (a_xy[1]+sp_xy[1])/2,
                    f"P={pp}", fontsize=5.8, color=C_MATH)
            draw_circle(ax, sp_xy, NODE_R_S, C_STATE, label="s'")

    # "max" bracket
    if highlight in ("max", "all"):
        ax.annotate("", xy=(0.78+0.09+0.01, 0.55), xytext=(0.22-0.09-0.01, 0.55),
                    arrowprops=dict(arrowstyle="-", color=C_ACTIVE, lw=1.5,
                                   connectionstyle="arc3,rad=-0.3"))
        ax.text(0.5, 0.44, "← select max →", fontsize=7, color=C_ACTIVE, ha="center")

    formula = (r"$V^*(s) = \max_a \sum_{s'} P(s'|s,a)$" + "\n"
               r"$\times [r + \gamma V^*(s')]$")
    ax.text(0.5, 0.06, formula, ha="center", va="bottom", fontsize=8.5,
            color=C_MATH, bbox=dict(boxstyle="round,pad=0.3", fc="#FFECEC", ec=C_ACTIVE, lw=1))
    ax.set_title("(3)  Bellman Optimality  –  $V^*(s)$",
                 fontsize=10, fontweight="bold", color="#C0392B", pad=6)


def panel_Q_opt(ax, highlight=None):
    """Q*(s,a) backup tree – max over next actions."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(C_PANEL)

    a_root = (0.5, 0.82)
    draw_circle(ax, a_root, NODE_R_A, C_ACTION, label="s,a")

    s_primes = [(0.28, 0.55), (0.72, 0.55)]
    ps = [0.6, 0.4]
    for sp_xy, pp in zip(s_primes, ps):
        e_col = C_ACTIVE if highlight in ("states", "all") else C_EDGE
        draw_arrow(ax, a_root, sp_xy, color=e_col)
        ax.text((a_root[0]+sp_xy[0])/2 + 0.03, (a_root[1]+sp_xy[1])/2 + 0.02,
                f"P={pp}", fontsize=6.5, color=C_MATH)
        draw_circle(ax, sp_xy, NODE_R_S, C_STATE, label="s'")

        a_primes = [(sp_xy[0]-0.07, 0.28), (sp_xy[0]+0.07, 0.28)]
        for k, ap_xy in enumerate(a_primes):
            is_best = (k == 1)
            e_col2 = C_ACTIVE if (highlight == "all" and is_best) else C_EDGE
            a_col  = C_ACTIVE if (highlight == "all" and is_best) else C_ACTION
            draw_arrow(ax, sp_xy, ap_xy, color=e_col2)
            draw_circle(ax, ap_xy, NODE_R_A, a_col, label="a'")
            if highlight == "all" and is_best:
                ax.text(ap_xy[0]+0.01, ap_xy[1]-0.09, "max Q",
                        fontsize=6, color=C_ACTIVE, ha="center")

    formula = (r"$Q^*(s,a) = \sum_{s'} P(s'|s,a)$" + "\n"
               r"$\times [r + \gamma \max_{a'} Q^*(s',a')]$")
    ax.text(0.5, 0.06, formula, ha="center", va="bottom", fontsize=8.5,
            color=C_MATH, bbox=dict(boxstyle="round,pad=0.3", fc="#FFECEC", ec=C_ACTIVE, lw=1))
    ax.set_title("(4)  Bellman Optimality  –  $Q^*(s,a)$",
                 fontsize=10, fontweight="bold", color="#C0392B", pad=6)


# ── static figure ─────────────────────────────────────────────────────────────
def save_static():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=C_BG)
    fig.suptitle("Bellman Backup Diagrams  –  All Four Forms",
                 fontsize=14, fontweight="bold", color=C_MATH, y=0.98)

    panel_V_expect(axes[0, 0], highlight="all")
    panel_Q_expect(axes[0, 1], highlight="all")
    panel_V_opt   (axes[1, 0], highlight="all")
    panel_Q_opt   (axes[1, 1], highlight="all")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "bellman_backup_static.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── animation ─────────────────────────────────────────────────────────────────
HIGHLIGHT_STEPS = [
    # (panel_idx, highlight_key, title_note)
    (0, None,      "Step 1 – Root state node  s"),
    (0, "actions", "Step 2 – Expand actions  (weighted by pi)"),
    (0, "all",     "Step 3 – Expand transitions  (weighted by P)  →  back up r+γV(s')"),
    (1, None,      "Step 4 – Root action node  (s,a)"),
    (1, "states",  "Step 5 – Expand transitions  (weighted by P)"),
    (1, "all",     "Step 6 – Expand next actions  →  back up Q(s',a')"),
    (2, None,      "Step 7 – Optimality: root state node  s"),
    (2, "max",     "Step 8 – Select max action"),
    (2, "all",     "Step 9 – Back up  r + γV*(s')  for best action"),
    (3, None,      "Step 10 – Optimality: root action node  (s,a)"),
    (3, "states",  "Step 11 – Expand transitions"),
    (3, "all",     "Step 12 – Select  max_a' Q*(s',a')  →  back up"),
]

PANEL_FNS = [panel_V_expect, panel_Q_expect, panel_V_opt, panel_Q_opt]


def save_animation():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=C_BG)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # shared step label at bottom
    step_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10,
                         color=C_ACTIVE, fontweight="bold")
    fig.suptitle("Bellman Backup Diagrams  –  Animated",
                 fontsize=13, fontweight="bold", color=C_MATH)

    # hold 3 extra frames at end
    all_steps = HIGHLIGHT_STEPS + [HIGHLIGHT_STEPS[-1]] * 3

    def animate(frame):
        panel_idx, hl, note = all_steps[frame]
        for ax in axes.flat:
            ax.cla()
        PANEL_FNS[0](axes[0, 0], highlight=hl if panel_idx == 0 else "all" if frame > 2 else None)
        PANEL_FNS[1](axes[0, 1], highlight=hl if panel_idx == 1 else "all" if frame > 5 else None)
        PANEL_FNS[2](axes[1, 0], highlight=hl if panel_idx == 2 else "all" if frame > 8 else None)
        PANEL_FNS[3](axes[1, 1], highlight=hl if panel_idx == 3 else None)
        step_text.set_text(note)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(all_steps),
                                   interval=900, blit=False, repeat=True)
    out = os.path.join(OUT_DIR, "bellman_backup_anim.gif")
    anim.save(out, writer="pillow", fps=1.1, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[bellman_backup_viz]  Generating...")
    save_static()
    save_animation()
    print("[bellman_backup_viz]  Done.")
