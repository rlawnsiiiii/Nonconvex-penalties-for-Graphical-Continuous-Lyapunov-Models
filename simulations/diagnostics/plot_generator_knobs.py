#!/usr/bin/env python3
"""Illustrations for docs/GRAPH_CLASSES_AND_BETAMIN.md: the two knobs of the generator
(``gclm.data.simulate.sample_drift``; plan of 13 October 2026, §1.4).  **Run locally.**

``generator_classes.png``: what one seed gives under the three graph classes -- the thesis
generator (``random``: every ordered pair with probability d = k / p) and ``dag`` (a fresh draw
oriented along a random order of the nodes) -- with the class ``gclm.graphs.graph_class`` reads off
each support and whether the
truth is identified on it (``identified_on_support`` at C = 2I).
``weight_laws.png``: the law of a true weight |w| under the thesis generator (N(0, 1)) and under
``weights="uniform"`` for the four w_min of experiment E3, and the share of true edges weaker
than x -- what each lower bound removes from the thesis generator's graphs.
``dag_construction.png``: the three steps of the ``dag`` branch on one small seed -- a random order
of the nodes, the p (p - 1) / 2 forward slots, a Bernoulli(min(2d, 1)) coin on each -- with the result
in the node labels and its support matrix before and after sorting rows and columns by the order
(strictly lower triangular: the picture of "no directed cycle").  ``print_dag_steps`` prints the same
example as text for the note.

    python simulations/diagnostics/plot_generator_knobs.py                     # -> docs/figures/
    python simulations/diagnostics/plot_generator_knobs.py --out some/dir --seed 3 --p 8 --k 2 --dag-seed 7 --dag-p 6 --dag-k 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gclm.data.simulate import sample_drift  # noqa: E402
from gclm.graphs import graph_class, identified_on_support, two_cycles  # noqa: E402
from plot_bidirectional import save  # noqa: E402
from plot_campaign import GREY, INK, INK_2, SURFACE, style_axes  # noqa: E402

#: the three graph classes of experiment E4 (the pairs class of E5 is a knob, not a class: note §2.3)
CLASSES = ("random", "dag")
#: the four lower bounds of experiment E3 (wave 9, group betamin)
W_MINS = (0.0, 0.1, 0.25, 0.5)
EDGE_COLOUR, PAIR_COLOUR = "#2a78d6", "#eb6834"
TITLE = {"random": "random: every ordered pair\nwith probability d = k / p",
         "dag": "dag: a fresh draw, edges only\nalong a random order of the nodes"}
CLASS_LABEL = {"dag": "DAG", "simple": "cyclic, no 2-cycle", "two_cycles": "with 2-cycles"}
#: the uniform laws: one hue, light to dark with w_min
UNIFORM_COLOURS = ("#f6c4ad", "#f19a76", "#eb6834", "#b4461c")


def example_supports(seed: int = 3, p: int = 8, k: int = 2) -> dict[str, np.ndarray]:
    """The drift matrices one seed gives under the two classes: ``random``, and ``dag``, which draws
    afresh along a random order."""
    return {g: sample_drift(p, k / p, np.random.default_rng(seed), graph=g) for g in CLASSES}


def draw_graph(ax, m: np.ndarray, title: str) -> None:
    """The graph of ``m`` on a circle: ``M_ij != 0`` is the edge ``j -> i``; a pair with both
    directions is drawn in the pair colour.  Below it the class and whether the truth is identified
    on its support."""
    p = m.shape[0]
    theta = np.pi / 2 + 2 * np.pi * np.arange(p) / p
    pos = np.c_[np.cos(theta), np.sin(theta)]
    support = m != 0
    np.fill_diagonal(support, False)
    for i, j in zip(*np.nonzero(support)):
        pair = bool(support[j, i])
        ax.annotate("", xy=pos[i], xytext=pos[j],
                    arrowprops=dict(arrowstyle="-|>", color=PAIR_COLOUR if pair else EDGE_COLOUR,
                                    lw=1.6 if pair else 1.3, shrinkA=10, shrinkB=10, mutation_scale=12,
                                    connectionstyle="arc3,rad=0.18"), zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=300, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.2, zorder=3)
    for v in range(p):
        ax.text(pos[v, 0], pos[v, 1], str(v + 1), ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
    n_edges, n_pairs = int(support.sum()), two_cycles(support)
    ident = identified_on_support(m, 2.0 * np.eye(p)) > 1e-10
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.text(0.0, -1.42, f"{n_edges} edges, {n_pairs} two-cycle{'s' if n_pairs != 1 else ''}: class "
                        f"{CLASS_LABEL[graph_class(support)]}\nidentified on its support: {'yes' if ident else 'no'}",
            ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.85, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_classes(mats: dict[str, np.ndarray], out: Path, seed: int, k: int, name: str = "generator_classes") -> None:
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 4.4))
    for ax, (g, m) in zip(np.atleast_1d(axes), mats.items()):
        draw_graph(ax, m, TITLE[g])
    p = next(iter(mats.values())).shape[0]
    fig.suptitle(f"one seed ({seed}), p = {p}, k = {k}: the graph classes of the generator",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.legend(handles=[Line2D([], [], color=EDGE_COLOUR, lw=1.5, label="single edge"),
                        Line2D([], [], color=PAIR_COLOUR, lw=1.8, label="both directions of a pair (a 2-cycle)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, out, name)


def density(law, x: np.ndarray) -> np.ndarray:
    """The density of |w|: ``law="normal"`` is |N(0, 1)| (half-normal); a number is U[w_min, 1]."""
    if law == "normal":
        return np.sqrt(2 / np.pi) * np.exp(-x ** 2 / 2)
    w = float(law)
    return np.where((x >= w) & (x <= 1.0), 1.0 / (1.0 - w), 0.0)


def share_below(law, x: float) -> float:
    """The share of true weights with |w| < x under the law."""
    if law == "normal":
        return math.erf(x / math.sqrt(2))
    w = float(law)
    return float(min(max((x - w) / (1.0 - w), 0.0), 1.0))


def fig_weight_laws(out: Path, w_mins=W_MINS, name: str = "weight_laws") -> None:
    laws = ["normal", *w_mins]
    colours = {"normal": EDGE_COLOUR, **dict(zip(w_mins, UNIFORM_COLOURS))}
    labels = {"normal": "N(0, 1), the thesis generator", **{w: f"U[{w:g}, 1]" for w in w_mins}}
    bounds = [w for w in w_mins if w > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    x = np.linspace(0, 3, 601)
    for ax in axes:
        style_axes(ax)
        for b in bounds:
            ax.axvline(b, color=GREY, lw=0.7, ls=":", zorder=1)
    for law in laws:
        lw = 1.8 if law == "normal" else 1.3
        axes[0].plot(x, density(law, x), color=colours[law], lw=lw, label=labels[law], zorder=3)
        axes[1].plot(x, [share_below(law, v) for v in x], color=colours[law], lw=lw, zorder=3)
    axes[0].set_ylim(0, 2.2)
    axes[0].set_ylabel("density of |w|", fontsize=9, color=INK_2)
    axes[0].set_title("the law of a true weight", loc="left", fontsize=9.5, color=INK)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("share of true edges with |w| below x", fontsize=9, color=INK_2)
    axes[1].set_title("how many true edges are weaker than x", loc="left", fontsize=9.5, color=INK)
    for ax in axes:
        ax.set_xlim(0, 3)
        ax.set_xlabel("|w|   (dotted: the lower bounds w_min of E3)", fontsize=9, color=INK_2)
    fig.legend(loc="lower center", ncol=len(laws), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out, name)


def print_shares(w_mins=W_MINS, xs=(0.1, 0.25, 0.5, 1.0)) -> None:
    """The share of true edges weaker than x per law: at x = w_min, what the lower bound removes
    from the thesis generator's graphs."""
    print("share of true edges with |w| below x, per law of the weights")
    print(f"  {'law':<12}" + "".join(f"{'x = ' + format(b, 'g'):>10}" for b in xs))
    for law in ["normal", *w_mins]:
        label = "N(0, 1)" if law == "normal" else f"U[{law:g}, 1]"
        print(f"  {label:<12}" + "".join(f"{share_below(law, b):>10.3f}" for b in xs))


def dag_construction_steps(seed: int = 7, p: int = 6, k: int = 1) -> dict:
    """Replay the draws of ``sample_drift(p, k / p, rng, graph="dag")`` one by one, so that the figure
    and the printed example show exactly what the generator returns for the seed: ``eps`` and the
    Bernoulli(d) matrix of the thesis generator (both drawn first; the dag branch leaves the latter
    unused), then the order ``position`` (``position[v] = pi(v)``), the mask ``allowed`` (entry
    ``(i, j)``: ``pi(j) < pi(i)``, the forward slots), the coins ``Bernoulli(min(2d, 1))`` and the
    support ``omega = coins * allowed``.  Checked against the generator itself."""
    d = k / p
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(p, p))
    unused = rng.binomial(1, d, size=(p, p))
    position = rng.permutation(p)
    coins = rng.binomial(1, min(2.0 * d, 1.0), size=(p, p))
    allowed = position[None, :] < position[:, None]
    omega = coins * allowed
    m = sample_drift(p, d, np.random.default_rng(seed), graph="dag")
    support = m != 0
    np.fill_diagonal(support, False)
    if not np.array_equal(support, omega.astype(bool)):
        raise AssertionError("the replayed steps do not reproduce sample_drift's support")
    order = np.argsort(position)                                    # order[r] = the node at position r
    return {"seed": seed, "p": p, "k": k, "d": d, "d_prime": min(2.0 * d, 1.0), "eps": eps, "unused": unused,
            "position": position, "order": order, "allowed": allowed, "coins": coins, "omega": omega, "m": m}


def draw_order(ax, steps: dict, coins: bool, title: str) -> None:
    """The nodes on a line at their positions pi(v), labelled with the node number; above the line the
    forward slots as arcs (``coins=False``: all of them, in grey) or the coins (``coins=True``: a kept
    slot in the edge colour, a dropped one dashed in grey).  No arc can point to the left."""
    p, position, allowed, omega = steps["p"], steps["position"], steps["allowed"], steps["omega"]
    for i, j in zip(*np.nonzero(allowed)):                          # the slot j -> i, pi(j) < pi(i)
        kept = bool(omega[i, j])
        if coins:
            colour, lw, ls = (EDGE_COLOUR, 1.6, "-") if kept else ("#c9c7c2", 0.9, "--")
        else:
            colour, lw, ls = GREY, 1.0, "-"
        ax.annotate("", xy=(position[i], 0.0), xytext=(position[j], 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, linestyle=ls, shrinkA=9, shrinkB=9,
                                    mutation_scale=11, connectionstyle="arc3,rad=-0.42"), zorder=2)
    ax.scatter(position, np.zeros(p), s=330, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.2, zorder=3)
    for v in range(p):
        ax.text(position[v], 0.0, str(v + 1), ha="center", va="center", fontsize=9, color=INK, zorder=4)
        ax.text(position[v], -0.33, rf"$\pi$ = {position[v] + 1}", ha="center", va="top", fontsize=7.5, color=INK_2)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlim(-0.7, p - 0.3)
    ax.set_ylim(-0.7, 0.21 * (p - 1) + 0.3)      # with an equal aspect the arcs rise 0.21 × span above the line
    ax.set_aspect("equal")
    ax.axis("off")


def draw_support_matrix(ax, steps: dict, sorted_by_order: bool, title: str) -> None:
    """The support ``omega`` as a grid: a kept slot in the edge colour with a 1, an open slot that
    lost its coin white with a 0, a blocked slot (it would point backwards, or the diagonal) grey.
    ``sorted_by_order`` permutes rows and columns by pi: the blocked slots then form the diagonal and
    the strict upper triangle, and the matrix is strictly lower triangular."""
    p, omega, allowed = steps["p"], steps["omega"], steps["allowed"]
    idx = steps["order"] if sorted_by_order else np.arange(p)
    om, al = omega[np.ix_(idx, idx)], allowed[np.ix_(idx, idx)]
    colours = np.full((p, p, 3), 1.0)
    colours[~al] = (0.87, 0.865, 0.85)                              # blocked
    colours[al & (om == 1)] = (0.63, 0.78, 0.94)                    # kept
    ax.imshow(colours, interpolation="nearest")
    for a in range(p):
        for b in range(p):
            if al[a, b]:
                ax.text(b, a, str(int(om[a, b])), ha="center", va="center", fontsize=8.5,
                        color=INK if om[a, b] else INK_2)
            elif a != b:
                ax.text(b, a, "–", ha="center", va="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    labels = [str(v + 1) for v in idx]
    if sorted_by_order:
        ax.set_xticklabels([f"{lab}\n$\\pi$={r + 1}" for r, lab in enumerate(labels)], fontsize=7.5)
        ax.set_yticklabels([f"{lab} ($\\pi$={r + 1})" for r, lab in enumerate(labels)], fontsize=7.5)
    else:
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0, colors=INK_2)
    ax.set_xticks(np.arange(-0.5, p, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, p, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("column j  (entry (i, j) is the edge j → i)", fontsize=8, color=INK_2)
    ax.set_ylabel("row i", fontsize=8, color=INK_2)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def fig_dag_construction(steps: dict, out: Path, name: str = "dag_construction") -> None:
    p, k, d, dp = steps["p"], steps["k"], steps["d"], steps["d_prime"]
    n_open, n_kept = int(steps["allowed"].sum()), int(steps["omega"].sum())
    fig = plt.figure(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.14)
    gs = fig.add_gridspec(2, 6, height_ratios=(0.78, 1.25), hspace=0.28, wspace=0.65)
    ax1, ax2 = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    ax3, ax4, ax5 = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])
    draw_order(ax1, steps, coins=False,
               title=f"1–2.  a random order $\\pi$ of the nodes, and the {n_open} forward slots\n"
                     f"(the slot j → i is open only if $\\pi$(j) < $\\pi$(i): no arrow may point left)")
    draw_order(ax2, steps, coins=True,
               title=f"3.  an independent coin with probability min(2d, 1) = {dp:.3g} on every open slot\n"
                     f"here {n_kept} of {n_open} kept (expected {n_open * dp:.3g} = p(p−1)d, the random class's count)")
    draw_graph(ax3, steps["m"], "the result in the node labels,\nas sample_drift returns it")
    draw_support_matrix(ax4, steps, sorted_by_order=False,
                        title="its support ω in the node labels\n(grey: blocked slot, would point backwards)")
    draw_support_matrix(ax5, steps, sorted_by_order=True,
                        title="rows and columns sorted by $\\pi$: strictly lower\ntriangular, so no walk returns to its start")
    fig.suptitle(f"how the dag branch builds a DAG: seed {steps['seed']}, p = {p}, k = {k} (d = k/p = {d:.3g})",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.legend(handles=[Line2D([], [], color=GREY, lw=1.0, label="open slot"),
                        Line2D([], [], color=EDGE_COLOUR, lw=1.6, label="coin kept: an edge"),
                        Line2D([], [], color="#c9c7c2", lw=0.9, ls="--", label="coin dropped")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    save(fig, out, name)


def print_dag_steps(steps: dict) -> None:
    """The example of ``fig_dag_construction`` as text: the order, the edges with their positions,
    and the support matrix with rows and columns sorted by pi."""
    p, position, order, omega, allowed = steps["p"], steps["position"], steps["order"], steps["omega"], steps["allowed"]
    print(f"the dag branch, seed {steps['seed']}, p = {p}, k = {steps['k']}: d = {steps['d']:.4g}, "
          f"coin probability min(2d, 1) = {steps['d_prime']:.4g}")
    print("  nodes in the order drawn (position 1 .. p): " + "  ".join(str(v + 1) for v in order))
    print(f"  open slots: {int(allowed.sum())} of {p * (p - 1)} ordered pairs;  edges kept: {int(omega.sum())}")
    for i, j in zip(*np.nonzero(omega)):
        print(f"    edge {j + 1} -> {i + 1}   (position {position[j] + 1} -> {position[i] + 1})")
    print("  the support omega with rows and columns sorted by pi  ('-' = blocked slot, 0/1 = the coin on an open slot):")
    head = "  ".join(f"{v + 1:>2}" for v in order)
    print(f"    {'':<16}{head}")
    for r, i in enumerate(order):
        cells = []
        for j in order:
            cells.append(f"{int(omega[i, j]):>2}" if allowed[i, j] else f"{'-':>2}")
        print(f"    node {i + 1} (pi = {r + 1})  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dag-seed", type=int, default=7, help="the seed of the DAG-construction example")
    ap.add_argument("--dag-p", type=int, default=6)
    ap.add_argument("--dag-k", type=int, default=1)
    args = ap.parse_args()
    fig_classes(example_supports(args.seed, args.p, args.k), args.out, args.seed, args.k)
    fig_weight_laws(args.out)
    print_shares()
    steps = dag_construction_steps(args.dag_seed, args.dag_p, args.dag_k)
    fig_dag_construction(steps, args.out)
    print()
    print_dag_steps(steps)


if __name__ == "__main__":
    main()
