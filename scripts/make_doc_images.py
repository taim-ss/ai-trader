"""Regenerate the README images in docs/assets (needs matplotlib).

architecture.png is a drawing of the pipeline. verification.png renders the REAL output of
`pytest -q` and `mypy --strict src` from the moment you run this script.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
BG, FG, MUTE = "#0d1117", "#e6edf3", "#8b949e"
GREEN, BLUE, AMBER, RED = "#3fb950", "#58a6ff", "#d29922", "#f85149"


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        (0.2, "LLM analyst", "scheduled agent\nresearch + entry rules", BLUE),
        (2.55, "Validation", "schema, caps,\nfreshness, reasons", GREEN),
        (4.9, "Risk engine", "cash, quotes, loss halt,\ndrawdown latch, HALT", GREEN),
        (7.25, "Human approval", "tap Approve on phone\nsigned, order-bound", AMBER),
        (9.6, "Executor", "journal first, one DAY\nlimit order, reconcile", GREEN),
    ]
    for x, title, sub, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.7), 2.1, 1.5, boxstyle="round,pad=0.04", fc="#161b22", ec=colour, lw=2))
        ax.text(x + 1.05, 2.85, title, color=FG, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 1.05, 2.15, sub, color=MUTE, ha="center", va="center", fontsize=8.5)
    for x in (2.3, 4.65, 7.0, 9.35):
        ax.annotate("", xy=(x + 0.22, 2.45), xytext=(x, 2.45), arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.6))
    ax.text(1.25, 1.35, "proposes JSON,\nnever touches the broker", color=BLUE, ha="center", fontsize=8.5)
    ax.text(5.95, 1.35, "plain code decides what may exist", color=GREEN, ha="center", fontsize=8.5)
    ax.text(10.65, 1.35, "sent order == shown order", color=GREEN, ha="center", fontsize=8.5)
    ax.text(8.3, 3.55, "no tap = no order", color=AMBER, ha="center", fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 11.5, 0.7, boxstyle="round,pad=0.04", fc="#161b22", ec=MUTE, lw=1.2, ls="--"))
    ax.text(5.95, 0.55, "Forward-test arena: the analyst is scored against baselines on prices that did not exist when it committed",
            color=FG, ha="center", va="center", fontsize=9.5)
    ax.text(0.2, 4.35, "AI Trader: an LLM proposes, plain code disposes, a human approves", color=FG, fontsize=14, fontweight="bold")
    fig.savefig(OUT / "architecture.png", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=OUT.parents[1])
    return (res.stdout + res.stderr).strip()


def verification() -> None:
    tests = run([sys.executable, "-m", "pytest", "-q"]).splitlines()[-1]
    types = run([sys.executable, "-m", "mypy", "--strict", "src"]).splitlines()[-1]
    lines = [("$ pytest -q", MUTE), (tests, GREEN), ("$ mypy --strict src", MUTE), (types, GREEN)]
    fig, ax = plt.subplots(figsize=(8, 2.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    for i, (text, colour) in enumerate(lines):
        ax.text(0.02, 0.85 - i * 0.24, text, color=colour, family="monospace", fontsize=12, transform=ax.transAxes)
    fig.savefig(OUT / "verification.png", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    verification()
