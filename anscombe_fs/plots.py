"""Figures for the quartet and the score comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["I", "II", "III", "IV"]


def quartet_grid(df: pd.DataFrame, scores: pd.DataFrame):
    """2x2 scatter grid, each panel annotated with every measure's score."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, DATASETS):
        g = df[df["dataset"] == name]
        ax.scatter(g["x"], g["y"], s=40, color="#3b6ea8", zorder=3)
        slope, intercept = np.polyfit(g["x"], g["y"], 1)
        xs = np.array([2, 20])
        ax.plot(xs, intercept + slope * xs, color="#999999", lw=1, zorder=2)
        s = scores.loc[name]
        text = (
            f"Pearson  {s['Pearson r']:.2f}\n"
            f"Spearman {s['Spearman rho']:.2f}\n"
            f"Kendall  {s['Kendall tau-b']:.2f}\n"
            f"MI       {s['Mutual information']:.2f}\n"
            f"dCor     {s['Distance correlation']:.2f}\n"
            f"Chi2 p   {s['Chi-square p']:.3f}"
        )
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", family="monospace", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="#dddddd"))
        ax.set_title(f"Dataset {name}")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle("Same Pearson r, very different data: what each filter score sees", fontsize=12)
    fig.tight_layout()
    return fig


def score_heatmap(scores: pd.DataFrame, columns: list[str]):
    """Heatmap of the chosen score columns, normalised per column for colour only."""
    data = scores[columns]
    norm = (data - data.min()) / (data.max() - data.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(columns) + 1.5, 3.2))
    ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    ax.set_yticks(range(len(data)), [f"Dataset {i}" for i in data.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = norm.values[i, j]
            ax.text(j, i, f"{data.values[i, j]:.3f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_title("Raw scores (colour = rank within each measure)")
    fig.tight_layout()
    return fig
