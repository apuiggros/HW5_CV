"""Part 1 — Visualise the Binary Cross-Entropy loss surface for a single sample.

BCE(y, y_hat) = - [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

We sweep both y (target) and y_hat (prediction) continuously on [0, 1].
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import savefig


def bce(y, y_hat, eps=1e-7):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def main():
    y      = np.linspace(0, 1, 200)
    y_hat  = np.linspace(0, 1, 200)
    Y, Yh  = np.meshgrid(y, y_hat)
    L      = bce(Y, Yh)

    # ---- 3D surface ----
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, Yh, np.clip(L, 0, 6), cmap="viridis", alpha=0.9)
    ax.set_xlabel(r"target $y$")
    ax.set_ylabel(r"prediction $\hat{y}$")
    ax.set_zlabel("BCE loss (clipped at 6)")
    ax.set_title("Binary Cross-Entropy loss surface")
    savefig("part1_bce_surface.png", fig)

    # ---- 2D heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.clip(L, 0, 6), origin="lower", extent=[0, 1, 0, 1],
                   aspect="auto", cmap="viridis")
    ax.set_xlabel(r"target $y$")
    ax.set_ylabel(r"prediction $\hat{y}$")
    ax.set_title("BCE loss (heatmap, clipped at 6)")
    fig.colorbar(im, ax=ax, label="loss")
    savefig("part1_bce_heatmap.png", fig)

    # ---- 1D slices: classic curves at y=0 and y=1 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_hat, bce(0, y_hat), label="y = 0", lw=2)
    ax.plot(y_hat, bce(1, y_hat), label="y = 1", lw=2)
    ax.set_xlabel(r"prediction $\hat{y}$")
    ax.set_ylabel("BCE loss")
    ax.set_title("BCE loss for the two binary targets")
    ax.set_ylim(0, 6)
    ax.legend(); ax.grid(alpha=0.3)
    savefig("part1_bce_slices.png", fig)

    plt.show()


if __name__ == "__main__":
    main()
