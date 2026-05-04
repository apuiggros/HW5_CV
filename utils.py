"""Shared helpers: 3D checkerboard data + plotting."""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

FIG_DIR = os.path.join(os.path.dirname(__file__), "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name, fig=None, dpi=150):
    """Save a matplotlib figure into report/figures/."""
    fig = fig if fig is not None else plt.gcf()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved -> {path}")


def make_xor3d(n_per_cluster=200, radius=0.3, seed=0):
    """3D checkerboard with 8 clusters at the corners of the unit cube.

    Labels follow the parity of (i+j+k): adjacent clusters get opposite labels,
    so no two neighbours share a label (this is the 3D XOR pattern).

    Points are sampled uniformly inside a sphere of `radius` around each
    centre, then clipped so they stay inside their own cube cell of side 1.

    Returns
    -------
    X : (N, 3) float array
    y : (N,)   int array of 0/1 labels
    """
    rng = np.random.default_rng(seed)
    centres = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                       dtype=float)
    labels  = np.array([(i + j + k) % 2 for i, j, k in centres], dtype=int)

    Xs, ys = [], []
    for c, lab in zip(centres, labels):
        # sample uniformly in a sphere of radius `radius`
        pts = rng.normal(size=(n_per_cluster, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        r   = rng.uniform(0, radius, size=(n_per_cluster, 1)) ** (1 / 3) * radius
        pts = c + pts * r
        # clip to the cell [c-0.5, c+0.5]^3 so points don't bleed into neighbours
        keep = np.all(np.abs(pts - c) <= 0.5, axis=1)
        Xs.append(pts[keep])
        ys.append(np.full(keep.sum(), lab))
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def plot_xor3d(X, y, title="3D XOR checkerboard", ax=None):
    """Scatter-plot the dataset; class 0 = blue, class 1 = red."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
               c="tab:blue", s=8, alpha=0.6, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
               c="tab:red",  s=8, alpha=0.6, label="class 1")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax
