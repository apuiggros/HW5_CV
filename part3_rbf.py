"""Part 3 (Bonbon) — RBF network for the 3D XOR problem.

Architecture:
  input (3) -> RBF hidden layer (K Gaussian units) -> linear -> sigmoid

Centres are placed with k-means on the training data; the bandwidth (sigma)
is set heuristically from the mean inter-centre distance. Only the output
weights are trained, with BCE loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from utils import make_xor3d, plot_xor3d, savefig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RBFNet(nn.Module):
    """RBF network with fixed centres and a single trainable linear output."""

    def __init__(self, centres, sigma):
        super().__init__()
        # centres: (K, D) tensor — frozen
        self.register_buffer("centres", torch.as_tensor(centres, dtype=torch.float32))
        self.sigma = float(sigma)
        self.linear = nn.Linear(centres.shape[0], 1)

    def forward(self, x):
        # x: (N, D); centres: (K, D); distances: (N, K)
        diff = x.unsqueeze(1) - self.centres.unsqueeze(0)
        d2   = (diff ** 2).sum(-1)
        phi  = torch.exp(-d2 / (2 * self.sigma ** 2))
        return torch.sigmoid(self.linear(phi)).squeeze(-1)


def fit_rbf(X, y, K=8, epochs=300, lr=0.05, seed=0):
    """Fit centres with k-means, then train the output layer with BCE."""
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X)
    centres = km.cluster_centers_

    # bandwidth: mean pairwise distance between centres / sqrt(2K)
    from scipy.spatial.distance import pdist
    sigma = pdist(centres).mean() / np.sqrt(2 * K)

    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    model = RBFNet(centres, sigma).to(DEVICE)
    opt   = optim.Adam(model.linear.parameters(), lr=lr)  # only output weights
    bce   = nn.BCELoss()

    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out  = model(Xt)
        loss = bce(out, yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        pred = (model(Xt) > 0.5).long().cpu().numpy()
    acc = (pred == y).mean()
    return model, acc, losses, centres, sigma


def main():
    torch.manual_seed(0); np.random.seed(0)

    X, y = make_xor3d(n_per_cluster=200, radius=0.3, seed=0)

    # ---- 1. fit a reference RBF with K=8 (one centre per cluster) ----
    print("Training RBF network (K=8) ...")
    model, acc, losses, centres, sigma = fit_rbf(X, y, K=8, epochs=300)
    print(f"K=8 train accuracy: {acc*100:.2f}%, sigma={sigma:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.set_title(f"RBF training loss (K=8, sigma={sigma:.2f})")
    ax.grid(alpha=0.3)
    savefig("part3_rbf_loss.png", fig)

    # ---- 2. show the learned centres on top of the data ----
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")
    plot_xor3d(X, y, title="3D XOR + RBF centres (K=8)", ax=ax)
    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
               c="black", s=120, marker="X", label="RBF centres")
    ax.legend(loc="upper left", fontsize=8)
    savefig("part3_rbf_centres.png", fig)

    # ---- 3. compare RBF vs MLP across cluster radii ----
    radii = [0.15, 0.25, 0.35, 0.45, 0.55]
    rbf_accs = []
    repeats  = 3
    for r in radii:
        scores = []
        for s in range(repeats):
            Xr, yr = make_xor3d(n_per_cluster=200, radius=r, seed=s)
            n_tr   = int(0.8 * len(Xr))
            _, _, _, _, _ = fit_rbf(Xr[:n_tr], yr[:n_tr], K=8, epochs=300, seed=s)
            # re-evaluate on the held-out 20%
            model, _, _, _, _ = fit_rbf(Xr[:n_tr], yr[:n_tr], K=8, epochs=300, seed=s)
            with torch.no_grad():
                Xt = torch.from_numpy(Xr[n_tr:]).to(DEVICE)
                pred = (model(Xt) > 0.5).long().cpu().numpy()
            scores.append((pred == yr[n_tr:]).mean())
        rbf_accs.append(np.mean(scores))
        print(f"radius={r:.2f}  RBF acc={rbf_accs[-1]*100:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(radii, np.array(rbf_accs) * 100, marker="o", lw=2,
            label="RBF (K=8)", color="tab:purple")
    ax.set_xlabel("cluster radius"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("RBF network — accuracy vs cluster radius")
    ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(50, 102)
    savefig("part3_rbf_vs_radius.png", fig)

    plt.show()


if __name__ == "__main__":
    main()
