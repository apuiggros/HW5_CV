"""Part 2 — 3D XOR point cloud classification with a 1-hidden-layer FFN.

Workflow:
  1. Generate the 3D checkerboard.
  2. Train a small MLP (input -> hidden -> 1, sigmoid output, BCE loss).
  3. Sweep cluster radius x hidden-layer size and report test accuracy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import make_xor3d, plot_xor3d, savefig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- model ----------
class MLP(nn.Module):
    """1 hidden layer, tanh activation, sigmoid output (so we can use BCE)."""

    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def he_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        layer.bias.data.fill_(0.0)


# ---------- training ----------
def train_mlp(X, y, hidden=8, epochs=400, lr=0.05, verbose=False):
    """Train an MLP on the full dataset; return final accuracy and the model."""
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    model = MLP(hidden).to(DEVICE)
    model.apply(he_init)
    opt   = optim.Adam(model.parameters(), lr=lr)
    bce   = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(Xt)
        loss = bce(out, yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  loss = {loss.item():.4f}")

    # accuracy on the same data (with this much data it's effectively the test set;
    # we also do a held-out check below for the headline result)
    model.eval()
    with torch.no_grad():
        pred = (model(Xt) > 0.5).long().cpu().numpy()
    acc = (pred == y).mean()
    return model, acc, losses


# ---------- sweep ----------
def sweep(radii, hiddens, n_per_cluster=200, epochs=400, repeats=3):
    """Return a (len(radii), len(hiddens)) array of mean test accuracies."""
    accs = np.zeros((len(radii), len(hiddens)))
    for i, r in enumerate(radii):
        for j, h in enumerate(hiddens):
            scores = []
            for s in range(repeats):
                X, y = make_xor3d(n_per_cluster=n_per_cluster, radius=r, seed=s)
                # 80/20 split
                n_tr = int(0.8 * len(X))
                X_tr, y_tr = X[:n_tr], y[:n_tr]
                X_te, y_te = X[n_tr:], y[n_tr:]
                model, _, _ = train_mlp(X_tr, y_tr, hidden=h, epochs=epochs)
                with torch.no_grad():
                    Xt = torch.from_numpy(X_te).to(DEVICE)
                    pred = (model(Xt) > 0.5).long().cpu().numpy()
                scores.append((pred == y_te).mean())
            accs[i, j] = np.mean(scores)
            print(f"radius={r:.2f}  hidden={h:2d}  acc={accs[i,j]*100:.2f}%")
    return accs


# ---------- main ----------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. visualise the dataset ----
    X, y = make_xor3d(n_per_cluster=200, radius=0.3, seed=0)
    print(f"dataset: {len(X)} points, class balance = {y.mean():.3f}")

    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")
    plot_xor3d(X, y, title="3D XOR — radius = 0.3", ax=ax)
    savefig("part2_dataset.png", fig)

    # ---- 2. train a single reference model ----
    print("\nTraining reference MLP (hidden=8) ...")
    model, acc, losses = train_mlp(X, y, hidden=8, epochs=400, verbose=True)
    print(f"reference accuracy: {acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Training loss — MLP (hidden=8, radius=0.3)")
    ax.grid(alpha=0.3)
    savefig("part2_loss_curve.png", fig)

    # ---- 3. radius x hidden sweep (full: 5 x 5) ----
    radii   = [0.15, 0.25, 0.35, 0.45, 0.55]
    hiddens = [2, 4, 8, 16, 32]
    print(f"\nSweep over {len(radii)} radii x {len(hiddens)} hidden sizes "
          f"(this is the slow bit, ~5 min)...")
    accs = sweep(radii, hiddens, n_per_cluster=200, epochs=300, repeats=3)

    # heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(accs * 100, origin="lower", aspect="auto", cmap="viridis",
                   vmin=50, vmax=100)
    ax.set_xticks(range(len(hiddens))); ax.set_xticklabels(hiddens)
    ax.set_yticks(range(len(radii)));   ax.set_yticklabels([f"{r:.2f}" for r in radii])
    ax.set_xlabel("hidden neurons"); ax.set_ylabel("cluster radius")
    ax.set_title("Test accuracy (%) — radius x hidden sweep")
    for i in range(len(radii)):
        for j in range(len(hiddens)):
            ax.text(j, i, f"{accs[i,j]*100:.1f}", ha="center", va="center",
                    color="white" if accs[i,j] < 0.85 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="accuracy (%)")
    savefig("part2_sweep_heatmap.png", fig)

    # accuracy vs hidden, one curve per radius
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(radii):
        ax.plot(hiddens, accs[i] * 100, marker="o", label=f"r={r:.2f}")
    ax.set_xlabel("hidden neurons"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Effect of hidden-layer size for different cluster radii")
    ax.grid(alpha=0.3); ax.legend()
    savefig("part2_sweep_curves.png", fig)

    plt.show()


if __name__ == "__main__":
    main()
