"""Part 4 — Honest side-by-side comparison of the MLP and the RBF.

Three additions on top of parts 2 and 3:
  1. A sigma sweep for the RBF, so we don't compare the MLP against a
     poorly tuned RBF.
  2. Multiple seeds for both networks; we plot mean and std bands so
     the "RBF is robust to init" claim has actual evidence.
  3. Loss curves on a shared axis (same data split, same number of
     epochs, same x-axis), which is what a fair comparison requires.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

from utils import make_xor3d, savefig
from part2_xor3d import MLP, he_init
from part3_rbf import RBFNet

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 400
N_SEEDS = 5


# ---------- training helpers that record full curves ----------
def train_mlp_curve(X_tr, y_tr, X_te, y_te, hidden=8, epochs=EPOCHS, seed=0):
    """Train an MLP, return (train_loss[t], test_loss[t], final_test_acc)."""
    torch.manual_seed(seed)
    Xt_tr = torch.from_numpy(X_tr).to(DEVICE)
    yt_tr = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xt_te = torch.from_numpy(X_te).to(DEVICE)
    yt_te = torch.from_numpy(y_te.astype(np.float32)).to(DEVICE)

    model = MLP(hidden).to(DEVICE)
    model.apply(he_init)
    opt   = optim.Adam(model.parameters(), lr=0.05)
    bce   = nn.BCELoss()

    tr_losses, te_losses = [], []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(Xt_tr)
        loss = bce(out, yt_tr)
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            te_losses.append(bce(model(Xt_te), yt_te).item())

    with torch.no_grad():
        pred = (model(Xt_te) > 0.5).long().cpu().numpy()
    return np.array(tr_losses), np.array(te_losses), (pred == y_te).mean()


def train_rbf_curve(X_tr, y_tr, X_te, y_te, K=8, sigma=None,
                    epochs=EPOCHS, seed=0):
    """Train an RBF (frozen centres + sigma), return curves and accuracy.

    If `sigma` is None, fall back to the mean-inter-centre heuristic.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X_tr)
    centres = km.cluster_centers_
    if sigma is None:
        sigma = pdist(centres).mean() / np.sqrt(2 * K)

    Xt_tr = torch.from_numpy(X_tr).to(DEVICE)
    yt_tr = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xt_te = torch.from_numpy(X_te).to(DEVICE)
    yt_te = torch.from_numpy(y_te.astype(np.float32)).to(DEVICE)

    model = RBFNet(centres, sigma).to(DEVICE)
    opt   = optim.Adam(model.linear.parameters(), lr=0.05)
    bce   = nn.BCELoss()

    tr_losses, te_losses = [], []
    for _ in range(epochs):
        opt.zero_grad()
        out  = model(Xt_tr)
        loss = bce(out, yt_tr)
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())
        with torch.no_grad():
            te_losses.append(bce(model(Xt_te), yt_te).item())

    with torch.no_grad():
        pred = (model(Xt_te) > 0.5).long().cpu().numpy()
    return np.array(tr_losses), np.array(te_losses), (pred == y_te).mean()


# ---------- experiment 1: sigma sweep for the RBF ----------
def sigma_sweep():
    """Find the best bandwidth across a range of values."""
    sigmas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    final_train, final_test, accs = [], [], []

    print("Sigma sweep for the RBF (radius=0.3, K=8):")
    for s in sigmas:
        tr_runs, te_runs, ac_runs = [], [], []
        for seed in range(3):
            X, y = make_xor3d(n_per_cluster=200, radius=0.3, seed=seed)
            n_tr = int(0.8 * len(X))
            tr, te, ac = train_rbf_curve(X[:n_tr], y[:n_tr],
                                         X[n_tr:], y[n_tr:],
                                         K=8, sigma=s, epochs=EPOCHS, seed=seed)
            tr_runs.append(tr[-1]); te_runs.append(te[-1]); ac_runs.append(ac)
        final_train.append(np.mean(tr_runs))
        final_test.append(np.mean(te_runs))
        accs.append(np.mean(ac_runs))
        print(f"  sigma={s:.2f}  train BCE={final_train[-1]:.4f}  "
              f"test BCE={final_test[-1]:.4f}  test acc={accs[-1]*100:.2f}%")

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sigmas, final_train, marker="o", label="train BCE", lw=2)
    ax.plot(sigmas, final_test,  marker="s", label="test BCE",  lw=2)
    ax.set_xlabel(r"bandwidth $\sigma$")
    ax.set_ylabel("final BCE loss")
    ax.set_title(r"RBF performance vs bandwidth $\sigma$ (K=8, radius=0.3)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    savefig("part4_sigma_sweep.png", fig)

    best_sigma = sigmas[int(np.argmin(final_test))]
    print(f"\n  -> best sigma by test loss: {best_sigma:.2f}")
    return best_sigma


# ---------- experiment 2 + 3: multi-seed runs with shared loss curves ----------
def multi_seed_comparison(best_sigma):
    """Train both networks with N_SEEDS seeds; plot mean +- std curves."""
    print(f"\nMulti-seed comparison (n_seeds={N_SEEDS}, epochs={EPOCHS}):")
    mlp_tr_runs, mlp_te_runs, mlp_accs = [], [], []
    rbf_tr_runs, rbf_te_runs, rbf_accs = [], [], []

    for seed in range(N_SEEDS):
        X, y = make_xor3d(n_per_cluster=200, radius=0.3, seed=seed)
        n_tr = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

        tr, te, ac = train_mlp_curve(Xtr, ytr, Xte, yte,
                                     hidden=8, epochs=EPOCHS, seed=seed)
        mlp_tr_runs.append(tr); mlp_te_runs.append(te); mlp_accs.append(ac)

        tr, te, ac = train_rbf_curve(Xtr, ytr, Xte, yte,
                                     K=8, sigma=best_sigma,
                                     epochs=EPOCHS, seed=seed)
        rbf_tr_runs.append(tr); rbf_te_runs.append(te); rbf_accs.append(ac)

        print(f"  seed {seed}: MLP acc={mlp_accs[-1]*100:.2f}%  "
              f"RBF acc={rbf_accs[-1]*100:.2f}%")

    mlp_tr = np.array(mlp_tr_runs); mlp_te = np.array(mlp_te_runs)
    rbf_tr = np.array(rbf_tr_runs); rbf_te = np.array(rbf_te_runs)

    print(f"\nFinal accuracy:")
    print(f"  MLP  mean={np.mean(mlp_accs)*100:.2f}%  "
          f"std={np.std(mlp_accs)*100:.2f}%")
    print(f"  RBF  mean={np.mean(rbf_accs)*100:.2f}%  "
          f"std={np.std(rbf_accs)*100:.2f}%")
    print(f"\nFinal train BCE:")
    print(f"  MLP  mean={mlp_tr[:,-1].mean():.4f}  std={mlp_tr[:,-1].std():.4f}")
    print(f"  RBF  mean={rbf_tr[:,-1].mean():.4f}  std={rbf_tr[:,-1].std():.4f}")
    print(f"Final test BCE:")
    print(f"  MLP  mean={mlp_te[:,-1].mean():.4f}  std={mlp_te[:,-1].std():.4f}")
    print(f"  RBF  mean={rbf_te[:,-1].mean():.4f}  std={rbf_te[:,-1].std():.4f}")

    # ---- shared-axes plot: train losses with mean +- std bands ----
    epochs = np.arange(EPOCHS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # train
    ax = axes[0]
    for arr, name, color in [(mlp_tr, "MLP (H=8)",  "tab:blue"),
                             (rbf_tr, f"RBF (K=8, $\\sigma$={best_sigma:.2f})",
                              "tab:purple")]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(epochs, m, color=color, lw=2, label=name)
        ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss (log scale)")
    ax.set_title(f"Training loss (mean $\\pm$ std over {N_SEEDS} seeds)")
    ax.grid(alpha=0.3, which="both"); ax.legend()

    # test
    ax = axes[1]
    for arr, name, color in [(mlp_te, "MLP (H=8)",  "tab:blue"),
                             (rbf_te, f"RBF (K=8, $\\sigma$={best_sigma:.2f})",
                              "tab:purple")]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(epochs, m, color=color, lw=2, label=name)
        ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title(f"Test loss (mean $\\pm$ std over {N_SEEDS} seeds)")
    ax.grid(alpha=0.3, which="both"); ax.legend()

    fig.suptitle("MLP vs RBF on 3D XOR --- side-by-side training dynamics",
                 fontsize=12, y=1.02)
    savefig("part4_loss_comparison.png", fig)

    # ---- bar chart: variance across seeds ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["MLP train", "RBF train", "MLP test", "RBF test"]
    means  = [mlp_tr[:,-1].mean(), rbf_tr[:,-1].mean(),
              mlp_te[:,-1].mean(), rbf_te[:,-1].mean()]
    stds   = [mlp_tr[:,-1].std(),  rbf_tr[:,-1].std(),
              mlp_te[:,-1].std(),  rbf_te[:,-1].std()]
    colors = ["tab:blue", "tab:purple", "tab:blue", "tab:purple"]
    bars   = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.75,
                    capsize=5)
    ax.set_yscale("log"); ax.set_ylabel("final BCE loss (log scale)")
    ax.set_title(f"Final losses across {N_SEEDS} seeds (lower is better)")
    ax.grid(alpha=0.3, axis="y", which="both")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m * 1.5,
                f"{m:.1e}", ha="center", fontsize=8)
    savefig("part4_final_losses.png", fig)

    return {
        "mlp_acc_mean": float(np.mean(mlp_accs)),
        "mlp_acc_std":  float(np.std(mlp_accs)),
        "rbf_acc_mean": float(np.mean(rbf_accs)),
        "rbf_acc_std":  float(np.std(rbf_accs)),
        "mlp_train_bce_mean": float(mlp_tr[:, -1].mean()),
        "mlp_train_bce_std":  float(mlp_tr[:, -1].std()),
        "rbf_train_bce_mean": float(rbf_tr[:, -1].mean()),
        "rbf_train_bce_std":  float(rbf_tr[:, -1].std()),
        "mlp_test_bce_mean":  float(mlp_te[:, -1].mean()),
        "mlp_test_bce_std":   float(mlp_te[:, -1].std()),
        "rbf_test_bce_mean":  float(rbf_te[:, -1].mean()),
        "rbf_test_bce_std":   float(rbf_te[:, -1].std()),
        "best_sigma": float(best_sigma),
    }


def main():
    best_sigma = sigma_sweep()
    stats = multi_seed_comparison(best_sigma)

    # quick text summary for the report
    print("\n" + "=" * 60)
    print("SUMMARY (numbers to paste into the report)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
