# HW5 — My First Feedforward Neural Network

Compact PyTorch solution for the three parts of the homework.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Each script is independent and saves its figures into `report/figures/`.

```bash
python part1_bce_loss.py     # ~5 s   — BCE loss surface
python part2_xor3d.py        # ~5 min — 3D XOR + MLP + radius/hidden sweep
python part3_rbf.py          # ~30 s  — RBF network on the same problem
```

After running, all figures land in `report/figures/`. Compile the report:

```bash
cd report
pdflatex report.tex
```

## Files

| File              | Part | What it does                                                                |
| ----------------- | ---- | --------------------------------------------------------------------------- |
| `part1_bce_loss.py` | 1   | Plots BCE(y, ŷ) as a 3D surface and 2D heatmap                              |
| `part2_xor3d.py`    | 2   | Generates a 3D checkerboard, trains a 1-hidden-layer MLP, sweeps radius × hidden size |
| `part3_rbf.py`      | 3   | RBF network (k-means centers + linear output) on the same problem          |
| `utils.py`          | —   | Shared: data generation, plotting helpers                                   |
| `report/report.tex` | —   | LaTeX report with `[PLACEHOLDER]` gaps for figures and screenshots          |
