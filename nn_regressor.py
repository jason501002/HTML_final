"""Neural-network branch of the danceability prediction project.

A cleaned-up, standalone rewrite of the exploratory Colab notebooks used for
the "linear neural network" experiments in REPORT.md. It reproduces the same
pipeline and exposes the experiment axes from the report (optimizer, scaler,
imputer, depth) as command-line flags.

Pipeline
    1. Load train.csv / test.csv from --data-dir.
    2. Keep the 13 numeric columns and add a target-encoded Composer feature.
    3. Split 80/20 into train / validation (seeded).
    4. Fit the imputer and scaler on the training split only; apply to all.
    5. Train an MLP (ReLU + dropout) with Adam or RMSprop and a
       CosineAnnealingWarmRestarts schedule; early-stop on validation MAE.
    6. Round and clip predictions to the label range {0, ..., 9}, write a
       submission CSV (id, Danceability).

Example
    python nn_regressor.py --data-dir data --optimizer adam --scaler minmax \
        --impute mean --hidden 100 --dropout 0.3 --out pred.csv
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TARGET = "Danceability"
NUMERIC = [
    "Energy", "Key", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms",
    "Views", "Likes", "Stream",
]
LABEL_MIN, LABEL_MAX = 0, 9


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def composer_target_encoding(train: pd.DataFrame,
                             other: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean-target encoding of Composer, fitted on `train` only.

    Missing composers are treated as their own category. Unseen categories
    fall back to the global training mean.
    """
    key_train = train["Composer"].astype("object").fillna("__missing__")
    key_other = other["Composer"].astype("object").fillna("__missing__")
    global_mean = train[TARGET].mean()
    enc = train.groupby(key_train)[TARGET].mean()
    return (key_train.map(enc).fillna(global_mean).astype(float),
            key_other.map(enc).fillna(global_mean).astype(float))


def make_imputer(name: str, seed: int):
    if name in ("mean", "median"):
        return SimpleImputer(strategy=name)
    if name == "knn":
        return KNNImputer(n_neighbors=5)
    if name == "iterative":
        return IterativeImputer(random_state=seed)
    raise ValueError(f"unknown imputer: {name}")


class MeanMedianImputer:
    """Fill each column with the average of its mean and median."""

    def fit(self, x: np.ndarray):
        self.fill_ = (np.nanmean(x, axis=0) + np.nanmedian(x, axis=0)) / 2
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = self.fill_[cols]
        return x


SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler, "robust": RobustScaler}


def build_features(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                   test_df: pd.DataFrame, impute: str, scaler: str, seed: int):
    te_train, te_valid = composer_target_encoding(train_df, valid_df)
    _, te_test = composer_target_encoding(train_df, test_df)

    def frame(df, te):
        x = df[NUMERIC].apply(pd.to_numeric, errors="coerce").copy()
        x["Composer_te"] = te.values
        return x.to_numpy(dtype=np.float64)

    x_tr, x_va, x_te = frame(train_df, te_train), frame(valid_df, te_valid), frame(test_df, te_test)

    imp = MeanMedianImputer() if impute == "mean-median" else make_imputer(impute, seed)
    imp.fit(x_tr)
    x_tr, x_va, x_te = imp.transform(x_tr), imp.transform(x_va), imp.transform(x_te)

    sc = SCALERS[scaler]().fit(x_tr)
    return (sc.transform(x_tr).astype(np.float32),
            sc.transform(x_va).astype(np.float32),
            sc.transform(x_te).astype(np.float32))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, prev = [], input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(p=dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def to_labels(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(pred), LABEL_MIN, LABEL_MAX).astype(int)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size):
        out.append(model(xb.to(device)).cpu())
    return torch.cat(out).numpy()


def train(model: nn.Module, x_tr, y_tr, x_va, y_va, args, device: str):
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=args.batch_size, shuffle=True)
    criterion = nn.SmoothL1Loss() if args.loss == "smoothl1" else nn.L1Loss()
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9,
                                  weight_decay=args.weight_decay)
    # Stepped once per batch, as in the original notebooks: short warm restarts
    # early on, progressively longer cosine cycles later.
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=2, T_mult=2, eta_min=1e-5)

    best_mae, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()

        val_mae = float(np.mean(np.abs(to_labels(predict(model, x_va, device)) - y_va)))
        if val_mae < best_mae - 1e-6:
            best_mae, best_state, best_epoch, stale = val_mae, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:5d}  val MAE {val_mae:.4f}  best {best_mae:.4f} @ {best_epoch}")
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    return best_mae, best_epoch, epoch


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="folder containing train.csv and test.csv")
    p.add_argument("--out", type=Path, default=Path("pred.csv"))
    p.add_argument("--impute", choices=["mean", "median", "mean-median", "knn", "iterative"],
                   default="mean")
    p.add_argument("--scaler", choices=sorted(SCALERS), default="minmax")
    p.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    p.add_argument("--hidden", type=int, nargs="+", default=[100],
                   help="hidden layer widths, e.g. --hidden 100 or --hidden 128 64")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--loss", choices=["smoothl1", "l1"], default="smoothl1")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=6000)
    p.add_argument("--patience", type=int, default=100,
                   help="stop after this many epochs without validation improvement")
    p.add_argument("--valid-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_full = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    train_df, valid_df = train_test_split(train_full, test_size=args.valid_ratio,
                                          random_state=args.seed)

    x_tr, x_va, x_te = build_features(train_df, valid_df, test_df,
                                      args.impute, args.scaler, args.seed)
    y_tr = train_df[TARGET].to_numpy(np.float32)
    y_va = valid_df[TARGET].to_numpy(np.float32)
    print(f"train {x_tr.shape}  valid {x_va.shape}  test {x_te.shape}  device {device}")

    model = MLP(x_tr.shape[1], args.hidden, args.dropout).to(device)
    best_mae, best_epoch, last_epoch = train(model, x_tr, y_tr, x_va, y_va, args, device)
    print(f"best validation MAE {best_mae:.4f} at epoch {best_epoch} (stopped at {last_epoch})")

    preds = to_labels(predict(model, x_te, device))
    pd.DataFrame({"id": test_df["id"], TARGET: preds}).to_csv(args.out, index=False)
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
