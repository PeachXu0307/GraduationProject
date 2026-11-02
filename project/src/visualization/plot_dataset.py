"""Create two inspection plots from the exported IEEE-13 CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(a.folder / "index.csv")
    bus = pd.read_csv(a.folder / "bus_voltages.csv")
    bus = bus.merge(index[["sample_id", "split"]], on="sample_id", validate="many_to_one")
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, part in bus.groupby("split"):
        ax.hist(part.v_mag_pu, bins=35, alpha=.5, label=split)
    ax.set(xlabel="Voltage magnitude (p.u.)", ylabel="Phase-node observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "voltage_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, part in index.groupby("split"):
        ax.scatter(part.load_scale, part.iterations, label=split, alpha=.7)
    ax.set(xlabel="Load scale", ylabel="OpenDSS iterations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "iterations_vs_load.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
