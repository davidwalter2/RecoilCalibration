"""
Compare two sets of calibration inputs (input_mumu.pkl): yields, and the mean and RMS of the recoil components in
ranges of qT for data, the Z sample and the background. Also prints the RMS of the recoil along the generator proxy.

usage: python reproduce/compare_inputs.py <ref/input_mumu.pkl> <new/input_mumu.pkl>
"""

import argparse
import pickle

import hist
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("ref")
parser.add_argument("new")
parser.add_argument("--qtRanges", type=float, nargs="+", default=[0, 5, 10, 20, 30, 40, 60, 80, 100])
args = parser.parse_args()

ref = pickle.load(open(args.ref, "rb"))
new = pickle.load(open(args.new, "rb"))
s = hist.tag.Slicer()


def moments(h, lo, hi, umax=80):
    """Yield, mean and RMS of the recoil in lo <= qT < hi and |u| < umax (the range of the calibration)"""
    h = h[{"qt": s[complex(0, lo):complex(0, hi):hist.sum]}]
    centers = h.axes[0].centers
    vals = h.values()
    sel = np.abs(centers) < umax
    n = vals[sel].sum()
    mean = np.sum(vals[sel] * centers[sel]) / n
    rms = np.sqrt(np.sum(vals[sel] * (centers[sel] - mean) ** 2) / n)
    return n, mean, rms


print("yields: " + ", ".join(f"{k} {ref[k].sum(flow=True).value:.0f} -> {new[k].sum(flow=True).value:.0f}" for k in ("data_para", "z_para", "bkg_para")))
qts = args.qtRanges
for comp in ("para", "perp"):
    print(f"\n=== {comp}: mean / RMS of the recoil (GeV), ref -> new")
    print(f"  {'qT':>9s} {'data mean':>17s} {'data RMS':>17s} {'Z MC mean':>17s} {'Z MC RMS':>17s} {'N data ref/new':>18s} {'N Z ref/new':>13s} {'bkg frac ref/new':>18s}")
    for lo, hi in zip(qts[:-1], qts[1:]):
        r = {k: moments(ref[f"{k}_{comp}"], lo, hi) for k in ("data", "z", "bkg")}
        n = {k: moments(new[f"{k}_{comp}"], lo, hi) for k in ("data", "z", "bkg")}
        print(
            f"  {lo:4.0f}-{hi:<4.0f}"
            f" {r['data'][1]:7.3f} -> {n['data'][1]:7.3f} {r['data'][2]:7.3f} -> {n['data'][2]:7.3f}"
            f" {r['z'][1]:7.3f} -> {n['z'][1]:7.3f} {r['z'][2]:7.3f} -> {n['z'][2]:7.3f}"
            f" {n['data'][0] / r['data'][0]:18.4f} {n['z'][0] / r['z'][0]:13.4f}"
            f" {r['bkg'][0] / r['data'][0]:8.4f} / {n['bkg'][0] / n['data'][0]:.4f}"
        )

print("\n=== RMS of the recoil along the generator boson proxy (GeV), ref -> new")
for key in ("z_para_gen", "z_perp_gen", "z_para_qt_gen"):
    line = f"  {key:14s}"
    for lo, hi in zip(qts[:-1], qts[1:]):
        r = moments(ref[key], lo, hi)[2] if key in ref else np.nan
        n = moments(new[key], lo, hi)[2] if key in new else np.nan
        line += f" {lo:.0f}-{hi:.0f}: {r:6.2f} -> {n:6.2f}"
    print(line)
