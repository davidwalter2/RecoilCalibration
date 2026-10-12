"""
Data/MC weights in the reconstructed dilepton pt from the calibration inputs: w = (data - background) / Z MC, normalised
to keep the Z yield. Written in the format of the vptrw_mc_data json files of WRemnants (vpt_bins, weights), used for
the qT-reweighted validation histograms of the W-like histmaker (--recoilHists --recoilQtReweighting).

usage: python calibration/make_qt_reweighting.py -i <dataDir>/input_mumu.pkl -o <dataDir>/vptrw_mc_data_mumu.json
"""

import argparse
import json
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Calibration inputs (input_mumu.pkl)")
parser.add_argument("-o", "--output", required=True, help="Output json")
parser.add_argument("--bins", type=float, nargs="+", default=list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100, 200], help="Bins in qT")
args = parser.parse_args()

inputs = pickle.load(open(args.input, "rb"))
h_data, h_z, h_bkg = (inputs[f"{p}_para"].project("qt") for p in ("data", "z", "bkg"))
edges = h_data.axes[0].edges
bins = np.array(args.bins)
idx = np.array([np.argmin(np.abs(edges - b)) for b in bins])
if np.max(np.abs(edges[idx] - bins)) > 1e-6:
    raise ValueError(f"The bins {bins} have to be edges of the input histograms")


def rebin(h):
    vals, vars = h.values(), h.variances()
    return np.array([vals[a:b].sum() for a, b in zip(idx[:-1], idx[1:])]), np.array([vars[a:b].sum() for a, b in zip(idx[:-1], idx[1:])])


(data, data_var), (z, z_var), (bkg, _) = rebin(h_data), rebin(h_z), rebin(h_bkg)
weights = np.where(z > 0, (data - bkg) / np.where(z > 0, z, 1), 1.0)
norm = z.sum() / np.sum(z * weights)
weights *= norm
errors = np.sqrt(data_var / np.where(z > 0, z, 1) ** 2 + weights**2 * z_var / np.where(z > 0, z, 1) ** 2) * norm

print(f"normalisation {norm:.4f}")
for lo, hi, w, e, n in zip(bins[:-1], bins[1:], weights, errors, data):
    print(f"  qT {lo:5.0f}-{hi:<5.0f} weight {w:.4f} +- {e:.4f}  ({n:.0f} data events)")

with open(args.output, "w") as f:
    json.dump(
        {
            "vpt_bins": [float(x) for x in bins],
            "weights": [float(x) for x in weights],
            "weights_err": [float(x) for x in errors],
            "description": "(data - background) / Z MC in the reconstructed dilepton pt, normalised to the Z yield",
            "input": args.input,
        },
        f,
        indent=1,
    )
print(f"Written {args.output}")
