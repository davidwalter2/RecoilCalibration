"""Compare the parameters of two sets of fit results (pkl files of the quantile fits)"""

import argparse
import pickle

import numpy as np
import tensorflow as tf

import wums.fitutils

parser = argparse.ArgumentParser()
parser.add_argument("ref", help="Reference directory")
parser.add_argument("new", help="Directory to compare")
parser.add_argument("--fits", nargs="+", default=["z_para", "z_perp", "bkg_para", "bkg_perp", "data_para", "data_perp"])
args = parser.parse_args()


def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def quantiles(res, qts):
    """Positions of the quantiles in u at the given qT, independent of the null directions of unconstrained fits"""
    parms_2d = tf.transpose(tf.reshape(tf.constant(res["x"], tf.float64), (-1, res["order"])))
    qparms = wums.fitutils.cubic_spline_interpolate(res["knots_qt"], parms_2d, tf.constant(np.array(qts)[:, None], tf.float64), axis=0, extrpl=res["extrpl"])
    return wums.fitutils.qparms_to_quantiles(qparms, x_low=res["recoilMin"], x_high=res["recoilMax"], axis=1).numpy()



qts = [2, 5, 10, 20, 40, 60, 80, 100]
print(f"{'fit':12s} {'status ref/new':>15s} {'edm ref':>9s} {'edm new':>9s} {'loss ref - new':>14s} {'max|dx|/sigma':>14s} {'chi2(dx)/n':>11s}   max |shift of the quantiles| (GeV) at qT = {qts}")
for fit in args.fits:
    try:
        r, n = load(f"{args.ref}/{fit}_postfit.pkl"), load(f"{args.new}/{fit}_postfit.pkl")
    except FileNotFoundError as e:
        print(f"{fit:12s} missing: {e.filename}")
        continue
    dx = n["x"] - r["x"]
    # uncertainties of the new fit; constrained fits have a positive definite hessian
    cov = n["cov"]
    sig = np.sqrt(np.abs(np.diag(cov)))
    try:
        chi2 = dx @ np.linalg.solve(cov, dx) / len(dx)
    except np.linalg.LinAlgError:
        chi2 = np.nan
    dq = np.max(np.abs(quantiles(n, qts) - quantiles(r, qts)), axis=1)
    print(f"{fit:12s} {r['status']:>7d}/{n['status']:<7d} {r['edmval']:9.2e} {n['edmval']:9.2e} {float(r['loss_val'] - n['loss_val']):14.4f} {np.max(np.abs(dx) / sig):14.3f} {chi2:11.3e}   " + " ".join(f"{x:.3f}" for x in dq))
