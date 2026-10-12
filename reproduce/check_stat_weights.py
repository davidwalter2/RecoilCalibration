"""
Check the weights for the statistical uncertainties of the calibration fits (exportModel.add_transform_single):
1) the weights of the implementation (gradients of the log pdfs) against finite differences
2) closure: MC events sampled at fixed qT and transformed with the fit parameters shifted along an eigenvector
   vs. the nominal transform reweighted with the weights (new and the old, pre-2026 formula)
"""

import argparse
import pickle

import numpy as np
import tensorflow as tf

import exportModel
import wums.fitutils

parser = argparse.ArgumentParser()
parser.add_argument("dataDir", help="Directory with the {z,data}_{para,perp}_postfit.pkl fit results")
parser.add_argument("--comp", default="para", choices=["para", "perp"])
parser.add_argument("--qt", type=float, default=10.0, help="qT at which the closure is tested")
parser.add_argument("--nevents", type=int, default=4000000)
parser.add_argument("--eig", type=int, nargs="+", default=[0, 1, 4, 19, 99], help="Eigenvectors to test, sorted by decreasing uncertainty")
parser.add_argument("--scale", type=float, default=300.0, help="Shift of the parameters in units of the uncertainty")
args = parser.parse_args()

dtype = tf.float64
exp = exportModel.Export()
exp.set_limits(500, -80, 80)
source, target = f"{args.dataDir}/z_{args.comp}_postfit.pkl", f"{args.dataDir}/data_{args.comp}_postfit.pkl"


def load(fname):
    with open(fname, "rb") as f:
        res = pickle.load(f)
    vscaled = exp.get_scaled_eigenvectors(res["hess"], num_null=0 if res["withConstraint"] else res["order"])
    return res, vscaled


def quants_at(res, parms, qt):
    parms_2d = tf.transpose(tf.reshape(tf.constant(parms, dtype), (-1, res["order"])))
    qparms = wums.fitutils.cubic_spline_interpolate(res["knots_qt"], parms_2d, tf.constant([[qt]], dtype), axis=0, extrpl=res["extrpl"])
    return wums.fitutils.qparms_to_quantiles(qparms, x_low=tf.cast(res["recoilMin"], dtype), x_high=tf.cast(res["recoilMax"], dtype), axis=1)


def cdf(res, parms, qt, u):
    return wums.fitutils.pchip_interpolate(quants_at(res, parms, qt), res["quant_cdfvals_tf"], tf.constant(u[None, :], dtype), axis=1).numpy()[0]


def invcdf(res, parms, qt, q):
    return wums.fitutils.pchip_interpolate(res["quant_cdfvals_tf"], quants_at(res, parms, qt), tf.constant(q[None, :], dtype), axis=1).numpy()[0]


def pdf(res, parms, qt, u):
    ut = tf.constant(u[None, :], dtype)
    quants = quants_at(res, parms, qt)
    with tf.GradientTape() as t:
        t.watch(ut)
        c = wums.fitutils.pchip_interpolate(quants, res["quant_cdfvals_tf"], ut, axis=1)
    return t.gradient(c, ut).numpy()[0]


res_s, v_s = load(source)
res_t, v_t = load(target)
th_s, th_t = res_s["x"], res_t["x"]
n_s = v_s.shape[1]

# 1) implementation vs finite differences of log p at fixed u (MC) and fixed u' (data)
transform, nstat = exp.add_transform_single(source, target, exp.ut_min, exp.ut_max)
assert nstat == n_s + v_t.shape[1]
h = 1e-3
print(f"1) implementation vs finite differences ({args.comp}, qT={args.qt})")
maxdiff = 0
for u in [-60.0, -30.0, -10.0, -2.0, 0.0, 3.0, 12.0, 25.0, 45.0, 70.0]:
    up, w = transform(tf.constant(args.qt, dtype), tf.constant(u, dtype))
    up, w = up.numpy(), w.numpy()
    for k in args.eig:
        fd_mc = -(np.log(pdf(res_s, th_s + h * v_s[:, k], args.qt, np.array([u]))) - np.log(pdf(res_s, th_s - h * v_s[:, k], args.qt, np.array([u])))) / (2 * h)
        fd_data = (np.log(pdf(res_t, th_t + h * v_t[:, k], args.qt, np.array([up]))) - np.log(pdf(res_t, th_t - h * v_t[:, k], args.qt, np.array([up])))) / (2 * h)
        k_s, k_t = k % n_s, n_s + k % v_t.shape[1]
        maxdiff = max(maxdiff, abs(np.log(w[k_s]) - fd_mc[0]), abs(np.log(w[k_t]) - fd_data[0]))
    print(f"  u={u:6.1f} u'={up:7.2f}  log w (MC, data) of eigenvector {args.eig[0]}: {np.log(w[args.eig[0] % n_s]):+.2e} {np.log(w[n_s + args.eig[0] % v_t.shape[1]]):+.2e}")
print(f"  max |log w - finite difference of log p| = {maxdiff:.2e}")

# 2) closure with MC events sampled from the MC fit at fixed qT
rng = np.random.default_rng(1)
u = invcdf(res_s, th_s, args.qt, rng.uniform(size=args.nevents))
inrange = (u > -80) & (u < 80)
u = u[inrange]
up = invcdf(res_t, th_t, args.qt, cdf(res_s, th_s, args.qt, u))
edges = np.array([-80, -50, -40, -30, -20, -15, -10, -6, -3, 0, 3, 6, 10, 15, 20, 30, 40, 50, 80], dtype=float)
hnom = np.histogram(up, edges)[0].astype(float)
logp_t, logp_s = np.log(pdf(res_t, th_t, args.qt, up)), np.log(pdf(res_s, th_s, args.qt, u))

print(f"\n2) closure: (ratio - 1) x 1000 of the transform with shifted parameters to the nominal, {args.scale} sigma, per u' bin")
print("   " + " ".join(f"{0.5 * (a + b):6.1f}" for a, b in zip(edges[:-1], edges[1:])))
for k in args.eig[:3]:
    for which in ("data", "mc"):
        eps = args.scale
        if which == "data":
            th_var = th_t + eps * v_t[:, k]
            up_var = invcdf(res_t, th_var, args.qt, cdf(res_s, th_s, args.qt, u))
            w_new = np.exp(np.log(pdf(res_t, th_var, args.qt, up)) - logp_t)
            # old: d log p_target(F_target^-1(q)) at fixed quantile q
            w_old = np.exp(np.log(pdf(res_t, th_var, args.qt, up_var)) - logp_t)
        else:
            th_var = th_s + eps * v_s[:, k]
            up_var = invcdf(res_t, th_t, args.qt, cdf(res_s, th_var, args.qt, u))
            w_new = np.exp(logp_s - np.log(pdf(res_s, th_var, args.qt, u)))
            # old: d log p_target(u'(theta_source)), the target density at the shifted point
            w_old = np.exp(np.log(pdf(res_t, th_t, args.qt, up_var)) - logp_t)
        hvar = np.histogram(up_var, edges)[0]
        hnew = np.histogram(up, edges, weights=w_new)[0]
        hold = np.histogram(up, edges, weights=w_old)[0]
        print(f"{which:4s} eig {k}")
        print("  shifted  " + " ".join(f"{x:6.2f}" for x in 1e3 * (hvar / hnom - 1)))
        print("  new      " + " ".join(f"{x:6.2f}" for x in 1e3 * (hnew / hnom - 1)))
        print("  old      " + " ".join(f"{x:6.2f}" for x in 1e3 * (hold / hnom - 1)))
