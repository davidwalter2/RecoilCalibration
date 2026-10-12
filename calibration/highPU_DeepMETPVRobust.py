
import argparse
import logging
import os
import pickle

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = 6001

import utils
import quantileFitter as qf
import exportModel

# configuration shared by all fits (only min/max of the recoil binning enter the fit)
RECOIL_RANGE = (-80, 80)
BINS_FOR_QUANTILES = [-80, -70, -60, -50, -45, -40, -38, -36, -34, -32] + list(range(-30, 30, 2)) + [32, 34, 36, 38, 40, 45, 50, 60, 70, 80]
QT_REBIN = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 80, 5)) + [80, 90, 100]
KNOTS = {
    "para": [1, 4, 8, 15, 30, 60],
    "perp": [1, 4, 8, 20, 60],
}
# variation of the qT parametrisation above the last nominal knot: an extra knot at 90 GeV,
# extrapolating linearly beyond it instead of beyond 60 GeV
KNOTS_EXT = {
    "para": [1, 4, 8, 15, 30, 60, 90],
    "perp": [1, 4, 8, 20, 60, 90],
}
AXES = {"para": "recoil_para_qt", "perp": "recoil_perp"}
LABELS = {
    "z": "DY #rightarrow #mu^{+}#mu^{#minus}",
    "bkg": "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}",
    "data": "Data #rightarrow #mu^{+}#mu^{#minus}",
}
MET_LABELS = {"para": "U_{#parallel}#plusq_{T}", "perp": "U_{#perp}"}


def make_fitter(proc, comp, suffix="", knots=None):
    knots = knots if knots is not None else KNOTS[comp]
    plotDir = f"{basePlotDir}/{proc}_{comp}{suffix}"
    utils.mkdir(plotDir, False)
    rqf = qf.QuantileFitter(proc, met, LABELS[proc], f"{met}, {MET_LABELS[comp]}", dataDir, plotDir, comp, lumiLabel, logging, suffix=suffix)
    rqf.setHistConfig(data_z[f"{proc}_{comp}"], AXES[comp], min(QT_REBIN), max(QT_REBIN), *RECOIL_RANGE, qTRebin=QT_REBIN, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=BINS_FOR_QUANTILES)
    rqf.setSplineConfig(knots, extrpl=[None, knots[-1]])
    return rqf


def run_fit(rqf, start="", polish=False):
    if polish:
        # continue the minimization of an existing constrained fit
        rqf.fit(withConstraint=True, ext=rqf.postfit_pkl, polish_only=True)
    else:
        rqf.qparms_prefit()
        rqf.plot_refit_quantiles(prefit=True)
        if trust_exact:
            # minimize the constrained likelihood directly with trust-exact
            rqf.fit(withConstraint=True, ext=start, polish_only=True)
        else:
            # the unconstrained fit has one null direction per qT knot (normalisation of the quantiles),
            # the constrained refit removes them so that the status and covariance are meaningful
            parms = rqf.fit(withConstraint=False, ext=start)
            rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def fit_z(comp, ext=False, polish=False):
    suffix = "_ext" if ext else ""
    run_fit(make_fitter("z", comp, suffix=suffix, knots=KNOTS_EXT[comp] if ext else None), polish=polish)


def fit_bkg(comp, polish=False):
    run_fit(make_fitter("bkg", comp), polish=polish)


def fit_data(comp, bkg_sf=1.0, suffix="", ext=False, polish=False):
    """Fit data as signal + background, the background shape from the bkg fit scaled by bkg_sf"""
    suffix = suffix + ("_ext" if ext else "")
    rqf = make_fitter("data", comp, suffix=suffix, knots=KNOTS_EXT[comp] if ext else None)
    rqf.setBackground(f"{dataDir}/bkg_{comp}_postfit.pkl", data_z[f"bkg_{comp}"], sf=bkg_sf)
    # start from the MC (signal) fit with the same parametrisation
    run_fit(rqf, start=f"{dataDir}/z_{comp}{'_ext' if ext else ''}_postfit.pkl", polish=polish)


def export(out, bkg_var=True, ext=True):
    exp = exportModel.Export()
    exp.set_limits(500, *RECOIL_RANGE)
    exp.add_base_transform(f"{dataDir}/z_para_postfit.pkl", f"{dataDir}/data_para_postfit.pkl", f"{dataDir}/z_perp_postfit.pkl", f"{dataDir}/data_perp_postfit.pkl")
    for comp in ("para", "perp"):
        para = comp == "para"
        data_nom, mc_nom = f"{dataDir}/data_{comp}_postfit.pkl", f"{dataDir}/z_{comp}_postfit.pkl"
        if bkg_var:
            # background normalisation in the data fit varied by +-20%
            for var in ("Up", "Down"):
                exp.add_systematic(f"bkg_{comp}{var}", data_nom, f"{dataDir}/data_{comp}_bkg{var}_postfit.pkl", para=para)
        if ext:
            # alternative qT parametrisation of both the MC and the data fit, with an extra knot at 90 GeV
            exp.add_systematic(f"qtParam_{comp}", data_nom, f"{dataDir}/data_{comp}_ext_postfit.pkl", source=mc_nom, source_alt=f"{dataDir}/z_{comp}_ext_postfit.pkl", para=para)
        exp.add_pdf(f"data_{comp}", data_nom, para=para)
        exp.add_pdf(f"mc_{comp}", mc_nom, para=para)
    exp.export(out)


STEPS = {
    "z_para": (fit_z, dict(comp="para")),
    "z_perp": (fit_z, dict(comp="perp")),
    "bkg_para": (fit_bkg, dict(comp="para")),
    "bkg_perp": (fit_bkg, dict(comp="perp")),
    "data_para": (fit_data, dict(comp="para")),
    "data_perp": (fit_data, dict(comp="perp")),
    "data_para_bkgUp": (fit_data, dict(comp="para", bkg_sf=1.2, suffix="_bkgUp")),
    "data_perp_bkgUp": (fit_data, dict(comp="perp", bkg_sf=1.2, suffix="_bkgUp")),
    "data_para_bkgDown": (fit_data, dict(comp="para", bkg_sf=0.8, suffix="_bkgDown")),
    "data_perp_bkgDown": (fit_data, dict(comp="perp", bkg_sf=0.8, suffix="_bkgDown")),
    "z_para_ext": (fit_z, dict(comp="para", ext=True)),
    "z_perp_ext": (fit_z, dict(comp="perp", ext=True)),
    "data_para_ext": (fit_data, dict(comp="para", ext=True)),
    "data_perp_ext": (fit_data, dict(comp="perp", ext=True)),
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", default="data/highPU_DeepMETPVRobust_v2/", help="Directory with input_mumu.pkl, fit results are written here (the May 2024 fits are in data/highPU_DeepMETPVRobust/)")
    parser.add_argument("--plotDir", default="/home/submit/david_w/public_html/WMass/261011_recoilCalibration/highPU_DeepMETPVRobust/", help="Output directory for the plots")
    parser.add_argument("--steps", nargs="+", default=list(STEPS.keys()) + ["export"], choices=list(STEPS.keys()) + ["export"], help="Fits to run, in this order")
    parser.add_argument("--output", default="model_mc_data.tflite", help="Name of the exported model in dataDir")
    parser.add_argument("--polish", action="store_true", help="Continue the minimization of the existing constrained fits of the steps, instead of fitting from scratch")
    parser.add_argument("--trustExact", action="store_true", help="Minimize the constrained likelihood directly with trust-exact, instead of the unconstrained and constrained fits with trust-krylov")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    met = "DeepMETPVRobust"
    dataDir = args.dataDir
    basePlotDir = args.plotDir
    trust_exact = args.trustExact
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    with open(f"{dataDir}/input_mumu.pkl", "rb") as f:
        data_z = pickle.load(f)
    lumiLabel = data_z['lumi_header']

    for step in args.steps:
        logging.info(f"Run step {step}")
        if step == "export":
            export(os.path.join(dataDir, args.output))
        else:
            func, kwargs = STEPS[step]
            func(**kwargs, polish=args.polish)
