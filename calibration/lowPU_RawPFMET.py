
import sys,array,math,os
import numpy as np
import scipy
import copy
import pickle
import logging

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = 6001

import utils
import quantileFitter as qf
import responseCorr as rc
import exportModel

def bkg_perp():
    proc = "bkg"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "TTbar+EWK #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_perp']

    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 105, 5))

    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50]
    recoilRebin =  list(range(-50, 55, 5))
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 105, 5))

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def bkg_para():
    proc = "bkg"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "TTbar+EWK #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_para']

    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    #recoilRebin = list(range(-60, 65, 5)) 
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 105, 5))

    recoilRebin = [-60, -50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 60]
    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50]
    recoilRebin =  list(range(-50, 65, 5))
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 105, 5))

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

def z_perp():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "DY #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp']

    recoilRebin = [-100, -60, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 60, 100]
    recoilRebin = [-100, -60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60, 100]
    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 20, 1)) + list(range(20, 40, 2)) + [40, 45, 50, 60, 70, 100]

    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + [30, 33, 36, 40, 45, 50, 60, 70, 100]


    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def z_para():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "DY #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_para']

    recoilRebin = [-50, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 50, 60]
    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 60]
    recoilRebin = [-100, -60, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 60, 100]
    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 20, 1)) + list(range(20, 40, 2)) + [40, 45, 50, 60, 70, 100]



    recoilRebin = [-60, -50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 60]
    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 60]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + [30, 33, 36, 40, 45, 50, 60, 70, 100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

def data_perp(bkg_unc=False):
    suffix = "_bkg" if bkg_unc else ""
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_perp{suffix}"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "DY #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_perp']
    bhist_bkg = data_z['bkg_perp']
    plotDir_bkg = plotDir.replace("data", "bkg")

    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 20, 1)) + list(range(20, 40, 2)) + [40, 45, 50, 60, 70, 100]


    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + [30, 33, 36, 40, 45, 50, 60, 70, 100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging, suffix=suffix)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_perp_postfit.pkl", bhist_bkg, sf=(1.2 if bkg_unc else 1.0))

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_perp_postfit.pkl") # , ext=f"{dataDir}/z_perp_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def data_para(bkg_unc=False):
    suffix = "_bkg" if bkg_unc else ""
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_para{suffix}"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}" if flavor=="mumu" else "DY #rightarrow e^{+}e^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_para']
    bhist_bkg = data_z['bkg_para']
    plotDir_bkg = plotDir.replace("data", "bkg")

    recoilRebin = [-60, -50, -45] + list(range(-40, 40, 2)) + [40, 45, 50, 60]
    recoilRebin = [-50, -45] + list(range(-40, 40, 2)) + [40, 45, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 20, 1)) + list(range(20, 40, 2)) + [40, 45, 50, 60, 70, 100]

    recoilRebin = [-50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 60]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + [30, 33, 36, 40, 45, 50, 60, 70, 100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging, suffix=suffix)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([1, 4, 8, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_para_postfit.pkl", bhist_bkg, sf=(1.2 if bkg_unc else 1.0))

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl") # , ext=f"{dataDir}/z_para_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def z_perp_gen():
    proc = "z_gen"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp_gen']

    recoilRebin = [-80, -70, -65, -60, -55, -50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False) # , ext=f"{dataDir}/z_perp_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()
    #rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_perp_postfit.pkl")


def z_para_gen():
    proc = "z_gen"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = f"{met}, U_{{#parallel}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_para_gen']

    recoilRebin = [-80, -70, -65, -60, -55, -50, -45, -40, -35] + list(range(-30, 30, 2)) + [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl") # , ext=f"{dataDir}/z_para_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()
    #rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_para_postfit.pkl")


def export():

    exp = exportModel.Export()
    exp.set_limits(100, -50, 50) # limits on ut
    exp.set_limits_para(-50, 50) # override limits for para
    exp.set_limits_perp(-50, 50) # override limits for perp

    exp.add_base_transform(f"{dataDir}/z_para_postfit.pkl", f"{dataDir}/data_para_postfit.pkl", f"{dataDir}/z_perp_postfit.pkl", f"{dataDir}/data_perp_postfit.pkl")
    exp.add_systematic("bkg_para", f"{dataDir}/data_para_postfit.pkl", f"{dataDir}/data_para_bkg_postfit.pkl")
    exp.add_systematic("bkg_perp", f"{dataDir}/data_perp_postfit.pkl", f"{dataDir}/data_perp_bkg_postfit.pkl")
    exp.add_pdf("data_para", f"{dataDir}/data_para_postfit.pkl")
    exp.add_pdf("data_perp", f"{dataDir}/data_perp_postfit.pkl")
    exp.add_pdf("mc_para", f"{dataDir}/z_para_postfit.pkl")
    exp.add_pdf("mc_perp", f"{dataDir}/z_perp_postfit.pkl")
    #exp.add_pdf("gen_para", f"{dataDir}/z_gen_para_postfit.pkl")
    #exp.add_pdf("gen_perp", f"{dataDir}/z_gen_perp_postfit.pkl")
    exp.export(f"{dataDir}/model_mc_data.tflite")


if __name__ == "__main__":

    flavor = "ee"
    logging.basicConfig(level=logging.INFO)
    met = "RawPFMET"

    basePlotDir = f"/home/submit/jaeyserm/public_html/recoil/lowPU_RawPFMET/calibration_{flavor}/"
    dataDir = f"data/lowPU_RawPFMET_{flavor}/"
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    with open(f"{dataDir}/input_{flavor}.pkl", "rb") as f:
        data_z = pickle.load(f)
    lumiLabel = data_z['lumi_header']

    logging.basicConfig(level=logging.INFO)


    #z_para()
    #z_perp()

    #bkg_para()
    #bkg_perp()

    #data_para(bkg_unc=False)
    #data_perp(bkg_unc=False)

    #data_para(bkg_unc=True)
    #data_perp(bkg_unc=True)

    #z_perp_gen()
    #z_para_gen()

    export()

