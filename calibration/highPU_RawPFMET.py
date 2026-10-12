
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
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_perp']

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def bkg_para():
    proc = "bkg"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_para']

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def z_perp():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp']

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def z_para():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_para']

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def data_perp(bkg_unc=False):
    suffix = "_bkg" if bkg_unc else ""
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_perp{suffix}"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_perp']
    bhist_bkg = data_z['bkg_perp']
    plotDir_bkg = plotDir.replace("data", "bkg")

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging, suffix=suffix)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_perp_postfit.pkl", bhist_bkg, sf=(1.2 if bkg_unc else 1.0))

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False) # , ext=f"{dataDir}/z_perp_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def data_para(bkg_unc=False):
    suffix = "_bkg" if bkg_unc else ""
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_para{suffix}"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#plusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_para']
    bhist_bkg = data_z['bkg_para']
    plotDir_bkg = plotDir.replace("data", "bkg")

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging, suffix=suffix)
    rqf.setHistConfig(bhist, "recoil_para_qt", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_para_postfit.pkl", bhist_bkg, sf=(1.2 if bkg_unc else 1.0))

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def z_perp_gen():
    proc = "z_gen"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp_gen']

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_perp_postfit.pkl")
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

    recoilRebin = [-100, -95, -90, -85] + list(range(-80, 80, 2)) + [80, 85, 90, 95, 100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()
    #rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_para_postfit.pkl")

def export():

    exp = exportModel.Export()
    exp.set_limits(200, -100, 100) # limits on ut
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
    #exp.test(f"{dataDir}/model_mc_data.tflite")



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    met = "RawPFTMET"

    basePlotDir = "/home/submit/jaeyserm/public_html/recoil/highPU_RawPFMET/calibration/"
    dataDir = "data/highPU_RawPFMET/"
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    with open(f"{dataDir}/input_mumu.pkl", "rb") as f:
        data_z = pickle.load(f)
    lumiLabel = data_z['lumi_header']

    z_para()
    z_perp()

    bkg_para()
    bkg_perp()

    data_para(bkg_unc=False)
    data_perp(bkg_unc=False)

    #data_para(bkg_unc=True)
    #data_perp(bkg_unc=True)
    
    #z_perp_gen()
    #z_para_gen()

    export()