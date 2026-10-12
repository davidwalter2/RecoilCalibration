
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

def bkg_perp():
    proc = "bkg"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_perp']

    recoilRebin = [-50, -40, -30] + list(range(-26, 26, 2)) + [26, 30, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=2, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def bkg_para():
    proc = "bkg"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#minusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['bkg_para']

    recoilRebin = [-50, -40, -30] + list(range(-26, 26, 2)) + [26, 30, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 2)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=2, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()



def z_perp():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp']

    recoilRebin = [-50, -40, -35, -30] + list(range(-25, 25, 1)) + [25, 30, 35, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=1)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    #parms = rqf.fit(withConstraint=False)
    #rqf.fit(withConstraint=True, parms=parms)
    rqf.fit(withConstraint=True, parms=parms, ext=f"{dataDir}/z_perp_postfit.pkl")
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def z_para():
    proc = "z"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#minusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_para']

    recoilRebin = [-50, -40, -35, -30] + list(range(-25, 25, 1)) + [25, 30, 35, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=1)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def data_perp():
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_perp"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_perp']
    bhist_bkg = data_z['bkg_perp']

    recoilRebin = [-50, -40, -35, -30] + list(range(-25, 25, 1)) + [25, 30, 35, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=1)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_perp_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_perp_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()


def data_para():
    proc = "data"
    plotDir = f"{basePlotDir}/{proc}_para"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = f"{met}, U_{{#parallel}}#minusq_{{T}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['data_para']
    bhist_bkg = data_z['bkg_para']
    plotDir_bkg = plotDir.replace("data", "bkg")

    recoilRebin = [-50, -40, -35, -30] + list(range(-25, 25, 1)) + [25, 30, 35, 40, 50]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 120, 5)) + [120]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=1)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_para_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def z_perp_gen():
    proc = "z_gen"
    plotDir = "%s/%s_perp" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = f"{met}, U_{{#perp}}"
    utils.mkdir(plotDir, False)
    bhist = data_z['z_perp_gen']

    recoilRebin = [-110, -100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100, 110]
    bins_for_quantiles = [-100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]
    
    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True, xMin=min(recoilRebin), xMax=max(recoilRebin))
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_perp_postfit.pkl") # ext=f"{dataDir}/z_perp_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()
    #rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_perp_postfit.pkl")


def z_para_gen():
    proc = "z_gen"
    plotDir = "%s/%s_para" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data_z['z_para_gen']

    recoilRebin = [-150, -100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150]
    bins_for_quantiles = [-100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    
    recoilRebin = [-100,  -80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80,  100]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [-110, -100,  -80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]
    bins_for_quantiles = [-80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [-110, -100,  -80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]
    bins_for_quantiles = [-80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True, xMin=-100, xMax=100)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/z_para_postfit.pkl") # , ext=f"{dataDir}/zmumu_para_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=min(recoilRebin), xMax=max(recoilRebin))
    rqf.qparms_postfit()
    #rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_para_postfit.pkl")




if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    met = "DeepMETReso"
    lumiLabel = "200 pb^{#minus1} (13 TeV)"

    with open("data/lowPU_mumu_DeepMETReso/input_z.pkl", "rb") as f:
        data_z = pickle.load(f)

    basePlotDir = "/home/submit/jaeyserm/public_html/recoil/lowPU_mumu_DeepMETReso/calibration/"
    dataDir = "data/lowPU_mumu_DeepMETReso/"
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    #z_para()
    z_perp()

    #bkg_para()
    #bkg_perp()

    #data_para()
    #data_perp()
    
    #z_perp_gen()
    #z_para_gen()
