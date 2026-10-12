
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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    met = "DeepMETPVRobust"

    basePlotDir = "/home/submit/jaeyserm/public_html/recoil/highPU_DeepMETPVRobust/calibration/"
    dataDir = "data/highPU_DeepMETPVRobust/"
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    with open(f"{dataDir}/input_mumu_old.pkl", "rb") as f:
        data_old = pickle.load(f)

    with open(f"{dataDir}/input_mumu.pkl", "rb") as f:
        data_new = pickle.load(f)
        
        
    bhist_old = data_old['z_perp'].values()
    bhist_new = data_new['z_perp'].values()
    
    k = bhist_new
    #print(bhist_old/bhist_new)
    k_inf = np.isinf(k)
    #print(k_inf)
    
    print(k[k < 0.0])
    for i,outer in enumerate(k_inf):
        #print(i, outer)
        for j,inner in enumerate(outer):
            if inner:
                print(i, j, inner)

    