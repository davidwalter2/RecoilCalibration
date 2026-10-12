
import sys,array,math,os
import numpy as np
import scipy
import copy
import pickle
import logging
import narf

import ROOT
ROOT.gROOT.SetBatch(True)



with open("data/highPU_RawPFMET/input_mumu.pkl", "rb") as f:
    data_z = pickle.load(f)


    data_para = data_z['data_para']
    data_para = data_para.project("recoil_para_qt")
    print(data_para)
