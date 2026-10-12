
import sys
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import pickle
import utils
import plotUtils as plotter
import tensorflow as tf
import hist
import boost_histogram as bh
import baseFunctions as rfb
import narf
import wums.fitutils
import boostHistHelpers as bhh
from scipy.interpolate import CubicSpline

np.set_printoptions(threshold=sys.maxsize)


class ResponseCorr:

    def __init__(self, proc, met, procLabel, metLabel, baseDir, lumiLabel):
        self.proc = proc
        self.met = met
        self.procLabel = procLabel
        self.metLabel = metLabel
        self.baseDir = baseDir
        self.lumiLabel = lumiLabel
        self.dtype = tf.float64
        self.itype = tf.int64


    def setHist(self, bhist, qTRebin):
        self.qTRebin = qTRebin
        s = hist.tag.Slicer()
        self.bhist = bhist[{"qTbinned": s[complex(0,min(qTRebin)):complex(0,max(qTRebin))]}]
        self.bhist = bhh.rebinHist(self.bhist, "qTbinned", qTRebin)
        self.bins_qt = self.bhist.axes[0].edges
        self.bins_recoil = self.bhist.axes[1].edges
        self.centers_qt = self.bhist.axes[0].centers
        self.centers_recoil = self.bhist.axes[1].centers

    def rebin(self):
        s = hist.tag.Slicer()
        self.bhist = self.bhist[{"qTbinned": s[complex(0,self.qTMin):complex(0,self.qTMax)]}]
        bhist = bhh.rebinHist(bhist, "qTbinned", qTRebin)
        return bhist

        
    def correct(self, knots, extrpl=[None, None]):
        qTvals, respVals, respValsErr = [], [], []

        s = hist.tag.Slicer()
        for qTbin in range(0, len(self.bins_qt)-1):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]

            h = self.bhist[{"qTbinned": s[complex(0,qTlow):complex(0,qThigh)]}]
            h = h[{"qTbinned": s[0:hist.overflow:hist.sum]}]
            hist_root = narf.hist_to_root(h)

            qTvals.append(qT)
            respVals.append(hist_root.GetMean())
            respValsErr.append(hist_root.GetMeanError())

        gr = utils.makeTGraph(qTvals, respVals, y_err=respValsErr)
        
        edges_qt_tf = tf.constant(self.centers_qt, dtype=self.dtype)
        respVals_tf = tf.constant(respVals, dtype=self.dtype)
        knots_tf = tf.constant(knots, dtype=self.dtype)
        edges_qt_tf = edges_qt_tf[None, :]
        respVals_tf = respVals_tf[None, :]
        knots_tf = knots_tf[None, :]
        fit_vals = wums.fitutils.cubic_spline_interpolate(edges_qt_tf, respVals_tf, knots_tf, axis=-1, extrpl=extrpl)
        fit_vals_plt = wums.fitutils.cubic_spline_interpolate(knots_tf, fit_vals, edges_qt_tf, axis=-1, extrpl=extrpl)
        fit_vals_plt = fit_vals_plt.numpy()[0]
        print(fit_vals_plt)
        print(self.centers_qt)
        yRatio = 1.06
        cfgPlot = {

            'logy'              : False,
            'logx'              : False,

            'xmin'              : 0,
            'xmax'              : max(self.qTRebin),
            'ymin'              : -5,
            'ymax'              : 20,

            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "u_{#parallel} (GeV)",
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }


        g_fit = utils.makeTGraph(qTvals, fit_vals_plt, style="line")
        g_ratio = utils.makeTGraph(qTvals, np.divide(np.array(respVals), fit_vals_plt))

        plotter.cfg = cfgPlot
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()

        ## TOP PAD ##
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        padT.SetTickx()
        padT.SetTicky()
        dummyT.Draw("HIST")
        gr.Draw("PE SAME")
        g_fit.Draw("L SAME")  
        padT.RedrawAxis()
        padT.RedrawAxis("G")
        plotter.auxRatio()
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.04)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.2, 0.85, self.procLabel)
        latex.DrawLatex(0.2, 0.80, self.metLabel)



        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        padB.SetGrid()
        padB.SetTickx()
        padB.SetTicky()
        dummyB.Draw("HIST")

        dummyL.Draw("SAME")
        g_ratio.Draw("PE0 SAME")

        padB.RedrawAxis()
        padB.RedrawAxis("G")
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{self.baseDir}/response.png")
        canvas.SaveAs(f"{self.baseDir}/response.pdf")

        fOut = f"data/{self.met}_respCorr.pkl"
        out = {}
        out['fit_vals'] = fit_vals
        out['knots_tf'] = knots_tf
        out['edges_qt_tf'] = edges_qt_tf
        out['extrpl'] = extrpl
        with open(fOut, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
