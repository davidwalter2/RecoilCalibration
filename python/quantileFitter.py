
import sys,array,math,os,copy,decimal,random
import numpy as np
import ctypes
import json
import time
from scipy.interpolate import CubicSpline

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import utils
import plotUtils as plotter
import pickle
import tensorflow as tf
import hist
import boost_histogram as bh
import baseFunctions as rfb
import narf
import wums.fitutils
import wums.tfutils

from scipy.optimize import curve_fit
import scipy.optimize
import boostHistHelpers as bhh

np.set_printoptions(threshold=sys.maxsize)


def polish_fit(bhist, func, parmvals, norm_axes=None, func_constraint=None, args=(), args_constraint=(), edmtol=0.1, maxiter=1000, max_attempts=3):
    """
    Minimize the same binned likelihood as wums.fitutils.fit_hist (mode nll_bin_integrated) with scipy's trust-exact
    method, which uses the full hessian, starting from parmvals. For fits where trust-krylov stops before reaching
    the minimum. Returns the same dictionary as fit_hist.
    """
    dtype = tf.float64
    xvals = [tf.constant(c, dtype=dtype) for c in bhist.axes.centers]
    xwidths = [tf.constant(w, dtype=dtype) for w in bhist.axes.widths]
    xedges = [tf.constant(e, dtype=dtype) for e in bhist.axes.edges]
    yvals = tf.constant(bhist.values(), dtype=dtype)
    yvariances = tf.constant(bhist.variances(), dtype=dtype)
    val_grad_args = {
        "func_loss": wums.fitutils.nll_loss_bin_integrated,
        "func_constraint": func_constraint,
        "args_loss": (xvals, xwidths, xedges, yvals, yvariances, func, norm_axes),
        "extra_args_loss": args,
        "args_constraint": (xvals, xedges),
        "extra_args_constraint": args_constraint,
    }

    cache = {}
    def val_grad(x):
        if cache.get("x_grad") is None or not np.array_equal(cache["x_grad"], x):
            loss, grad = wums.fitutils.val_grad(wums.fitutils.loss_with_constraint, parms=tf.constant(x, dtype=dtype), **val_grad_args)
            cache["x_grad"], cache["val_grad"] = np.copy(x), (loss.numpy(), grad.numpy())
        return cache["val_grad"]

    def hess(x):
        _, _, h = wums.fitutils.val_grad_hess(wums.fitutils.loss_with_constraint, parms=tf.constant(x, dtype=dtype), **val_grad_args)
        return h.numpy()

    # the cdf is only once differentiable (pchip), so the minimization can stall near the minimum, restarting
    # resets the trust region
    x, nit = parmvals, 0
    for attempt in range(max_attempts):
        res = scipy.optimize.minimize(lambda x: val_grad(x)[0], x, method="trust-exact", jac=lambda x: val_grad(x)[1], hess=hess, options={"gtol": 1e-6, "maxiter": maxiter})
        x, nit = res.x, nit + res.nit

        loss, grad, h = wums.fitutils.val_grad_hess(wums.fitutils.loss_with_constraint, parms=tf.constant(x, dtype=dtype), **val_grad_args)
        loss, grad, h = loss.numpy(), grad.numpy(), h.numpy()
        eigvals = np.linalg.eigvalsh(h)
        try:
            edmval = 0.5 * grad @ np.linalg.solve(h, grad)
            cov = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            edmval, cov = 99., np.zeros_like(h)
        if 0. <= edmval < edmtol and eigvals[0] > 0.:
            break

    return {
        "x": x,
        "hess": h,
        "cov": cov,
        "status": 0 if 0. <= edmval < edmtol else 1,
        "covstatus": 0 if eigvals[0] > 0. else 1,
        "hess_eigvals": eigvals,
        "edmval": edmval,
        "loss_val": loss,
        "polish_message": res.message,
        "polish_nit": nit,
    }


class QuantileFitter:

    def __init__(self, proc, met, procLabel, metLabel, dataDir, plotDir, comp, lumiLabel, logging, suffix=""):
        self.proc = proc
        self.met = met
        self.procLabel = procLabel
        self.metLabel = metLabel
        self.dataDir = dataDir
        self.plotDir = plotDir
        self.comp = comp
        self.lumiLabel = lumiLabel
        self.dtype = tf.float64
        self.itype = tf.int64
        self.logging = logging
        self.suffix = suffix

        self.prefit_pkl = f"{self.dataDir}/{self.proc}_{self.comp}{self.suffix}_prefit.pkl"
        self.postfit_pkl = f"{self.dataDir}/{self.proc}_{self.comp}{self.suffix}_postfit.pkl"
        
    def setHistConfig(self, bhist, axis, qTMin, qTmax, recoilMin, recoilMax, qTRebin=1, recoilRebinFit=1, recoilRebinPlt=1):

        self.axis = axis
        self.recoilMin, self.recoilMax = recoilMin, recoilMax
        self.qTMin, self.qTMax = qTMin, qTmax
        self.qTRebin = qTRebin
        self.recoilRebinFit = recoilRebinFit
        self.recoilRebinPlt = recoilRebinPlt
        self.sub_bkg = False # toggle background subtraction

        # rebin the histograms (for fitting and plotting)
        self.bhist_fit = self.rebin(bhist, qTRebin, self.recoilRebinFit)
        self.bhist_plt = self.rebin(bhist, qTRebin, self.recoilRebinPlt)
        self.bhist_bkg_fit = None
        self.bhist_bkg_plt = None
        self.bhist_minus_bkg_fit = copy.deepcopy(self.bhist_fit) # histogram with background subtracted (or nominal if no background)

        self.xvals_fit = [tf.constant(center, dtype=self.dtype) for center in self.bhist_fit.axes.centers]
        self.xedges_fit = [tf.constant(edge, dtype=self.dtype) for edge in self.bhist_fit.axes.edges]
        self.xvals_plt = [tf.constant(center, dtype=self.dtype) for center in self.bhist_plt.axes.centers]
        self.xedges_plt = [tf.constant(edge, dtype=self.dtype) for edge in self.bhist_plt.axes.edges]

        self.bins_qt = self.bhist_fit.axes[0].edges
        self.bins_recoil = self.bhist_fit.axes[1].edges
        self.centers_qt = self.bhist_fit.axes[0].centers
        self.centers_recoil = self.bhist_fit.axes[1].centers
        self.nBins_qt = len(self.bins_qt)-1
        self.bkg_cdf = None
        self.scale_sf_bkg = None
        self.scale_sf_sig = None
        self.bins_recoil_plt = self.bhist_plt.axes[1].edges
        self.centers_recoil_plt = self.bhist_plt.axes[1].centers

        self.edges_qt_tf = tf.constant(self.centers_qt, dtype=self.dtype)
        self.edges_qt_tf = self.edges_qt_tf[:, None] # col vector

        self.bkg_cdf_fit = None
        self.bkg_cdf_plt = None
        self.scale_sf_bkg_fit = None
        self.scale_sf_sig_fit = None
        self.scale_sf_bkg_plt = None
        self.scale_sf_sig_plt = None



    def rebin(self, bhist, qTRebin, recoilRebin):
        s = hist.tag.Slicer()
        bhist = bhist[{"qt": s[complex(0,self.qTMin):complex(0,self.qTMax)], self.axis: s[complex(0,self.recoilMin):complex(0,self.recoilMax)]}]
        bhist = bhh.rebinHist(bhist, self.axis, recoilRebin)
        bhist = bhh.rebinHist(bhist, "qt", qTRebin)
        return bhist


    def setQuantiles(self, quantiles=[], quantiles_hi=[], bins_for_quantiles=[]):
        if len(bins_for_quantiles) > 0: 
            self.logging.info(f"Automatically calculate the quantiles based on the binning provided")
            s = hist.tag.Slicer()
            
            tmp = self.rebin(self.bhist_fit, self.qTRebin, bins_for_quantiles)
            tmp = tmp[{"qt" : s[self.qTMin:complex(0,self.qTMax):hist.sum]}] # sum over qT

            yvals = tf.constant(tmp.values(), dtype=self.dtype)
            hist_cdfvals = tf.cumsum(yvals, axis=0)/tf.reduce_sum(yvals, axis=0, keepdims=True) # normalized
            hist_cdfvals = tf.concat([tf.constant([0], dtype=self.dtype), hist_cdfvals], axis=0) # 53 add 0

            edges =  tf.constant(tmp.axes[0].edges, dtype=self.dtype)
            centers =  tf.constant(tmp.axes[0].centers, dtype=self.dtype)
            self.quant_cdfvals = quants = wums.fitutils.pchip_interpolate(edges, hist_cdfvals, centers)
            
            self.quant_cdfvals = tf.concat([tf.constant([0], dtype=self.dtype), self.quant_cdfvals, tf.constant([1], dtype=self.dtype)], axis=0) # 53 add 0
            
            '''
            tmp = self.rebin(self.bhist_fit, self.qTRebin, test)
            tmp = tmp[{"qt" : s[50:complex(0,self.qTMax):hist.sum]}] # sum over qT
            yvals = tf.constant(tmp.values(), dtype=self.dtype)
            hist_cdfvals = tf.cumsum(yvals, axis=0)/tf.reduce_sum(yvals, axis=0, keepdims=True) # normalized
            hist_cdfvals = tf.concat([tf.constant([0], dtype=self.dtype), hist_cdfvals], axis=0) # 53 add 0

            edges =  tf.constant(tmp.axes[0].edges, dtype=self.dtype)
            centers =  tf.constant(tmp.axes[0].centers, dtype=self.dtype)
            self.quant_cdfvals_hi = quants = wums.fitutils.pchip_interpolate(edges, hist_cdfvals, centers)
            
            self.quant_cdfvals_hi = tf.concat([tf.constant([0], dtype=self.dtype), self.quant_cdfvals_hi, tf.constant([1], dtype=self.dtype)], axis=0) # 53 add 0
            quantiles_hi = self.quant_cdfvals_hi.numpy()
            '''
            print(self.quant_cdfvals)
            #print(quantiles_hi)

        elif len(quantiles) > 0:
            if quantiles[-1] != 1:
                self.quant_cdfvals = quantiles + [1.-q for q in reversed(quantiles[:-1])] # symmetrize
            else:
                self.quant_cdfvals = quantiles
        else:
            self.logging.info(f"Provide either quantiles or bins_for_quantiles")
            quit()
        self.nQuants = len(self.quant_cdfvals)
        self.nParams = self.nQuants-1
        self.nQparms = self.nQuants-1

        self.logging.info(f"Number of quantiles: {self.nQuants}")
        self.logging.info(f"Number of qparms: {self.nQparms}")

        if len(quantiles_hi) == 0: # constant quantiles
            self.quant_cdfvals_tf = tf.constant(self.quant_cdfvals, dtype=self.dtype)
            self.quant_cdfvals_tf = self.quant_cdfvals_tf[None, :]

        else:
            if True: # linear quantiles
                if quantiles_hi[-1] != 1:
                    quant_cdfvals_hi = quantiles_hi + [1.-q for q in reversed(quantiles_hi[:-1])]
                else:
                    quant_cdfvals_hi = quantiles_hi
                    print("lollll")
                quant_cdfvals_lo = np.array(self.quant_cdfvals)
                quant_cdfvals_lo_tf = tf.constant(quant_cdfvals_lo, dtype=self.dtype)
                quant_cdfvals_hi_tf = tf.constant(quant_cdfvals_hi, dtype=self.dtype)
                qt_tf = tf.constant(self.centers_qt, dtype=self.dtype)
                quant_cdfvals_lo_tf_row = tf.expand_dims(quant_cdfvals_lo_tf, axis=0)
                quant_cdfvals_hi_tf_row = tf.expand_dims(quant_cdfvals_hi_tf, axis=0)
                qt_tf_col = tf.expand_dims(qt_tf, axis=1)
                self.quant_cdfvals_tf = qt_tf_col*(quant_cdfvals_hi_tf_row-quant_cdfvals_lo_tf_row)/max(self.bins_qt) + quant_cdfvals_lo_tf_row

            if False: # log quantiles
                if quantiles[-1] != 1:
                    quant_cdfvals_hi = quantiles_hi + [1.-q for q in reversed(quantiles_hi[:-1])]
                else:
                    quant_cdfvals_hi = quantiles_hi
                quant_cdfvals_lo = np.array(self.quant_cdfvals)
                quant_cdfvals_lo_tf = tf.math.log(tf.constant(quant_cdfvals_lo, dtype=self.dtype))
                quant_cdfvals_hi_tf = tf.math.log(tf.constant(quant_cdfvals_hi, dtype=self.dtype))
                qt_tf = tf.constant(self.centers_qt, dtype=self.dtype)
                quant_cdfvals_lo_tf_row = tf.expand_dims(quant_cdfvals_lo_tf, axis=0)
                quant_cdfvals_hi_tf_row = tf.expand_dims(quant_cdfvals_hi_tf, axis=0)
                qt_tf_col = tf.expand_dims(qt_tf, axis=1)
                self.quant_cdfvals_tf = qt_tf_col*(quant_cdfvals_hi_tf_row-quant_cdfvals_lo_tf_row)/max(self.bins_qt)+ quant_cdfvals_lo_tf_row
                self.quant_cdfvals_tf = tf.math.exp(self.quant_cdfvals_tf)

            if False: # qT weighted quantiles
                if quantiles[-1] != 1:
                    quant_cdfvals_hi = quantiles_hi + [1.-q for q in reversed(quantiles_hi[:-1])]
                else:
                    quant_cdfvals_hi = quantiles_hi
                quant_cdfvals_lo = np.array(self.quant_cdfvals)
                quant_cdfvals_lo_tf = tf.constant(quant_cdfvals_lo, dtype=self.dtype)
                quant_cdfvals_hi_tf = tf.constant(quant_cdfvals_hi, dtype=self.dtype)
                quant_cdfvals_lo_tf_row = tf.expand_dims(quant_cdfvals_lo_tf, axis=0)
                quant_cdfvals_hi_tf_row = tf.expand_dims(quant_cdfvals_hi_tf, axis=0)
                bhist_incl_rec = self.bhist_quants[{1: bh.sum}] # inclusive over recoil axis
                qt_np = bhist_incl_rec.to_numpy(flow=False)[0]
                qt_np /= np.max(qt_np)
                qt_tf = tf.constant(qt_np, dtype=self.dtype)
                qt_tf_col = tf.expand_dims(qt_tf, axis=1)        
                self.quant_cdfvals_tf = qt_tf_col*(quant_cdfvals_lo_tf_row) + (1.0-qt_tf_col)*quant_cdfvals_hi_tf_row 

            tmp = self.quant_cdfvals_tf.numpy()
            tmp[:,0] = 0
            tmp[:,-1] = 1
            self.quant_cdfvals_tf = tf.constant(tmp, dtype=self.dtype)

        s = bh.tag.Slicer()
        hist_quantiles, hist_quantiles_err = wums.fitutils.hist_to_quantiles(self.bhist_fit, self.quant_cdfvals_tf, axis=1)
        ev_map = ROOT.TH2D("ev_map", "", self.nParams, 0, self.nParams, len(self.bins_qt)-1, 0, len(self.bins_qt)-1)
        for qTbin in range(0, len(self.bins_qt)-1):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            h = self.qTslice(self.bhist_fit, qTlow, qThigh)
            hist_quantiles_ = hist_quantiles[qTbin]
            #print(hist_quantiles_)
            for i in range(0, len(hist_quantiles_)-1):
                evts = -1
                try:
                    h_ = h[{0: s[complex(0,hist_quantiles_[i]):complex(0,hist_quantiles_[i+1])]}]
                    evts = h_.sum().value
                except:
                    evts = -1
                ev_map.SetBinContent(i+1, qTbin+1, evts)

        c = ROOT.TCanvas("c", "", 1200, 1200)
        c.SetRightMargin(c.GetRightMargin()*1.2)
        c.SetLeftMargin(c.GetLeftMargin()*1.1)
        c.SetLogz()

        ev_map.GetZaxis().SetRangeUser(0.1, 1e6)
        ev_map.Draw("COL Z")
        ev_map.GetXaxis().SetTitle("Quantile bin")
        ev_map.GetYaxis().SetTitle("q_{T} bin")
        c.SaveAs(f"{self.plotDir}/ev_map.png")
        c.SaveAs(f"{self.plotDir}/ev_map.pdf")


    def setSplineConfig(self, knots_qt, extrpl=[None, None]):
        self.extrpl = extrpl
        self.order = len(knots_qt)
        self.knots_qt = tf.constant(knots_qt, dtype=self.dtype)
        self.knots_qt = self.knots_qt[:, None] # col vector
        self.knots_qt_np = knots_qt

        self.baseFunction = rfb.CubicSpline()
        self.func_model = getattr(self.baseFunction, f"cond_spline")
        self.func_constraint = getattr(self.baseFunction, f"cond_spline_constraint")



    def subtractBackground(self, bhist, sf=1.):
        bhist = self.rebin(bhist)
        bhist_no_errors = copy.deepcopy(bhist)
        bhist_no_errors.reset()
        bkg_vals = bhist.values()
        newvals = self.bhist.view().value - bkg_vals*sf
        newvals = np.where(newvals < 0, 0, newvals) # force positive or zero bin contents
        self.bhist.view().value = newvals

    def readpkl(self, fIn):
        fIn = open(fIn, "rb")
        pkl = pickle.load(fIn)
        return pkl


    def setBackground(self, pkl_bkg, bhist_bkg, sf=1.):
        self.sub_bkg = True
        bhist_bkg = bhist_bkg * sf # not in place, the input histogram is reused for other variations
        self.bhist_bkg_fit = self.rebin(bhist_bkg, self.qTRebin, self.recoilRebinFit)
        self.bhist_bkg_plt = self.rebin(bhist_bkg, self.qTRebin, self.recoilRebinPlt)
        #self.bhist_minus_bkg_fit = self.bhist_minus_bkg_fit + self.bhist_bkg_fit*(-1)
        self.bhist_minus_bkg_fit += self.bhist_bkg_fit*(-1)
        np.clip(self.bhist_minus_bkg_fit.values(flow=True), a_min=0, a_max=None, out=self.bhist_minus_bkg_fit.values(flow=True)) # remove negative values

        cfg_bkg = self.readpkl(pkl_bkg)
        parms_bkg = cfg_bkg['x']
        parms_bkg_tf = tf.constant(parms_bkg, dtype=self.dtype)

        # construct cdf/pdf -- assume the same configuration as for the data fit
        self.bkg_args_fit = (cfg_bkg['quant_cdfvals_tf'], cfg_bkg['knots_qt'], self.edges_qt_tf, cfg_bkg['order'], cfg_bkg['extrpl'], None, None, None)
        self.bkg_args_plt = (cfg_bkg['quant_cdfvals_tf'], cfg_bkg['knots_qt'], self.edges_qt_tf, cfg_bkg['order'], cfg_bkg['extrpl'], None, None, None)
        self.bkg_func = self.func_model # assume the same, fetch from pkl
        self.bkg_cdf_fit = self.bkg_func(self.xvals_fit, self.xedges_fit, parms_bkg_tf, *self.bkg_args_fit)
        self.bkg_cdf_plt = self.bkg_func(self.xvals_plt, self.xedges_plt, parms_bkg_tf, *self.bkg_args_plt)

        yields_bkg_fit = tf.math.reduce_sum(tf.convert_to_tensor(self.bhist_bkg_fit.to_numpy()[0]), axis=1)
        yields_data_fit = tf.math.reduce_sum(tf.convert_to_tensor(self.bhist_fit.to_numpy()[0]), axis=1)
        #yields_bkg_fit = tf.where(tf.equal(yields_bkg_fit, 0), 1, yields_bkg_fit) 
        #yields_data_fit = tf.where(tf.equal(yields_data_fit, 0), 1, yields_data_fit) 
        self.scale_sf_bkg_fit = tf.linalg.diag(tf.math.divide(yields_bkg_fit, yields_data_fit))
        self.scale_sf_sig_fit = tf.linalg.diag(tf.math.divide(yields_data_fit-yields_bkg_fit, yields_data_fit))

        yields_bkg_plt = tf.math.reduce_sum(tf.convert_to_tensor(self.bhist_bkg_plt.to_numpy()[0]), axis=1)
        yields_data_plt = tf.math.reduce_sum(tf.convert_to_tensor(self.bhist_plt.to_numpy()[0]), axis=1)
        #yields_bkg_plt = tf.where(tf.equal(yields_bkg_plt, 0), 1, yields_bkg_plt) 
        #yields_data_plt = tf.where(tf.equal(yields_data_plt, 0), 1, yields_data_plt) 
        self.scale_sf_bkg_plt = tf.linalg.diag(tf.math.divide(yields_bkg_plt, yields_data_plt))
        self.scale_sf_sig_plt = tf.linalg.diag(tf.math.divide(yields_data_plt-yields_bkg_plt, yields_data_plt))
        
        #print(yields_bkg_fit)
        #print(self.bhist_bkg_fit)
        #print(self.scale_sf_bkg_fit)
        #print(self.scale_sf_sig_fit)
        #quit()

    def get_eigenvectors(self, hess, num_null = 0, scale=False):
        e,v = np.linalg.eigh(hess)

        # remove the null eigenvectors
        e = e[None, num_null:]
        v = v[:, num_null:]

        if scale:
            return e, v/np.sqrt(e)
        else:
            return e, v

  
    def qTslice(self, bhist, qTlow, qThigh, norm=False):
        s = hist.tag.Slicer()
        h = bhist[{"qt": s[complex(0,qTlow):complex(0,qThigh)]}]
        h = h[{"qt": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum  
        if norm:
            h = h/h.sum().value
        return h
        
    def makeTGraph(self, x, y, x_err=[], y_err=[], style="marker"):
        gr = ROOT.TGraphErrors()
        gr.SetName(f"{random.randint(0,9999)}")
        if style == "marker":
            gr.SetLineColor(ROOT.kBlack)
            gr.SetMarkerStyle(20)
            gr.SetMarkerSize(0.3)
            gr.SetMarkerColor(ROOT.kBlack)
            gr.SetLineColor(ROOT.kBlack)
        else:
            gr.SetLineWidth(2)
            gr.SetMarkerSize(0)
            gr.SetLineColor(ROOT.kRed)
        if len(x_err) == 0:
            x_err = [0]*len(x)
        if len(y_err) == 0:
            y_err = [0]*len(x)
        for i in range(0, len(x)):
            gr.SetPoint(i, x[i], y[i])
            gr.SetPointError(i, x_err[i], y_err[i])
        return gr
        

    def qparms_prefit(self, yRatio=1.08, newOrder=-1):
        self.logging.info("Plot qparms prefit")
        outDir = f"{self.plotDir}/qparms_prefit/"
        utils.mkdir(outDir, True)

        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]

        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = wums.fitutils.hist_to_quantiles(self.bhist_minus_bkg_fit, self.quant_cdfvals_tf, axis=1)
        qparms, qparms_err =  wums.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax, axis=1)
        cdfvals =  wums.fitutils.qparms_to_quantiles(qparms, x_low=self.recoilMin, x_high=self.recoilMax, axis=1)

        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
                y_vals[iPar].append(qparms[qTbin][iPar].numpy())
                y_vals_tf_err[iPar].append(qparms_err[qTbin][iPar])

        cfgPlot = {

            'logy'              : False,
            'logx'              : False,
        
            'xmin'              : self.qTMin,
            'xmax'              : self.qTMax,
            'ymin'              : -10,
            'ymax'              : 0,
            
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "qparm (prefit)",
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }

        outDict = {}
        outDict['x'] = np.empty(0) # nominal parameters of fit

        qparms_val_init = wums.fitutils.cubic_spline_interpolate(self.edges_qt_tf, qparms, self.knots_qt, axis=0, extrpl=self.extrpl)
        qparms_init = wums.fitutils.cubic_spline_interpolate(self.knots_qt, qparms_val_init, self.edges_qt_tf, axis=0, extrpl=self.extrpl)
        qparms_val_init_np = qparms_val_init.numpy().T
        qparms_init_np = qparms_init.numpy().T

        for i in range(0, self.nParams):
            
            g_data = self.makeTGraph(x_vals, y_vals_tf[i], y_err=y_vals_tf_err[i])

            popt = qparms_val_init_np[i]
            outDict['x'] = np.append(outDict['x'], qparms_val_init_np[i])
            func_eval = qparms_init_np[i]

            # scipy scpline (for testing)
            #cs = CubicSpline(x_vals, y_vals_tf[i], bc_type="natural")
            #popt = cs(self.qT_vals_np) # coeff at the knots = y values
            #cs_eval = CubicSpline(self.qT_vals_np, popt, bc_type="natural")
            #func_eval = cs_eval(self.centers_qt)

            g_fit = self.makeTGraph(x_vals, func_eval, style="line")
            g_ratio = self.makeTGraph(x_vals, np.divide(np.array(y_vals_tf[i]), func_eval))

            cfgPlot['ytitle'] = "qparm %d (prefit)" % i
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
            g_data.Draw("PE SAME")
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

            latex.SetTextSize(0.035)
            for k in range(0, len(popt)):
                latex.DrawLatex(0.6, 0.86-k*0.04, "a_{%d} = %.4e" % (k, popt[k]))

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
            canvas.SaveAs("{}/q{:03d}_tf.png".format(outDir, i))
            canvas.SaveAs("{}/q{:03d}_tf.pdf".format(outDir, i))

        with open(self.prefit_pkl, 'wb') as handle:
            outDict['qTMin'] = self.qTMin
            outDict['qTMax'] = self.qTMax
            outDict['recoilMin'] = self.recoilMin
            outDict['recoilMax'] = self.recoilMax
            outDict['bins_recoil'] = self.bins_recoil
            outDict['bins_qt'] = self.bins_qt
            outDict['centers_recoil'] = self.centers_recoil
            outDict['centers_qt'] = self.centers_qt
            outDict['quant_cdfvals'] = self.quant_cdfvals
            outDict['order'] = self.order
            outDict['nParams'] = self.nParams
            pickle.dump(outDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def qparms_postfit(self, yRatio=1.08):
        self.logging.info("Plot qparms postfit")
        outDir = f"{self.plotDir}/qparms_postfit/"
        utils.mkdir(outDir, True)

        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]

        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = wums.fitutils.hist_to_quantiles(self.bhist_minus_bkg_fit, self.quant_cdfvals_tf, axis=1) # , cdfvals_bkg=self.bkg_cdf 
        qparms, qparms_err =  wums.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax) # shape (qTbins, nQparms)

        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
                y_vals[iPar].append(qparms[qTbin][iPar].numpy())
                y_vals_tf_err[iPar].append(qparms_err[qTbin][iPar])

        fIn = open(self.postfit_pkl, "rb")
        pkl = pickle.load(fIn)
        parms_postfit = pkl['x']

        # unpack parms_postfit to (nKnots, nQuants)
        parms_postfit_tf = tf.constant(parms_postfit, dtype=self.dtype)
        parms_postfit_tf = tf.reshape(parms_postfit_tf, (-1, self.order))
        parms_postfit_tf = tf.transpose(parms_postfit_tf)

        cfgPlot = {

            'logy'              : False,
            'logx'              : False,
        
            'xmin'              : self.qTMin,
            'xmax'              : self.qTMax,
            'ymin'              : -10,
            'ymax'              : 0,
            
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "qparm (postfit)",
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }

        qparms_init = wums.fitutils.cubic_spline_interpolate(self.knots_qt, parms_postfit_tf, self.edges_qt_tf, axis=0, extrpl=self.extrpl)
        qparms_init_np = qparms_init.numpy().T

        for i in range(0, self.nParams):

            g_data = self.makeTGraph(x_vals, y_vals_tf[i], y_err=y_vals_tf_err[i])
            popt = parms_postfit[i*(self.order):(i+1)*(self.order)]
            func_eval = qparms_init_np[i]

            g_fit = self.makeTGraph(x_vals, func_eval, style="line")
            g_ratio = self.makeTGraph(x_vals, np.divide(np.array(y_vals_tf[i]), func_eval))

            cfgPlot['ytitle'] = "qparm %d (postfit)" % i
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
            g_data.Draw("PE SAME")
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
            latex.SetTextSize(0.035)
            for k in range(0, len(popt)):
                latex.DrawLatex(0.6, 0.86-k*0.04, "a_{%d} = %.4e" % (k, popt[k]))

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
            canvas.SaveAs("{}/q{:03d}_tf.png".format(outDir, i))
            canvas.SaveAs("{}/q{:03d}_tf.pdf".format(outDir, i))




    def plotQuantiles(self, yRatio=1.08):
        outDir = f"{self.plotDir}/quantiles_param_postfit/"
        utils.mkdir(outDir, True)
                
        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]
        
        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = wums.fitutils.hist_to_quantiles(self.bhist_quants, self.quant_cdfvals_tf, axis=1) # , cdfvals_bkg=self.bkg_cdf 
        qparms, qparms_err =  wums.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax) # shape (qTbins, nQparms)
        
        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
                y_vals[iPar].append(qparms[qTbin][iPar].numpy())
                y_vals_tf_err[iPar].append(qparms_err[qTbin][iPar])

        cfgPlot = {

            'logy'              : False,
            'logx'              : False,
        
            'xmin'              : self.qTMin,
            'xmax'              : self.qTMax,
            'ymin'              : 0,
            'ymax'              : 1,
            
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "Qparm_{#perp}",
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }


        if self.doSpline:

            #qparms_val_init = wums.fitutils.pchip_interpolate(self.edges_qt_tf, qparms, self.qT_vals, axis=0) # (nKnots, nQuants)
            #qparms_init = wums.fitutils.pchip_interpolate(self.qT_vals, qparms_val_init, self.edges_qt_tf, axis=0) # to visualize spline
            #qparms_init_np = qparms_init.numpy().T
            
            qparms_val_init = rfb.CubicSpline.cubic_spline_interpolate_tf(self.edges_qt_tf, qparms, self.qT_vals, axis=0) # (nKnots, nQuants)
            qparms_init = rfb.CubicSpline.cubic_spline_interpolate_tf(self.qT_vals, qparms_val_init, self.edges_qt_tf, axis=0) # to visualize spline
            qparms_init_np = qparms_init.numpy().T
            
            #self.qparms_val_init = qparms_val_init
            self.qparms_val_init = []
           
        else:
            self.qparms_val_init = []
            func = getattr(self.baseFunction, f"pol{self.order}")
            nZeros = 0
            initial_parms = [1]*(self.order+1)
        
        for i in range(0, self.nParams):
            
            g_data = self.makeTGraph(x_vals, y_vals_tf[i], y_err=y_vals_tf_err[i])

            if self.doSpline:
                if (i < 0 or i >= self.nParams-0):
                    print(i)
                    initial_parms_ = [1]*(1+1)
                    func_ = getattr(self.baseFunction, f"pol1")
                    popt, pcov, infodict, mesg, ier = curve_fit(func_, xdata=x_vals, ydata=y_vals_tf[i], sigma=y_vals_tf_err[i], p0=initial_parms_, full_output=True)
                    errs = np.sqrt(np.diagonal(pcov))
                    func_eval = func_(np.array(x_vals), *popt)
                    for x in popt:
                        self.qparms_val_init.append(x)
                    func_eval = func_(np.array(x_vals), *popt)
                else:
                    tmp = qparms_val_init.numpy().T
                    for x in tmp[i]:
                        self.qparms_val_init.append(x)
                    func_eval = qparms_init_np[i]
                    #print(tmp[i])
                    
                    
                    ## cubic spline
                    # cspline
                    #splines_val_init = rfb.CubicSpline.cubic_spline_interpolate(x_vals, y_vals[i], self.qT_vals_np)
                    #print(splines_val_init)
                    
                    #splines_val_init = rfb.CubicSpline.cubic_spline_interpolate_alt(x_vals, y_vals[i], self.qT_vals_np)
                    #print(splines_val_init)
               
                    
                    
                    #splines_val_init = rfb.CubicSpline.cubic_spline_interpolate_test(x_vals, y_vals[i], self.qT_vals_np)
                    #func_eval = rfb.CubicSpline.cubic_spline_interpolate_test(self.qT_vals_np, splines_val_init, x_vals)
       
                    #qparms_init = wums.fitutils.pchip_interpolate(self.qT_vals, qparms_val_init, self.edges_qt_tf) # to visualize spline
                    #qparms_init_np = qparms_init.numpy().T
            
                g_fit = self.makeTGraph(x_vals, func_eval, style="line")
                g_ratio = self.makeTGraph(x_vals, np.divide(np.array(y_vals_tf[i]), func_eval))
            else:
                if (i < 0 or i >= self.nParams-0):
                    print(i)
                    initial_parms_ = [1]*(1+1)
                    func_ = getattr(self.baseFunction, f"pol1")
                    popt, pcov, infodict, mesg, ier = curve_fit(func_, xdata=x_vals, ydata=y_vals_tf[i], sigma=y_vals_tf_err[i], p0=initial_parms_, full_output=True)
                    errs = np.sqrt(np.diagonal(pcov))
                    func_eval = func_(np.array(x_vals), *popt)
                    for x in popt:
                        self.qparms_val_init.append(x)
                else:
                    popt, pcov, infodict, mesg, ier = curve_fit(func, xdata=x_vals, ydata=y_vals_tf[i], sigma=y_vals_tf_err[i], p0=initial_parms, full_output=True)
                    errs = np.sqrt(np.diagonal(pcov))
                    func_eval = func(np.array(x_vals), *popt)
                    for x in popt:
                        self.qparms_val_init.append(x)
                g_fit = self.makeTGraph(x_vals, func_eval, style="line")
                g_ratio = self.makeTGraph(x_vals, np.divide(np.array(y_vals_tf[i]), func_eval))
                
                
            cfgPlot['ymin'] = min(y_vals_tf[i])*(0.3 if min(y_vals_tf[i]) > 0 else 1.7)
            cfgPlot['ymax'] = max(y_vals_tf[i])*(1.3 if max(y_vals_tf[i]) > 0 else 0.7)
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
            g_data.Draw("PE SAME")
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
            if not self.doSpline:
                latex.DrawLatex(0.2, 0.75, "Fit status = %d" % ier)
                
                latex.SetTextSize(0.035)
                for k in range(0, len(popt)):
                    latex.DrawLatex(0.6, 0.86-k*0.04, "a_{%d} = %.2e #pm %.2e" % (k, popt[k], errs[k]))

  
  
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
            canvas.SaveAs("{}/q{:03d}_tf.png".format(outDir, i))
            canvas.SaveAs("{}/q{:03d}_tf.pdf".format(outDir, i))
            

    def paramsVsQuantile(self):
        outDir = f"{self.plotDir}/quantiles_param_vs_q/"
        utils.mkdir(outDir, True)
        jsIn = utils.loadJSON(f"{self.plotDir}/quantiles_param/results_param.json")

        x_vals = [0.5*(self.quant_cdfvals[i-1]+self.quant_cdfvals[i]) for i in range(1, self.nQuants)] # quantile midpoints
        y_vals = [[] for g in range(0, self.order+1)]
        y_vals_err = [[] for g in range(0, self.order+1)]
        
        for iPar in range(self.nParams): 
            for iC in range(jsIn[str(iPar)]['order']+1):
                y_vals[iC].append(jsIn[str(iPar)]['coeff'][iC])
                y_vals_err[iC].append(jsIn[str(iPar)]['coeff_err'][iC])

    
        yRatio = 1.08
        cfg = {
            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 1,
            'ymin'              : -2,
            'ymax'              : 2,
                
            'xtitle'            : "CDF Quantile",
            'ytitle'            : "REC",
                
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
                
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }   
   
        for i in range(0, len(y_vals)):
            g = self.makeTGraph(x_vals, y_vals[i], y_err=y_vals[i])
            g.SetMarkerSize(0.75)

            yMin, yMax = min(y_vals[i]), max(y_vals[i])
            cfg['ymin'] = yMin*(0.8 if yMin > 0 else 1.2)
            cfg['ymax'] = yMax*(1.2 if yMax > 0 else 0.8)
            plotter.cfg = cfg
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
            g.Draw("PE SAME")
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
            padB.RedrawAxis()
            padB.RedrawAxis("G")
            canvas.Modify()
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs("%s/q%d.png" % (outDir, i))
            canvas.SaveAs("%s/q%d.pdf" % (outDir, i))
            ROOT.SetOwnership(canvas, False) # deleted here, not again by python
            canvas.Delete()



    def knots_vs_cdfvals(self):
        self.logging.info("Plot qparms postfit")
        outDir = f"{self.plotDir}/knots_cdfvals/"
        utils.mkdir(outDir, True)

        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]

        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = wums.fitutils.hist_to_quantiles(self.bhist_minus_bkg_fit, self.quant_cdfvals_tf, axis=1) # , cdfvals_bkg=self.bkg_cdf 
        qparms, qparms_err =  wums.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax) # shape (qTbins, nQparms)

        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
                y_vals[iPar].append(qparms[qTbin][iPar].numpy())
                y_vals_tf_err[iPar].append(qparms_err[qTbin][iPar])

        fIn = open(self.postfit_pkl, "rb")
        pkl = pickle.load(fIn)
        parms_postfit = pkl['x']
        print(pkl.keys())
        quant_cdfvals = pkl['quant_cdfvals_tf'][0].numpy()

        parms =  tf.constant(parms_postfit, dtype=self.dtype)
        print(parms)
        parms_2d = tf.reshape(parms, (-1, pkl['order']))
        
        parms_2d = tf.transpose(parms_2d).numpy()
        #parms_2d = parms_2d.numpy()
        print(parms_2d)
        print(pkl['order'])
        quant_cdfvals_centers = (quant_cdfvals[1:] + quant_cdfvals[:-1]) / 2

        




        yRatio = 1.08
        cfg = {
            'logy'              : False,
            'logx'              : True,
            
            'xmin'              : 1e-6,
            'xmax'              : 1,
            'ymin'              : -6,
            'ymax'              : -2,
                
            'xtitle'            : "CDF Quantile",
            'ytitle'            : "REC",
                
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
                
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }

        for iKnot in range(len(parms_2d)):
            g_data = self.makeTGraph(quant_cdfvals_centers, parms_2d[iKnot])
        
            plotter.cfg = cfg
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
            g_data.Draw("PE SAME")
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
            padB.RedrawAxis()
            padB.RedrawAxis("G")
            canvas.Modify()
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs(f"{outDir}/knot_{iKnot}.png")
            canvas.SaveAs(f"{outDir}/knot_{iKnot}.pdf")
            ROOT.SetOwnership(canvas, False) # deleted here, not again by python
            canvas.Delete()


    def fit(self, withConstraint=False, parms=[], ext="", polish_only=False):
        self.logging.info("Start fit")
        outDir = f"{self.plotDir}/quantiles_refit/"
        utils.mkdir(outDir, True)

        fit_args = (self.quant_cdfvals_tf, self.knots_qt, self.edges_qt_tf, self.order, self.extrpl, self.bkg_cdf_fit, self.scale_sf_sig_fit, self.scale_sf_bkg_fit)

        if ext == "":
            fIn = open(self.prefit_pkl, "rb")
        else:
            fIn = open(ext, "rb")

        pkl = pickle.load(fIn)
        parms_prefit = pkl['x']
        
        if len(parms) != 0:
            parms_prefit = parms

        start = time.time()
        res = None
        if polish_only:
            pass
        elif withConstraint:
            res = wums.fitutils.fit_hist(self.bhist_fit, self.func_model, parms_prefit, mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args, func_constraint=self.func_constraint, args_constraint=fit_args)
        else:
            res = wums.fitutils.fit_hist(self.bhist_fit, self.func_model, parms_prefit, mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args)

        # trust-krylov can stop before the minimum, continue constrained fits (no null directions) with trust-exact
        if withConstraint and (res is None or res['status'] != 0 or res['covstatus'] != 0):
            if res is not None:
                self.logging.info(" -> fit_hist ended with status={}, cov_status={}, edm={:.3e}, continue with trust-exact".format(res['status'], res['covstatus'], res['edmval']))
            res = polish_fit(self.bhist_fit, self.func_model, parms_prefit if res is None else res['x'], norm_axes=[1], func_constraint=self.func_constraint, args=fit_args, args_constraint=fit_args, edmtol=0.1)
            self.logging.info(" -> trust-exact: {} after {} iterations".format(res['polish_message'], res['polish_nit']))
        end = time.time()

        parms_vals = res['x']
        fit_status = res['status']
        cov_status = res['covstatus']
        hess_eigvals = res['hess_eigvals']
        cov_matrix = res['cov']
        edmval = res['edmval']
        hess = res['hess']
        self.cov_matrix = cov_matrix
        if not withConstraint:
            hess_eigvals_trunc = hess_eigvals[self.order:] # remove first N DOF
            cov_status_trunc = 1
            if hess_eigvals_trunc[0] > 0.:
                cov_status_trunc = 0
        else:
            cov_status_trunc = cov_status

        self.logging.info(" -> Fit ended, time={time:.3f} s, status={fit_status}, cov_status={cov_status}, cov_status_trunc={cov_status_trunc} edm={edm:.3e}".format(time=end-start, fit_status=fit_status, cov_status=cov_status, cov_status_trunc=cov_status_trunc, edm=edmval))
        print("parms_vals")
        print(parms_vals)

        print("hess_eigvals")
        print(hess_eigvals)

        if not withConstraint:
            print("hess_eigvals_trunc")
            print(hess_eigvals_trunc)

        # plot non-zero eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hess)
        #np.set_printoptions(sign=‘ ‘, suppress = True, linewidth=7*11, formatter = {'float_kind':'{:.4e}'.format})
        np.set_printoptions(suppress=True)

        for ieig, eig in enumerate(eigenvalues):
            if eig > 0:
                continue
            vec = eigenvectors[:, ieig]
            print(f" negative eigenvalue {ieig}, {eig}, min={np.min(np.abs(vec))}, max={np.max(np.abs(vec))}")
            print(vec)

        with open(self.postfit_pkl, 'wb') as handle:
            res['qTMin'] = self.qTMin
            res['qTMax'] = self.qTMax
            res['recoilMin'] = self.recoilMin
            res['recoilMax'] = self.recoilMax
            res['bins_recoil'] = self.bins_recoil
            res['bins_qt'] = self.bins_qt
            res['centers_recoil'] = self.centers_recoil
            res['centers_qt'] = self.centers_qt
            res['quant_cdfvals'] = self.quant_cdfvals
            res['quant_cdfvals_tf'] = self.quant_cdfvals_tf
            res['order'] = self.order
            res['knots_qt'] = self.knots_qt
            res['edges_qt_tf'] = self.edges_qt_tf
            res['extrpl'] = self.extrpl
            res['nParams'] = self.nParams
            res['withConstraint'] = withConstraint
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return parms_vals

    def ratioHist_tgraph(self, hNom, hRatio, pdf_values, pdf_sigmas, xMin=-9e99, xMax=9e99,):

        hist_pdf = hNom.Clone("hist_pdf")
        for i in range(1, hRatio.GetNbinsX()+1):
            x = hRatio.GetBinCenter(i)
            xLow , xHigh = hRatio.GetBinLowEdge(i), hRatio.GetBinLowEdge(i+1)
            if xLow > xMax:
                continue
            if xHigh < xMin:
                continue

            pdfVal = pdf_values[i-1]/(self.bins_recoil_plt[i]-self.bins_recoil_plt[i-1])
            pdfErr = pdf_sigmas[i-1]/(self.bins_recoil_plt[i]-self.bins_recoil_plt[i-1]) if len(pdf_sigmas) > 0 else 0
            y = hNom.GetBinContent(i)/pdfVal
            y_err = hNom.GetBinError(i)/hNom.GetBinContent(i) if hNom.GetBinContent(i) > 0 else 0
            hRatio.SetBinContent(i, y)
            hRatio.SetBinError(i, y_err) # datapoint error

            # only fill PDF when data is present in bin (neglect for chi2)
            if hNom.GetBinContent(i) > 0:
                hist_pdf.SetBinContent(i, pdfVal)
                hist_pdf.SetBinError(i, pdfErr)

        pval_chi2 = hist_pdf.Chi2Test(hNom, "WW")*100. # UU changes the NDF
        chi2 = hist_pdf.Chi2Test(hNom, "WW CHI2") # UU changes the NDF
        chndof = hist_pdf.Chi2Test(hNom, "WW CHI2/NDF") # UU changes the NDF
        ndof = chi2 / chndof

        pval_KS = hist_pdf.KolmogorovTest(hNom)*100.
        pval_AD = hist_pdf.AndersonDarlingTest(hNom)*100.
 
        #print(chndof, pval_chi2, pval_KS, pval_AD)
        #quit()
        if math.isnan(chi2) or math.isnan(chndof) or math.isnan(ndof):
            return -1, -1, -1, -1, -1, hRatio
        return chi2, ndof, pval_chi2, pval_KS, pval_AD, hRatio
   

    def pdf(self, parms, fit_args, xvals=[], edges=[]):
        #print("parms", parms[100])
        #parms[100] = 0
        tfparms = tf.constant(parms, dtype=self.dtype)
        if len(xvals) == 0 and len(edges) == 0:
            cdf = self.func_model(self.xvals_plt, self.xedges_plt, tfparms, *fit_args)
        else:
            cdf = self.func_model(xvals, edges, tfparms, *fit_args)

        pdf = cdf[:,1:] - cdf[:,:-1]
        pdf = tf.maximum(pdf, tf.zeros_like(pdf))

        # normalize the recoil PDF in each qT bin 
        # PDF is already normalized in each qT bin
        norms = np.sum(pdf, axis=1)
        norms[norms == 0] = 1 # avoid division by zero
        pdf = pdf / norms[:, np.newaxis]
        #print("pdf", pdf[0])
        return pdf


    def plot_refit_quantiles(self, prefit=False, yRatio=1.15, logY=True, yMin=-1, yMax=-1, xMin=-1, xMax=-1, ext=""):

        rebin = 1
        plotSignal = self.sub_bkg

        xMin = self.recoilMin if xMin == -1 else xMin
        xMax = self.recoilMax if xMax == -1 else xMax
        
        cov_status = -1
        fit_status = -1
        parms_unc = []
        doStat = True
        if ext == "":
            if prefit:
                doStat = False
                self.logging.info("Plot prefit")
                outDir = f"{self.plotDir}/pdf_prefit"

                fIn = open(self.prefit_pkl, "rb")
                pkl = pickle.load(fIn)
                parms_nom = pkl['x']

            else:
                self.logging.info("Plot postfit")
                outDir = f"{self.plotDir}/pdf_postfit"

                fIn = open(self.postfit_pkl, "rb")
                pkl = pickle.load(fIn)
                hess = pkl['hess']
                parms_nom = pkl['x']
                cov_status = pkl['covstatus']
                fit_status = pkl['status']

                if cov_status != 0:
                    doStat = False

                if doStat:
                    n_zeros = (0 if pkl['withConstraint'] else self.order)
                    #eigenvalues, eigenvectors = self.get_eigenvectors(hess, num_null=n_zeros)
                    #eig_vals, eig_vec = np.linalg.eigh(hess) # sorted
                    #eig_vec = np.delete(eig_vec, list(range(0, n_zeros)), axis=1) # remove first order+1 vectors
                    #eig_vals = np.delete(eig_vals, list(range(0, n_zeros)), axis=0) # remove first order+1 vectors
                    #eig_vals, eig_vec = self.get_eigenvectors(hess, num_null=n_zeros)
                    #eig_vals, eig_vec = np.linalg.eigh(hess) # sorted
                    #variances = np.reciprocal(np.sqrt(eig_vals)) # diag space # (1, 250)

                    eig_vals, eig_vec = np.linalg.eigh(self.cov_matrix) # sorted
                    variances = np.sqrt(eig_vals)

                    sigma = 1
                    for iPert, var in enumerate(variances):
                        var_pert = sigma*(np.array([0. if iPert != k else val for k,val in enumerate(variances)]))
                        var = eig_vec @ var_pert
                        parms_pert = list(parms_nom + var)
                        parms_unc.append(parms_pert)
        else:
            prefit = True
            outDir = f"{self.plotDir}/pdf_ext"

            fIn = open(ext, "rb")
            pkl = pickle.load(fIn)
            parms_nom = pkl['x']

            self.quant_cdfvals_tf = pkl['quant_cdfvals_tf']
            self.knots_qt = pkl['knots_qt']
            self.edges_qt_tf = pkl['edges_qt_tf'] #self.edges_qt_tf ??
            self.order = pkl['order']
            self.extrpl = pkl['extrpl']
            self.bkg_cdf_plt = None
            self.scale_sf_sig_plt = None
            self.scale_sf_bkg_plt = None


        if not logY:
            outDir += "_lin"
        utils.mkdir(outDir, True)

        # get the nominal pdf
        fit_args = (self.quant_cdfvals_tf, self.knots_qt, self.edges_qt_tf, self.order, self.extrpl, self.bkg_cdf_plt, self.scale_sf_sig_plt, self.scale_sf_bkg_plt)
        pdf = self.pdf(parms_nom, fit_args)
       
        if plotSignal:
            fit_args_sig = (self.quant_cdfvals_tf, self.knots_qt, self.edges_qt_tf, self.order, self.extrpl, None, None, None)
            pdf_sig = self.pdf(parms_nom, fit_args_sig)

        # sum all uncertainty in quadrature get uncertainty band (denser grid)
        if doStat:
            pdf_stat_variances = None
            nrecbins_stat = 200+1
            lspace = np.linspace(min(self.bins_recoil_plt), max(self.bins_recoil_plt), num=nrecbins_stat)
            xvals_plot_unc = copy.deepcopy(self.xvals_plt)
            edges_plot_unc = copy.deepcopy(self.xedges_plt)
            xvals_plot_unc[1] = tf.constant((lspace[1:] + lspace[:-1]) / 2, dtype=self.dtype) # midpoints
            edges_plot_unc[1] = tf.constant(lspace, dtype=self.dtype)[None, :]
            #pdf_nom_stat = self.pdf(parms_nom, fit_args, xvals=xvals_plot_unc, edges=edges_plot_unc)

            pdf_stat_variances = np.zeros_like(pdf)
            for istat, parms_pert in enumerate(parms_unc): # add all unc in quadrature
                #pdf_pert = self.pdf(parms_pert, fit_args, xvals=xvals_plot_unc, edges=edges_plot_unc)
                pdf_pert = self.pdf(parms_pert, fit_args)
                pdf_dpert = np.square(pdf - pdf_pert)
                pdf_stat_variances = np.add(pdf_stat_variances, pdf_dpert)
            pdf_stat = np.sqrt(pdf_stat_variances) + pdf # / pdf_nom_stat


        cfgPlot = {

            'logy'              : logY,
            'logx'              : False,

            'xmin'              : xMin,
            'xmax'              : xMax,
            'ymin'              : yMin,
            'ymax'              : yMax,

            'xtitle'            : "Recoil U_{#perp}   (GeV)" if self.comp == "perp" else "Recoil U_{#parallel} #plus q_{T} (GeV)",
            'ytitle'            : "Events" ,

            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }

        

        g_chi2 = ROOT.TGraphErrors()
        g_pchi2 = ROOT.TGraphErrors()
        g_pks = ROOT.TGraphErrors()
        hist_root_tot = None
        pdf_tot, sig_pdf_tot = [0]*len(self.centers_recoil_plt), [0]*len(self.centers_recoil_plt)
        yields, yields_err, pdf_fracs = [], [], []

        yield_tot = self.bhist_plt.sum().value
        hist_root_tot_unc_ratio = None
        for qTbin in range(0, len(self.bins_qt)-1):
            qT, qTlow, qThigh = self.centers_qt[qTbin], self.bins_qt[qTbin], self.bins_qt[qTbin+1]
            h = self.qTslice(self.bhist_plt, qTlow, qThigh)

            #print("############# Recoil bin %d [%.2f, %.2f]" % (qTbin, qTlow, qThigh))
            yield_, yield_err = h.sum().value, math.sqrt(h.sum().variance)
            yields.append(yield_)
            yields_err.append(yield_err)

            pdf_values = pdf[qTbin,:] # is normalized (or should be)
            pdf_values *= h.sum().value/np.sum(pdf_values) # renormalize PDF in each qT bin

            if plotSignal:
                h_bkg = self.qTslice(self.bhist_bkg_plt, qTlow, qThigh)
                pdf_sig_values = pdf_sig[qTbin,:] # is normalized (or should be)
                pdf_sig_values *= (h.sum().value-h_bkg.sum().value)/np.sum(pdf_sig_values) # renormalize PDF in each qT bin

            hist_root = narf.hist_to_root(h)
            #hist_root.Scale(1.0, "width")
            hist_root.SetLineColor(ROOT.kBlack)
            hist_root.SetMarkerStyle(20)
            hist_root.SetMarkerColor(ROOT.kBlack)
            hist_root.SetLineColor(ROOT.kBlack)
            hist_root.Scale(1, "width")
            if hist_root_tot == None: hist_root_tot = hist_root.Clone("hist_root_tot")
            else: hist_root_tot.Add(hist_root)
            
            
            if logY:
                if yMin < 0:
                    cfgPlot['ymin'] = math.pow(10., math.floor(math.log10(utils.getNonZeroMinimum(hist_root, xMin=xMin, xMax=xMax)))-1)
                if yMax < 0:
                    cfgPlot['ymax'] = math.pow(10., math.ceil(math.log10(hist_root.GetMaximum()))+4)
            else:
                cfgPlot['ymin'] = 0
                cfgPlot['ymax'] = 1.75*hist_root.GetMaximum()

            pdf_sigmas = []
            if doStat:
                hist_root_tot_unc_ratio_ = hist_root.Clone("hist_root_tot_unc_ratio_%d"%qTbin)
                hist_root_tot_unc_ratio_.SetFillColor(18)
                hist_root_tot_unc_ratio_.SetMarkerSize(0)
                hist_root_tot_unc_ratio_.SetLineWidth(0)
                hist_root_tot_unc_ratio_.Reset("ICE") # reset errors
                hist_root_tot_unc_ratio_.ResetStats()

                hist_root_tot_unc_ = hist_root.Clone("hist_root_tot_unc_%d"%qTbin)
                hist_root_tot_unc_.SetFillColor(18)
                hist_root_tot_unc_.SetMarkerSize(0)
                hist_root_tot_unc_.SetLineWidth(0)
                hist_root_tot_unc_.Reset("ICE") # reset errors
                hist_root_tot_unc_.ResetStats()

                pdf_stat_values = pdf_stat[qTbin,:]
                pdf_stat_values *= h.sum().value/np.sum(pdf_stat_values)
                for i in range(1, hist_root_tot_unc_ratio_.GetNbinsX()+1):
                    yld_nom = float(pdf_values[i-1])
                    yld_var = float(pdf_stat_values[i-1])
                    hist_root_tot_unc_ratio_.SetBinContent(i, 1)
                    hist_root_tot_unc_ratio_.SetBinError(i, (yld_var-yld_nom)/yld_nom)

                    sf = (self.bins_recoil_plt[i]-self.bins_recoil_plt[i-1])
                    hist_root_tot_unc_.SetBinContent(i, yld_nom/sf)
                    hist_root_tot_unc_.SetBinError(i, (yld_var-yld_nom)/sf)

                    pdf_sigmas.append(yld_var-yld_nom)

                if hist_root_tot_unc_ratio == None:
                    hist_root_tot_unc_ratio = hist_root_tot_unc_.Clone("hist_root_tot_unc_ratio")
                else:
                    hist_root_tot_unc_ratio.Add(hist_root_tot_unc_)
                #quit()
            plotter.cfg = cfgPlot
            canvas, padT, padB = plotter.canvasRatio()
            dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)

            canvas.cd()
            padT.Draw()
            padT.cd()
            padT.SetGrid()
            padT.SetTickx()
            padT.SetTicky()
            dummyT.Draw("HIST")

            if doStat:
                hist_root_tot_unc_.Draw("E4 SAME")

            hist_root.Draw("PE SAME")

            yield_ /= rebin # reduce yield to accomodate for binning
            g_pdf = ROOT.TGraphErrors()
            g_sig_pdf = ROOT.TGraphErrors()
            for i in range(0, len(self.centers_recoil_plt)): 
                #g_pdf.SetPoint(i, self.centers_recoil[i], pdf_values[i]*yield_/pdf_frac_range)
                g_pdf.SetPoint(i, self.centers_recoil_plt[i], pdf_values[i]/(self.bins_recoil_plt[i+1]-self.bins_recoil_plt[i]))
                pdf_tot[i] += pdf_values[i] #*yield_ #/pdf_frac_range
                if plotSignal:
                    sig_pdf_tot[i] += pdf_sig_values[i]/(self.bins_recoil_plt[i+1]-self.bins_recoil_plt[i]) #*yield_ #/pdf_frac_range


            g_pdf.SetLineColor(ROOT.kRed)
            g_pdf.SetLineWidth(3)
            g_pdf.Draw("L SAME")

            histRatio = hist_root.Clone("ratio")
            histRatio.Reset("ACE")
            histRatio.SetMarkerStyle(8)
            histRatio.SetMarkerSize(0.7)
            histRatio.SetMarkerColor(ROOT.kBlack)
            histRatio.SetLineColor(ROOT.kBlack)
            histRatio.SetLineWidth(1)
            chi2, ndof, pval_chi2, pval_KS, pval_AD, histRatio = self.ratioHist_tgraph(hist_root, histRatio, pdf_values, pdf_sigmas, xMin=xMin, xMax=xMax)


            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.040)
            latex.SetTextColor(1)
            latex.SetTextFont(42)
            latex.DrawLatex(0.20, 0.85, self.procLabel)
            latex.DrawLatex(0.20, 0.80, self.metLabel)
            latex.DrawLatex(0.20, 0.75, "q_{T} = [%.1f, %.1f] GeV" % (qTlow, qThigh))
            latex.DrawLatex(0.20, 0.70, "Mean = %.3f #pm %.3f" % (hist_root.GetMean(), hist_root.GetMeanError()))
            latex.DrawLatex(0.20, 0.65, "RMS = %.3f #pm %.3f" % (hist_root.GetRMS(), hist_root.GetRMSError()))
            latex.DrawLatex(0.20, 0.60, "Yield = %.3f #pm %.3f" % (yield_, yield_err))

            latex.DrawLatex(0.60, 0.85, "Fit/Cov status = %d/%d" % (fit_status, cov_status))
            latex.DrawLatex(0.60, 0.80, "#chi^{2}/ndof = %.3f/%d = %.3f" % (chi2, ndof, chi2/ndof))
            latex.DrawLatex(0.60, 0.75, "p-value #chi^{2} = %.1f %%" % pval_chi2)
            latex.DrawLatex(0.60, 0.70, "p-value KS = %.1f %%" % pval_KS)
            #latex.DrawLatex(0.60, 0.70, "p-value AD = %.1f %%" % pval_AD)

            padT.RedrawAxis()
            padT.RedrawAxis("G")
            plotter.auxRatio()
            canvas.cd()
            padB.Draw()
            padB.cd()
            padB.SetGrid()
            padB.SetTickx()
            padB.SetTicky()
            dummyB.Draw("HIST")
            if doStat:
                hist_root_tot_unc_ratio_.Draw("E4 SAME")
            dummyL.Draw("SAME")
            histRatio.Draw("SAME E0")
            padB.RedrawAxis()
            padB.RedrawAxis("G")
            canvas.Modify()
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs("%s/%03d_recoil.png" % (outDir, qTbin))
            canvas.SaveAs("%s/%03d_recoil.pdf" % (outDir, qTbin))
            ROOT.SetOwnership(dummyB, False) # deleted here, not again by python
            dummyB.Delete()
            ROOT.SetOwnership(dummyT, False) # deleted here, not again by python
            dummyT.Delete()
            ROOT.SetOwnership(padT, False) # deleted here, not again by python
            padT.Delete()
            ROOT.SetOwnership(padB, False) # deleted here, not again by python
            padB.Delete()
            g_chi2.SetPoint(qTbin, qT, chi2/ndof)
            g_pchi2.SetPoint(qTbin, qT, pval_chi2)
            g_pks.SetPoint(qTbin, qT, pval_KS)

        if doStat:
            hist_root_tot_unc_ratio.SetFillColor(18)
            hist_root_tot_unc_ratio.SetMarkerSize(0)
            hist_root_tot_unc_ratio.SetLineWidth(0)
            for i in range(1, hist_root_tot_unc_ratio.GetNbinsX()+1):
                yld_nom = hist_root_tot_unc_ratio.GetBinContent(i)
                hist_root_tot_unc_ratio.SetBinError(i, hist_root_tot_unc_ratio.GetBinError(i)/yld_nom)
                hist_root_tot_unc_ratio.SetBinContent(i, 1)


        if logY:
            cfgPlot['ymin'] = math.pow(10., math.floor(math.log10(utils.getNonZeroMinimum(hist_root_tot, xMin=xMin, xMax=xMax)))-1)
            cfgPlot['ymax'] = math.pow(10., math.ceil(math.log10(hist_root_tot.GetMaximum()))+3)
        else:
            cfgPlot['ymin'] = 0
            cfgPlot['ymax'] = 1.75*hist_root_tot.GetMaximum()
        
        # global plot
        plotter.cfg = cfgPlot
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        padT.SetTickx()
        padT.SetTicky()
        dummyT.Draw("HIST")

        hist_root_tot.SetLineColor(ROOT.kBlack)
        hist_root_tot.SetMarkerStyle(20)
        hist_root_tot.SetMarkerColor(ROOT.kBlack)
        hist_root_tot.SetLineColor(ROOT.kBlack)
        hist_root_tot.Draw("PE SAME")

        g_pdf = ROOT.TGraphErrors()
        g_pdf.SetName("g_pdf")
        g_sig_pdf.SetName("g_sig_pdf")
        g_sig_pdf = ROOT.TGraphErrors()
        for i in range(0, len(self.centers_recoil_plt)): 
            g_pdf.SetPoint(i, self.centers_recoil_plt[i], pdf_tot[i]/(self.bins_recoil_plt[i+1]-self.bins_recoil_plt[i]))
            if plotSignal:
                g_sig_pdf.SetPoint(i, self.centers_recoil_plt[i], sig_pdf_tot[i])
        g_pdf.SetLineColor(ROOT.kRed)
        g_pdf.SetLineWidth(3)

        if plotSignal:
            g_sig_pdf.SetLineColor(ROOT.kBlue)
            g_sig_pdf.SetLineWidth(3)
            g_sig_pdf.Draw("L SAME")

        g_pdf.Draw("L SAME")

        histRatio = hist_root_tot.Clone("ratio")
        histRatio.Reset("ACE")
        histRatio.SetMarkerStyle(8)
        histRatio.SetMarkerSize(0.7)
        histRatio.SetMarkerColor(ROOT.kBlack)
        histRatio.SetLineColor(ROOT.kBlack)
        histRatio.SetLineWidth(1)
        chi2, ndof, pval_chi2, pval_KS, pval_AD, histRatio = self.ratioHist_tgraph(hist_root_tot, histRatio, pdf_tot, [0]*len(pdf_tot), xMin=xMin, xMax=xMax)

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, self.procLabel)
        latex.DrawLatex(0.20, 0.80, self.metLabel)
        #latex.DrawLatex(0.20, 0.75, "#chi^{2}/ndof = %.3f/%d = %.3f" % (chi2, ndof, chi2/ndof))
        
        latex.DrawLatex(0.60, 0.85, "Fit/Cov status = %d/%d" % (fit_status, cov_status))
        latex.DrawLatex(0.60, 0.80, "#chi^{2}/ndof = %.3f/%d = %.3f" % (chi2, ndof, chi2/ndof))
        latex.DrawLatex(0.60, 0.75, "p-value #chi^{2} = %.1f %%" % pval_chi2)
        latex.DrawLatex(0.60, 0.70, "p-value KS = %.1f %%" % pval_KS)
        #latex.DrawLatex(0.60, 0.70, "p-value AD = %.1f %%" % pval_AD)

        padT.RedrawAxis()
        padT.RedrawAxis("G")
        plotter.auxRatio()
        canvas.cd()
        padB.Draw()
        padB.cd()
        padB.SetGrid()
        padB.SetTickx()
        padB.SetTicky()
        dummyB.Draw("HIST")
        if doStat:
            hist_root_tot_unc_ratio.Draw("E4 SAME")
        dummyL.Draw("SAME")
        histRatio.Draw("SAME E0")
        padB.RedrawAxis()
        padB.RedrawAxis("G")
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/global.png" % outDir)
        canvas.SaveAs("%s/global.pdf" % outDir)
        ROOT.SetOwnership(dummyB, False) # deleted here, not again by python
        dummyB.Delete()
        ROOT.SetOwnership(dummyT, False) # deleted here, not again by python
        dummyT.Delete()
        ROOT.SetOwnership(padT, False) # deleted here, not again by python
        padT.Delete()
        ROOT.SetOwnership(padB, False) # deleted here, not again by python
        padB.Delete()



       # plot the chi2
        cfg_chi2 = {

            'logy'              : False,
            'logx'              : False,

            'xmin'              : 0,
            'xmax'              : self.bins_qt[-1],
            'ymin'              : 0,
            'ymax'              : 3,

            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#chi^{2}",

            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        } 

        g_chi2.SetLineColor(ROOT.kRed)
        g_chi2.SetMarkerStyle(8)
        g_chi2.SetMarkerSize(1)
        g_chi2.SetMarkerColor(ROOT.kRed)

        plotter.cfg = cfg_chi2
        canvas = plotter.canvas()
        canvas.SetGrid()
        dummy = plotter.dummy()
        canvas.cd()
        dummy.Draw("HIST")
        g_chi2.Draw("SAME LP")
        plotter.aux()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, self.procLabel)
        latex.DrawLatex(0.20, 0.85, self.metLabel)

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/chi2.png" % outDir)
        canvas.SaveAs("%s/chi2.pdf" % outDir)
        ROOT.SetOwnership(canvas, False) # deleted here, not again by python
        canvas.Delete()
        



       # plot the p values
        cfg_chi2 = {

            'logy'              : False,
            'logx'              : False,

            'xmin'              : 0,
            'xmax'              : self.bins_qt[-1],
            'ymin'              : 0,
            'ymax'              : 130,

            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "p-value (%)",

            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        } 


        g_pchi2.SetLineColor(ROOT.kRed)
        g_pchi2.SetMarkerStyle(8)
        g_pchi2.SetMarkerSize(1)
        g_pchi2.SetLineWidth(2)
        g_pchi2.SetMarkerColor(ROOT.kRed)

        g_pks.SetLineColor(ROOT.kBlue)
        g_pks.SetMarkerStyle(8)
        g_pks.SetMarkerSize(1)
        g_pks.SetLineWidth(2)
        g_pks.SetMarkerColor(ROOT.kBlue)
        
        leg = ROOT.TLegend(.60, 0.8, .95, .92)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.03)
        #leg.SetMargin(0.1)
        #leg.AddEntry(g_pchi2, "P-value #chi^{2}", "LP")
        leg.AddEntry(g_pks, "P-value KS", "LP")

        plotter.cfg = cfg_chi2
        canvas = plotter.canvas()
        canvas.SetGrid()
        dummy = plotter.dummy()
        canvas.cd()
        dummy.Draw("HIST")
        #g_pchi2.Draw("SAME LP")
        g_pks.Draw("SAME LP")
        plotter.aux()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, self.procLabel)
        latex.DrawLatex(0.20, 0.85, self.metLabel)
        leg.Draw()
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/pval.png" % outDir)
        canvas.SaveAs("%s/pval.pdf" % outDir)
        ROOT.SetOwnership(canvas, False) # deleted here, not again by python
        canvas.Delete()