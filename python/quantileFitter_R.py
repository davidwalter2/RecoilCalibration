
import sys,array,math,os,copy,decimal,random
import numpy as np
import ctypes
import json
import time
import threading

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import utils
import plotUtils as plotter

import lz4.frame
import pickle
import narf
import pickle
import tensorflow as tf

import narf.fitutils as fitter
import hist
import boost_histogram as bh
import baseFunctions as rfb
import narf
import narf.fitutils

from scipy.optimize import curve_fit
import boostHistHelpers as bhh



np.set_printoptions(threshold=sys.maxsize)



class QuantileFitter:

    def __init__(self, procLabel, metLabel, baseDir, comp, lumiLabel, logging):
        self.procLabel = procLabel
        self.metLabel = metLabel
        self.baseDir = baseDir
        self.comp = comp
        self.lumiLabel = lumiLabel
        self.dtype = tf.float64
        self.itype = tf.int64
        self.logging = logging
        
    def setHistConfig(self, bhist, axis, qTMin, qTmax, recoilMin, recoilMax, qTRebin=1, recoilRebin=1):
        self.bhist = bhist
        self.axis = axis
        self.recoilMin, self.recoilMax = recoilMin, recoilMax
        self.qTMin, self.qTMax = qTMin, qTmax
        self.qTRebin = qTRebin
        self.recoilRebin = recoilRebin
        self.bhist = self.rebin(bhist)
        self.bins_qT = self.bhist.axes[0].edges
        self.bins_recoil = self.bhist.axes[1].edges
        self.centers_qT = self.bhist.axes[0].centers
        self.centers_recoil = self.bhist.axes[1].centers
        self.nBins_qT = len(self.bins_qT)-1
        self.bkg_cdf = None
        self.scale_sf_bkg = None
        self.scale_sf_sig = None
        self.bhist_quants = copy.deepcopy(self.bhist)
        
        self.edges_qT_tf = tf.constant(self.centers_qT, dtype=self.dtype)
        self.edges_qT_tf = self.edges_qT_tf[:, None] # col vector

    def rebin(self, bhist):
        s = hist.tag.Slicer()
        bhist = bhist[{"qTbinned": s[complex(0,self.qTMin):complex(0,self.qTMax)], self.axis: s[complex(0,self.recoilMin):complex(0,self.recoilMax)]}]
        bhist = bhh.rebinHist(bhist, self.axis, self.recoilRebin)
        bhist = bhh.rebinHist(bhist, "qTbinned", self.qTRebin)
        return bhist
        
     
    def hist_to_quantiles(self, h, quant_cdfvals, axis = -1):
        dtype = tf.float64

        xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
        xwidths = [tf.constant(width, dtype=dtype) for width in h.axes.widths]
        xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]
        yvals = tf.constant(h.values(), dtype=dtype)

        if not isinstance(quant_cdfvals, tf.Tensor):
            quant_cdfvals = tf.constant(quant_cdfvals, tf.float64)

        x_flat = tf.reshape(xedges[axis], (-1,))
        x_low = x_flat[0]
        x_high = x_flat[-1]
        

        hist_cdfvals = tf.cumsum(yvals, axis=axis)/tf.reduce_sum(yvals, axis=axis, keepdims=True)
        
        yvals_up = tf.constant(h.values() + np.sqrt(h.values()), dtype=dtype)
        hist_cdfvals_up = tf.cumsum(yvals_up, axis=axis)/tf.reduce_sum(yvals_up, axis=axis, keepdims=True)
        
        yvals_dw = tf.constant(h.values() - np.sqrt(h.values()), dtype=dtype)
        hist_cdfvals_dw = tf.cumsum(yvals_dw, axis=axis)/tf.reduce_sum(yvals_dw, axis=axis, keepdims=True)
        
  

        x0shape = list(hist_cdfvals.shape)
        x0shape[axis] = 1
        x0 = tf.zeros(x0shape, dtype = dtype)

        hist_cdfvals = tf.concat([x0, hist_cdfvals], axis=axis)
        hist_cdfvals_up = tf.concat([x0, hist_cdfvals_up], axis=axis)
        hist_cdfvals_dw = tf.concat([x0, hist_cdfvals_dw], axis=axis)

        #quants = narf.fitutils.pchip_interpolate(hist_cdfvals, xedges[axis], quant_cdfvals, axis=axis)
        quants = narf.fitutils.pchip_interpolate(xedges[axis], hist_cdfvals, quant_cdfvals, axis=axis) # quants[0] = 0 force?
        
        quants_up = narf.fitutils.pchip_interpolate(xedges[axis], hist_cdfvals_up, quant_cdfvals, axis=axis) # quants[0] = 0 force?
        quants_down = narf.fitutils.pchip_interpolate(xedges[axis], hist_cdfvals_dw, quant_cdfvals, axis=axis) # quants[0] = 0 force?
        

        '''
        quants = tf.where(quant_cdfvals == 0., x_low, quants)
        quants = tf.where(quant_cdfvals == 1., x_high, quants)

        ntot = tf.math.reduce_sum(yvals, axis=axis, keepdims=True)


        quant_cdf_bar = ntot/(1.+ntot)*(quant_cdfvals + 0.5/ntot)
        quant_cdfval_errs = ntot/(1.+ntot)*tf.math.sqrt(quant_cdfvals*(1.-quant_cdfvals)/ntot + 0.25/ntot/ntot)
        quant_cdfval_errs = ntot/(1.+ntot)*tf.math.sqrt(quant_cdfvals*(80.-quant_cdfvals)/ntot + 0.25/ntot/ntot)


        quant_cdfvals_up = quant_cdf_bar + quant_cdfval_errs
        quant_cdfvals_up = tf.clip_by_value(quant_cdfvals_up, 0., 1.)

        quant_cdfvals_down = quant_cdf_bar - quant_cdfval_errs
        quant_cdfvals_down = tf.clip_by_value(quant_cdfvals_down, 0., 1.)

        
        quants_up = narf.fitutils.pchip_interpolate(xedges[axis], hist_cdfvals, quant_cdfvals_up, axis=axis)
        
        quants_up = tf.where(quant_cdfvals_up == 0., x_low, quants_up)
        quants_up = tf.where(quant_cdfvals_up == 1., x_high, quants_up)
        
        
        quit()

        quants_down = narf.fitutils.pchip_interpolate(xedges[axis], hist_cdfvals, quant_cdfvals_down, axis=axis)
        quants_down = tf.where(quant_cdfvals_down == 0., x_low, quants_down)
        quants_down = tf.where(quant_cdfvals_down == 1., x_high, quants_down)
        '''
        quant_errs = 0.5*(quants_up - quants_down)

        zero_const = tf.constant(0., dtype)

        quant_errs = tf.where(quant_cdfvals == 0., zero_const, quant_errs)
        quant_errs = tf.where(quant_cdfvals == 1., zero_const, quant_errs)

        return quants.numpy(), quant_errs.numpy()

     
    def setQuantiles(self, quant_recvals):
        self.quant_recvals = quant_recvals
            
        self.nQuants = len(self.quant_recvals)
        self.nParams = self.nQuants-1
        
        self.logging.info(f"Quantiles used {self.nQuants}")
            
        self.quant_recvals_tf = tf.constant(self.quant_recvals, dtype=self.dtype)
        self.quant_recvals_tf = self.quant_recvals_tf[None, :]

        s = bh.tag.Slicer()
        bhist_incl = self.bhist_quants[{0: bh.sum}]
        sum_ = bhist_incl.sum().value
        self.logging.info(f"Do integral checks for each quantile (total sum of events: {sum_})")
   
        hist_quantiles, hist_quantiles_err = self.hist_to_quantiles(bhist_incl, self.quant_recvals, axis=0)
        ev_map = ROOT.TH2D("ev_map", "", len(hist_quantiles)-1, 0, len(hist_quantiles)-1, len(self.bins_qT)-1, 0, len(self.bins_qT)-1)
        #qparms, qparms_err =  narf.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax) # shape (qTbins, nQparms)
        for i in range(0, len(hist_quantiles)-1):
            print(" > Probe [{lo:.2f}, {hi:.2f}]".format(lo=hist_quantiles[i], hi=hist_quantiles[i+1]))
            try:
                h = bhist_incl[{0: s[complex(0,hist_quantiles[i]):complex(0,hist_quantiles[i+1])]}]
                print("   Events {ev:.4f}".format(ev=h.sum().value))
            except:
                print("   ERROR: no events found in bin!")
          
          
 
        # differential in qT
        hist_quantiles, hist_quantiles_err = narf.fitutils.hist_to_quantiles(self.bhist_quants, self.quant_recvals_tf, axis=1)
        for qTbin in range(0, len(self.bins_qT)-1):
            qT, qTlow, qThigh = self.centers_qT[qTbin], self.bins_qT[qTbin], self.bins_qT[qTbin+1]
            h = self.qTslice(self.bhist, qTlow, qThigh)
            hist_quantiles_ = hist_quantiles[qTbin]
            
            for i in range(0, len(hist_quantiles_)-1):
                evts = -1
                try:
                    h_ = h[{0: s[complex(0,hist_quantiles_[i]):complex(0,hist_quantiles_[i+1])]}]
                    evts = h_.sum().value
                except:
                    evts = -1
                
                ev_map.SetBinContent(i+1, qTbin+1, evts)
            
        
        c = ROOT.TCanvas("c", "", 1200, 1200)
        c.SetLogz()
        
        ev_map.GetZaxis().SetRangeUser(0.1, 1e6)
        ev_map.Draw("COL Z")
        ev_map.GetXaxis().SetTitle("Quantile bin")
        ev_map.GetYaxis().SetTitle("q_{T} bin")
        c.SaveAs(f"{self.baseDir}/ev_map.png")
        c.SaveAs(f"{self.baseDir}/ev_map.pdf")


    def setFitOrder(self, order):
        self.doSpline = False
        self.order = order
        self.chebyshev = rfb.Chebyshev(self.order, self.qTMin, self.qTMax)
        
    def setSplineVals(self, qT_vals):
        self.doSpline = True
        self.qT_vals = tf.constant(qT_vals, dtype=self.dtype)
        self.qT_vals = self.qT_vals[:, None]
        
    def setqTInterpolator(self, type_, cfg):
        if type_ == "spline":
            self.doSpline = True
            self.order = len(cfg)
            self.qT_vals = tf.constant(cfg, dtype=self.dtype)
            self.qT_vals = self.qT_vals[:, None] # col vector
            self.qT_vals_np = cfg
            self.baseFunction = rfb.Spline(self.order, self.qTMin, self.qTMax)
            self.func_model = getattr(self.baseFunction, f"cond_spline")
            self.func_constraint = getattr(self.baseFunction, f"cond_spline_constraint")
        elif type_ == "poly":
            self.doSpline = False
            self.order = cfg
            self.baseFunction = rfb.Chebyshev(self.order, self.qTMin, self.qTMax)
            self.func_model = getattr(self.baseFunction, f"cond_pol{self.order}")
            self.func_constraint = getattr(self.baseFunction, f"cond_pol{self.order}_constraint")
            
            self.qT_vals = None
        else:
            self.logging.error("Interpolator for qT should be either spline or poly.")
            quit()
        
    def subtractBackground(self, bhist, sf=1.):
        bhist = self.rebin(bhist)
        bhist_no_errors = copy.deepcopy(bhist)
        bhist_no_errors.reset()
        bkg_vals = bhist.values()
        newvals = self.bhist.view().value - bkg_vals*sf
        newvals = np.where(newvals < 0, 0, newvals) # force positive or zero bin contents
        self.bhist.view().value = newvals


    def setBackground(self, cfg, bhist_bkg, sf=1.):

        
        cfg = utils.loadJSON(cfg)
        quant_cdfvals = cfg['quant_cdfvals']
        quant_cdfvals_tf = tf.constant(quant_cdfvals, dtype=self.dtype)
        nParams = len(quant_cdfvals)-1
        order = cfg['order']
        
        func_parms = []
        for iPar in range(0, nParams):
            for ic,c in enumerate(cfg[str(iPar)]['coeff']):
                func_parms.append(c)
        tfparms = tf.constant(func_parms, dtype=self.dtype)
        
        # construct PDF
        fit_args = (quant_cdfvals_tf, None, None, None)
        xvals = [tf.constant(center, dtype=self.dtype) for center in self.bhist.axes.centers]
        xedges = [tf.constant(edge, dtype=self.dtype) for edge in self.bhist.axes.edges]
        func_model = getattr(self.chebyshev, f"cond_pol{order}")
        self.bkg_cdf = func_model(xvals, xedges, tfparms, *fit_args)

        bhist_bkg = self.rebin(bhist_bkg) * sf
        bhist_tf = tf.convert_to_tensor(bhist_bkg.to_numpy()[0])
        yields_bkg = tf.math.reduce_sum(bhist_tf, axis=1)
        
        self.bhist_quants += bhist_bkg*(-1)
        
        bhist_tf = tf.convert_to_tensor(self.bhist.to_numpy()[0])
        yields_data = tf.math.reduce_sum(bhist_tf, axis=1)
        yields_data = tf.where(tf.equal(yields_data, 0), 1, yields_data)
        
        self.scale_sf_bkg = tf.math.divide(yields_bkg, yields_data)
        self.scale_sf_sig = tf.math.divide(yields_data-yields_bkg, yields_data)

        self.scale_sf_bkg = tf.linalg.diag(self.scale_sf_bkg)
        self.scale_sf_sig = tf.linalg.diag(self.scale_sf_sig)

        
        return
        
        ########################
        pdf = cdf[:,1:] - cdf[:,:-1]
        pdf = tf.maximum(pdf, tf.zeros_like(pdf)) # PDF already normalized in each qT bin
        
        # scale PDFs with yields in each qT bin
        bhist_tf = tf.convert_to_tensor(bhist.to_numpy()[0])
        yields_ = tf.math.reduce_sum(bhist_tf, axis=1)
        yields_ = tf.where(tf.equal(yields_, 0), 1, yields_)
        pdf = pdf * yields_[:, np.newaxis] * sf
        #pdf_flatten = tf.reshape(pdf, [-1]).numpy()
        #bhist.view(flow=True)[:] = pdf_flatten
        #print(bhist.view(flow=True))
        
        
        # Convert the tensor to a numpy array
        tensor_array = pdf.numpy()

        # Flatten the tensor array
        flat_array = tensor_array.flatten()

        # Assign the flattened array to the Boost histogram bins
        bhist_new = bh.Histogram(self.bhist.axes[0], self.bhist.axes[1])
        bhist_new.view(flow=False)[:] = tensor_array
        
        print(self.bhist.sum().value)
        self.bhist.view(flow=False)[:] -= tensor_array
        #self.bhist = self.bhist - bhist_new
        print(self.bhist.sum().value)
        print(bhist_new.sum())

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
        h = bhist[{"qTbinned": s[complex(0,qTlow):complex(0,qThigh)]}]
        h = h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum  
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
        

    def parameterizeQuantiles(self, yRatio=1.08, newOrder=-1):
        outDir = f"{self.baseDir}/quantiles_param/"
        utils.mkdir(outDir, True)
                
        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]
        
        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = self.hist_to_quantiles(self.bhist_quants, self.quant_recvals_tf, axis=1) # , cdfvals_bkg=self.bkg_cdf 
        qparms, qparms_err =  narf.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax) # shape (qTbins, nQparms)
        
        
        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qT[qTbin], self.bins_qT[qTbin], self.bins_qT[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
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


        outDict = {}
        outDict['order'] = self.order
        outDict['quant_recvals'] = list(self.quant_recvals)
        if self.doSpline:
            #qT_vals = tf.constant([0, 3, 10, 20, 50, 150], dtype=self.dtype)
            # (85, 91) hist_cdfvals -> edges_qT (85, 1)
            # (1,91) xedges[axis] -> qparms (85, 41)
            # (1, 42) quant_recvals -> qT_vals in qT
            qparms_val_init = narf.fitutils.pchip_interpolate(self.edges_qT_tf, qparms, self.qT_vals, axis=0) # (nKnots, nQuants)
            qparms_init = narf.fitutils.pchip_interpolate(self.qT_vals, qparms_val_init, self.edges_qT_tf, axis=0) # to visualize spline
            qparms_init_np = qparms_init.numpy().T
            #self.qparms_val_init = qparms_val_init
            self.qparms_val_init = []
           
        else:
            self.qparms_val_init = []
            if newOrder > 0:
                if newOrder > self.order:
                    print("New order cannot be larger than default order")
                    quit()
                func = getattr(self.baseFunction, f"pol{newOrder}")
                nZeros = self.order-newOrder
                initial_parms = [1]*(newOrder+1)
            else:
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
                    ## pchip
                    tmp = qparms_val_init.numpy().T
                    for x in tmp[i]:
                        self.qparms_val_init.append(x)
                    #func_eval = qparms_init_np[i]
            
                    # cspline
                    splines_val_init = rfb.CubicSpline.cublic_spline_interpolate(x_vals, y_vals_tf[i], self.qT_vals_np)
                    func_eval = splines_val_init
            
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
                    for k,y in enumerate(y_vals_tf[i]):
                        print(x_vals[k], y, y_vals_tf_err[i][k])

                    popt, pcov, infodict, mesg, ier = curve_fit(func, xdata=x_vals, ydata=y_vals_tf[i], sigma=y_vals_tf_err[i], p0=initial_parms, full_output=True)
                    #popt, pcov, infodict, mesg, ier = curve_fit(func, xdata=x_vals, ydata=y_vals_tf[i], p0=initial_parms, full_output=True)
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

    def parameterizeQuantiles_old(self, yRatio=1.08, newOrder=-1):
        outDir = f"{self.baseDir}/quantiles_param/"
        utils.mkdir(outDir, True)
                
        x_vals = []
        y_vals = [[] for i in range(0, self.nQuants)]
        y_vals_err = [[] for i in range(0, self.nQuants)]
        y_vals_tf = [[] for i in range(0, self.nParams)]
        y_vals_tf_err = [[] for i in range(0, self.nParams)]
        
        # compute CDF vals for the backgrounds
        hist_quantiles, hist_quantiles_err = narf.fitutils.hist_to_quantiles(self.bhist_quants, self.quant_cdfvals_tf, axis=1) # , cdfvals_bkg=self.bkg_cdf 
        qparms, qparms_err =  narf.fitutils.quantiles_to_qparms(hist_quantiles, quant_errs=hist_quantiles_err, x_low=self.recoilMin, x_high=self.recoilMax)
        
        print(hist_quantiles.shape)
        print(qparms.shape) # qTbins, nQparms (85, 41)

        
        qT_vals = tf.constant([0, 5, 20, 50, 150], dtype=self.dtype)
        qT_vals = qT_vals[:, None]
        
        edges_qT = tf.constant(self.centers_qT, dtype=self.dtype) # there are len(centers_qT) qparms
        edges_qT = edges_qT[:, None] # (85, 1)
        #edges_qT = edges_qT[:, None]
        # (85, 91) hist_cdfvals -> edges_qT (85, 1)
        # (1,91) xedges[axis] -> qparms (85, 41)
        # (1, 42) quant_cdfvals -> qT_vals in qT
        # pchip_interpolate(hist_cdfvals, xedges[axis], quant_cdfvals, axis=axis)
        qparms_val = narf.fitutils.pchip_interpolate(edges_qT, qparms, qT_vals, axis=0) # (5, 41)
        

        
        # visualize spline
        qparms_init = narf.fitutils.pchip_interpolate(qT_vals, qparms_val, edges_qT, axis=0)
        

        qparms_init_np = qparms_init.numpy().T
        print(qparms_init_np)
        # func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, axis=-1, transform = None):
        #  quants = qparms_to_quantiles(qparms, x_low = x_low, x_high = x_high, axis = axis)
        
        
        quit()
        
        for qTbin,quantiles in enumerate(hist_quantiles):
            qT, qTlow, qThigh = self.centers_qT[qTbin], self.bins_qT[qTbin], self.bins_qT[qTbin+1]
            x_vals.append(qT)
            for iPar in range(0, self.nParams):
                y_vals_tf[iPar].append(qparms[qTbin][iPar])
                y_vals_tf_err[iPar].append(qparms_err[qTbin][iPar])

        cfgPlot = {

            'logy'              : False,
            'logx'              : False,
        
            'xmin'              : self.qTMin,
            'xmax'              : self.qTMax,
            'ymin'              : 0,
            'ymax'              : 1,
            
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "Recoil U_{#perp}   (GeV)" if self.comp == "perp" else "Recoil U_{#parallel} (GeV)" ,
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }


        outDict = {}
        outDict['order'] = self.order
        outDict['quant_cdfvals'] = list(self.quant_cdfvals)
        if newOrder > 0:
            if newOrder > self.order:
                print("New order cannot be larger than default order")
                quit()
            func = getattr(self.chebyshev, f"pol{newOrder}")
            nZeros = self.order-newOrder
            initial_parms = [1]*(newOrder+1)
        else:
            func = getattr(self.chebyshev, f"pol{self.order}")
            nZeros = 0
            initial_parms = [1]*(self.order+1)
        for i in range(0, self.nParams):
            g_data = self.makeTGraph(x_vals, y_vals_tf[i], y_err=y_vals_tf_err[i])

            popt, pcov, infodict, mesg, ier = curve_fit(func, xdata=x_vals, ydata=y_vals_tf[i], sigma=y_vals_tf_err[i], p0=initial_parms, full_output=True)
            errs = np.sqrt(np.diagonal(pcov))
                        
            outDict[i] = {}
            outDict[i]["order"] = self.order
            outDict[i]['coeff'] = list(popt) + [0]*nZeros
            outDict[i]['coeff_err'] = list(errs) + [0]*nZeros
            
            func_eval = func(np.array(x_vals), *popt)                
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
            
        utils.writeJSON(f"{outDir}/results_param.json", outDict)


    def plot_coeff_recoil(self):
        fIn = open(f"{self.baseDir}/quantiles_refit/results_param.pkl", "rb")
        pkl = pickle.load(fIn)
        parms_nom = pkl['x']
        
        outDir = f"{self.baseDir}/spline_coeff_vs_rec/"
        utils.mkdir(outDir, True)

        x_vals = [0.5*(self.quant_recvals[i-1]+self.quant_recvals[i]) for i in range(1, self.nQuants)] # recoil midpoints
        y_vals = [[] for g in range(0, len(self.qT_vals))]
        y_vals_err = [[] for g in range(0, len(self.qT_vals))]
        
        for iC in range(0, len(self.qT_vals)):
            for iR in range(0, len(x_vals)):
                y_vals[iC].append(parms_nom[iR+iC])
                y_vals_err[iC].append(0)


    
        yRatio = 1.08
        cfg = {
            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : min(self.quant_recvals),
            'xmax'              : max(self.quant_recvals),
            'ymin'              : -20,
            'ymax'              : 0,
                
            'xtitle'            : "Recoil value (GeV)",
            'ytitle'            : "Spline coefficient",
                
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
            canvas.SaveAs("%s/coeff%d.png" % (outDir, i))
            canvas.SaveAs("%s/coeff%d.pdf" % (outDir, i))
            canvas.Delete()                  

                    
            
    def refitQuantiles(self, newOrder=-1, withConstraint=False, parms=[]):
        outDir = f"{self.baseDir}/quantiles_refit/"
        utils.mkdir(outDir, True)
        #jsIn = utils.loadJSON(f"{self.baseDir}/quantiles_param/results_param.json")

        fit_args = (self.quant_recvals_tf, self.qT_vals, self.edges_qT_tf, self.bkg_cdf, self.scale_sf_sig, self.scale_sf_bkg)

        self.logging.info("Start refit")
       
        jsIn = {}

        parms_init = self.qparms_val_init
        if len(parms) != 0:
            parms_init = parms 

        print(" -> Start NLL fitting")
        start = time.time()
        if withConstraint:
            res = narf.fitutils.fit_hist(self.bhist, self.func_model, parms_init, mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args, func_constraint=self.func_constraint, args_constraint=fit_args)
        else:
            res = narf.fitutils.fit_hist(self.bhist, self.func_model, parms_init, mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args)
        end = time.time()
              
        parms_vals = res['x']
        fit_status = res['status']
        cov_status = res['covstatus']
        hess_eigvals = res['hess_eigvals']
        cov_matrix = res['cov']
        edmval = res['edmval']
        hess = res['hess']
        hess_eigvals_trunc = hess_eigvals[(self.order+1):] # remove first N DOF
        cov_status_trunc = 1
        if hess_eigvals_trunc[0] > 0.:
            cov_status_trunc = 0
        
        #self.logging.info("Start refitting")
        self.logging.info(" -> Fit ended, time={time:.3f} s, status={fit_status}, cov_status={cov_status}, cov_status_trunc={cov_status_trunc} edm={edm:.3e}".format(time=end-start, fit_status=fit_status, cov_status=cov_status, cov_status_trunc=cov_status_trunc, edm=edmval))
        #print(" -> Fit ended, time=%.3f s, status=%d, cov_status=%d, cov_trunc_status=%d edm=%.3e" % (end-start, fit_status, cov_status, cov_status_trunc, edmval))
        print("parms_vals")
        print(parms_vals)
        
        print("hess_eigvals")
        print(hess_eigvals)
        
        print("hess_eigvals_trunc")
        print(hess_eigvals_trunc)
        
        
        # plot non-zero eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hess)
        #np.set_printoptions(sign=‘ ‘, suppress = True, linewidth=7*11, formatter = {'float_kind':'{:.4e}'.format})
        np.set_printoptions(suppress=True)
        
        for ieig, eig in enumerate(eigenvalues):
            continue
            if eig > 0:
                continue
            vec = eigenvectors[:, ieig]
            print(f" negative eigenvalue {ieig}, {eig}, min={np.min(np.abs(vec))}, max={np.max(np.abs(vec))}")
            print(vec)
        
        with open(f"{outDir}/results_param.pkl", 'wb') as handle:
            res['qTMin'] = self.qTMin
            res['qTMax'] = self.qTMax
            res['recoilMin'] = self.recoilMin
            res['recoilMax'] = self.recoilMax
            res['bins_recoil'] = self.bins_recoil
            res['bins_qT'] = self.bins_qT
            res['centers_recoil'] = self.centers_recoil
            res['centers_qT'] = self.centers_qT
            res['quant_recvals'] = self.quant_recvals
            res['order'] = self.order
            res['nParams'] = self.nParams
            res['withConstraint'] = withConstraint
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return parms_vals
        jsOut = copy.deepcopy(jsIn)
        func_parms = []
        i=0
        for iPar in range(0, self.nParams):
            for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                jsOut[str(iPar)]['coeff'][ic] = parms_vals[i]
                #print(f"  -> iPar={iPar}, c={parms_vals[i]}")
                i+=1
                
        def unc(hess, parms_nom, order, sigma):
            ret = []
            eig_vals, eig_vec = np.linalg.eigh(hess) # sorted
            eig_vec = np.delete(eig_vec, list(range(0, order+1)), axis=1) # remove first order+1 vectors
            eig_vals = np.delete(eig_vals, list(range(0, order+1)), axis=0) # remove first order+1 vectors
            variances = np.reciprocal(np.sqrt(eig_vals)) # diag space
            for iPert, var in enumerate(variances):
                var_pert = sigma*(np.array([0. if iPert != k else val for k,val in enumerate(variances)]))
                var = eig_vec @ var_pert
                parms_pert = list(parms_nom + var)
                ret.append(parms_pert)
            return ret
                
        # uncertainties
        nStatVar = len(parms_vals) - (self.order+1)
        jsOut['nStatVar'] = nStatVar
        unc_plus = unc(hess, parms_vals, self.order, 1.)
        for iStat,parms_pert in enumerate(unc_plus):
            jsOut[f'stat{iStat}_p'] = {}
            i=0
            for iPar in range(0, self.nParams):
                jsOut[f'stat{iStat}_p'][iPar] = []
                for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                    jsOut[f'stat{iStat}_p'][iPar].append(parms_pert[i])
                    i+=1

        unc_minus = unc(hess, parms_vals, self.order, -1.)
        for iStat,parms_pert in enumerate(unc_minus):
            jsOut[f'stat{iStat}_m'] = {}
            i=0
            for iPar in range(0, self.nParams):
                jsOut[f'stat{iStat}_m'][iPar] = []
                for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                    jsOut[f'stat{iStat}_m'][iPar].append(parms_pert[i])
                    i+=1
        
        
        jsOut['qTMin'] = self.qTMin
        jsOut['qTMax'] = self.qTMax
        jsOut['recoilMin'] = self.recoilMin
        jsOut['recoilMax'] = self.recoilMax
        jsOut['recoilRebin'] = self.recoilRebin
        jsOut['qTRebin'] = self.qTRebin

        utils.writeJSON(f"{outDir}/results_param.json", jsOut)
        return parms_vals
        
    def refitQuantiles_old(self, newOrder=-1, withConstraint=False, parms=[]):
        outDir = f"{self.baseDir}/quantiles_refit/"
        utils.mkdir(outDir, True)
        jsIn = utils.loadJSON(f"{self.baseDir}/quantiles_param/results_param.json")

        quant_cdfvals_tf = tf.constant(self.quant_cdfvals, dtype=self.dtype)
        quant_cdfvals_tf = quant_cdfvals_tf[None, :]
        fit_args = (quant_cdfvals_tf, self.bkg_cdf, self.scale_sf_sig, self.scale_sf_bkg)
        func_model = getattr(self.chebyshev, f"cond_pol{self.order}")
        
        self.logging.info("Start refit")
        
        #print("Initial parameters")
        if len(parms) == 0:
            func_parms = []
            for iPar in range(0, self.nParams):
                for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                    func_parms.append(c)
                    #print(f"  -> iPar={iPar}, c={c}")
        else:
            func_parms = parms 

        print(" -> Start NLL fitting")
        start = time.time()
        if withConstraint:
            func_constraint = getattr(self.chebyshev, f"cond_pol{self.order}_constraint")
            res = narf.fitutils.fit_hist(self.bhist, func_model, np.array(func_parms), mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args, func_constraint=func_constraint, args_constraint=fit_args)
        else:
            res = narf.fitutils.fit_hist(self.bhist, func_model, np.array(func_parms), mode="nll_bin_integrated", norm_axes=[1], edmtol=0.1, args=fit_args)
        end = time.time()
              
        parms_vals = res['x']
        fit_status = res['status']
        cov_status = res['covstatus']
        hess_eigvals = res['hess_eigvals']
        cov_matrix = res['cov']
        edmval = res['edmval']
        hess = res['hess']
        hess_eigvals_trunc = hess_eigvals[(self.order+1):] # remove first N DOF
        cov_status_trunc = 1
        if hess_eigvals_trunc[0] > 0.:
            cov_status_trunc = 0
        
        #self.logging.info("Start refitting")
        print(" -> Fit ended, time=%.3f s, status=%d, cov_status=%d edm=%.3e" % (end-start, fit_status, cov_status, edmval))
        print(" -> Fit ended, time=%.3f s, status=%d, cov_status=%d edm=%.3e" % (end-start, fit_status, cov_status_trunc, edmval))
        print("parms_vals")
        print(parms_vals)
        
        print("hess_eigvals")
        print(hess_eigvals)
        
        print("hess_eigvals_trunc")
        print(hess_eigvals_trunc)
        
        with open(f"{outDir}/results_param.pkl", 'wb') as handle:
            res['qTMin'] = self.qTMin
            res['qTMax'] = self.qTMax
            res['recoilMin'] = self.recoilMin
            res['recoilMax'] = self.recoilMax
            res['bins_recoil'] = self.bins_recoil
            res['bins_qT'] = self.bins_qT
            res['centers_recoil'] = self.centers_recoil
            res['centers_qT'] = self.centers_qT
            res['quant_cdfvals'] = self.quant_cdfvals
            res['order'] = self.order
            res['nParams'] = self.nParams
            res['withConstraint'] = withConstraint
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        jsOut = copy.deepcopy(jsIn)
        func_parms = []
        i=0
        for iPar in range(0, self.nParams):
            for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                jsOut[str(iPar)]['coeff'][ic] = parms_vals[i]
                #print(f"  -> iPar={iPar}, c={parms_vals[i]}")
                i+=1
                
        def unc(hess, parms_nom, order, sigma):
            ret = []
            eig_vals, eig_vec = np.linalg.eigh(hess) # sorted
            eig_vec = np.delete(eig_vec, list(range(0, order+1)), axis=1) # remove first order+1 vectors
            eig_vals = np.delete(eig_vals, list(range(0, order+1)), axis=0) # remove first order+1 vectors
            variances = np.reciprocal(np.sqrt(eig_vals)) # diag space
            for iPert, var in enumerate(variances):
                var_pert = sigma*(np.array([0. if iPert != k else val for k,val in enumerate(variances)]))
                var = eig_vec @ var_pert
                parms_pert = list(parms_nom + var)
                ret.append(parms_pert)
            return ret
                
        # uncertainties
        nStatVar = len(parms_vals) - (self.order+1)
        jsOut['nStatVar'] = nStatVar
        unc_plus = unc(hess, parms_vals, self.order, 1.)
        for iStat,parms_pert in enumerate(unc_plus):
            jsOut[f'stat{iStat}_p'] = {}
            i=0
            for iPar in range(0, self.nParams):
                jsOut[f'stat{iStat}_p'][iPar] = []
                for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                    jsOut[f'stat{iStat}_p'][iPar].append(parms_pert[i])
                    i+=1

        unc_minus = unc(hess, parms_vals, self.order, -1.)
        for iStat,parms_pert in enumerate(unc_minus):
            jsOut[f'stat{iStat}_m'] = {}
            i=0
            for iPar in range(0, self.nParams):
                jsOut[f'stat{iStat}_m'][iPar] = []
                for ic,c in enumerate(jsIn[str(iPar)]['coeff']):
                    jsOut[f'stat{iStat}_m'][iPar].append(parms_pert[i])
                    i+=1
        
        
        jsOut['qTMin'] = self.qTMin
        jsOut['qTMax'] = self.qTMax
        jsOut['recoilMin'] = self.recoilMin
        jsOut['recoilMax'] = self.recoilMax
        jsOut['recoilRebin'] = self.recoilRebin
        jsOut['qTRebin'] = self.qTRebin

        utils.writeJSON(f"{outDir}/results_param.json", jsOut)
        return parms_vals
          
         
    def ratioHist_tgraph(self, hNom, hRatio, pdf):

        chi2 = 0
        nBins = 0
        for i in range(1, hRatio.GetNbinsX()+1):
                
            xLow , xHigh = hRatio.GetBinLowEdge(i), hRatio.GetBinLowEdge(i+1)
            pdfLow, pdfHigh = pdf.Eval(xLow), pdf.Eval(xHigh)
            pdfVal = 0.5*(pdfLow + pdfHigh)
            pdfVal = pdf.Eval(hRatio.GetBinCenter(i))
            y, y_err = 0, 0
            if(pdfVal > 0): 
                y = hNom.GetBinContent(i)/pdfVal
                y_err = hNom.GetBinError(i)/hNom.GetBinContent(i) if hNom.GetBinContent(i) > 0 else 0
            hRatio.SetBinContent(i, y)
            hRatio.SetBinError(i, y_err)
        
            if hNom.GetBinContent(i) > 0:
                chi2 += (((pdfVal - hNom.GetBinContent(i)) / hNom.GetBinError(i)) ** 2)
                nBins += 1
                
        chi2 /= nBins      
        return chi2, hRatio
        
    def pdf(self, parms, fit_args, xvals=[], edges=[]):
        tfparms = tf.constant(parms, dtype=self.dtype)
        if len(xvals) == 0 and len(edges) == 0:
            cdf = self.func_model(self.xvals_plot, self.edges_plot, tfparms, *fit_args)
        else:
            cdf = self.func_model(xvals, edges, tfparms, *fit_args)
        pdf = cdf[:,1:] - cdf[:,:-1]
        pdf = tf.maximum(pdf, tf.zeros_like(pdf))
        
        # normalize the recoil PDF in each qT bin 
        # PDF is already normalized in each qT bin
        norms = np.sum(pdf, axis=1)
        norms[norms == 0] = 1 # avoid division by zero
        pdf = pdf / norms[:, np.newaxis]
        return pdf
        
    def plot_refit_quantiles(self, prefit=False, yRatio=1.15, logY=True, yMin=-1, yMax=-1, recoilRebin=[]):
        if len(recoilRebin) > 0:
            self.bhist_plot = bhh.rebinHist(self.bhist, self.axis, recoilRebin)
        else:
            self.bhist_plot = self.bhist
        self.xvals_plot = [tf.constant(center, dtype=self.dtype) for center in self.bhist_plot.axes.centers]
        self.edges_plot = [tf.constant(edge, dtype=self.dtype) for edge in self.bhist_plot.axes.edges]
        self.bins_recoil_plot = self.bhist_plot.axes[1].edges
        self.centers_recoil_plot = self.bhist_plot.axes[1].centers

        if prefit:
            self.logging.info("Plot prefit")
            jsIn = utils.loadJSON(f"{self.baseDir}/quantiles_param/results_param.json")
            outDir = f"{self.baseDir}/quantiles_prefit"
        else:
            self.logging.info("Plot postfit")
            outDir = f"{self.baseDir}/quantiles_postfit"
            
            fIn = open(f"{self.baseDir}/quantiles_refit/results_param.pkl", "rb")
            pkl = pickle.load(fIn)
            hess = pkl['hess']
            parms_nom = pkl['x']
            
            eigvals = np.linalg.eigvalsh(hess)
            n_zeros = np.count_nonzero(eigvals < 0)
            #eigenvalues, eigenvectors = self.get_eigenvectors(hess, num_null=n_zeros)
            
            n_zeros = len(self.qT_vals)
            
            eig_vals, eig_vec = np.linalg.eigh(hess) # sorted
            eig_vec = np.delete(eig_vec, list(range(0, n_zeros)), axis=1) # remove first order+1 vectors
            eig_vals = np.delete(eig_vals, list(range(0, n_zeros)), axis=0) # remove first order+1 vectors
            variances = np.reciprocal(np.sqrt(eig_vals)) # diag space
            
            sigma = 1
            parms_unc = []
            for iPert, var in enumerate(variances):
                var_pert = sigma*(np.array([0. if iPert != k else val for k,val in enumerate(variances)]))
                var = eig_vec @ var_pert
                parms_pert = list(parms_nom + var)
                parms_unc.append(parms_pert)

        if not logY:
            outDir += "_lin"
        utils.mkdir(outDir, True)

        # get the nominal pdf
        fit_args = (self.quant_recvals_tf, self.qT_vals, self.edges_qT_tf, self.bkg_cdf, self.scale_sf_sig, self.scale_sf_bkg)
        pdf = self.pdf(parms_nom, fit_args)


        # uncertainty band (denser grid)
        pdf_stat = None
        nrecbins_stat = 500+1
        lspace = np.linspace(min(self.bins_recoil_plot), max(self.bins_recoil_plot), num=nrecbins_stat)
        xvals_plot_unc = copy.deepcopy(self.xvals_plot)
        edges_plot_unc = copy.deepcopy(self.edges_plot)
        xvals_plot_unc[1] = tf.constant((lspace[1:] + lspace[:-1]) / 2, dtype=self.dtype) # midpoints
        edges_plot_unc[1] = tf.constant(lspace, dtype=self.dtype)[None, :]
        pdf_nom_stat = self.pdf(parms_nom, fit_args, xvals=xvals_plot_unc, edges=edges_plot_unc)

        if not prefit:
            for istat, punc in enumerate(parms_unc):
                #if istat == 1:
                #    continue
                pdf_pert = self.pdf(punc, fit_args, xvals=xvals_plot_unc, edges=edges_plot_unc)
                pdf_dpert = (pdf_nom_stat - pdf_pert)**2
                if pdf_stat == None:
                    pdf_stat = pdf_dpert
                else:
                    np.add(pdf_stat, pdf_dpert)
            pdf_stat = np.sqrt(pdf_stat) / pdf_nom_stat
        
        

        cfgPlot = {

            'logy'              : logY,
            'logx'              : False,
        
            'xmin'              : self.recoilMin,
            'xmax'              : self.recoilMax,
            'ymin'              : yMin,
            'ymax'              : yMax,
            
            'xtitle'            : "Recoil U_{#perp}   (GeV)" if self.comp == "perp" else "Recoil U_{#parallel} (GeV)",
            'ytitle'            : "Events" ,
            
            'topRight'          : self.lumiLabel, 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            'yminR'             : (1-(yRatio-1)),
            'ymaxR'             : yRatio,
        }
        

            
        rebin = 1
        plotSignal = False
                   
        g_chi2 = ROOT.TGraphErrors()
        hist_root_tot = None
        pdf_tot, sig_pdf_tot = [0]*len(self.centers_recoil_plot), [0]*len(self.centers_recoil_plot)
        yields, yields_err, pdf_fracs = [], [], []
        stat_unc = [0]*len(lspace)
        yield_tot = self.bhist_plot.sum().value
        for qTbin in range(0, len(self.bins_qT)-1):
            qT, qTlow, qThigh = self.centers_qT[qTbin], self.bins_qT[qTbin], self.bins_qT[qTbin+1]
            h = self.qTslice(self.bhist_plot, qTlow, qThigh)

            #print("############# Recoil bin %d [%.2f, %.2f]" % (qTbin, qTlow, qThigh))
            yield_, yield_err = h.sum().value, math.sqrt(h.sum().variance)
            yields.append(yield_)
            yields_err.append(yield_err)

            pdf_values = pdf[qTbin,:] # is normalized (or should be)
            pdf_values *= h.sum().value/np.sum(pdf_values) # renormalize PDF in each qT bin

            if not prefit:
                pdf_values_unc = pdf_stat[qTbin,:]
                w = yield_ / yield_tot
                for k,val in enumerate(pdf_values_unc):
                    stat_unc[k] += val*val*w
            #pdf_frac_range = sum(pdf_values)
            #pdf_fracs.append(pdf_frac_range)

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
                    cfgPlot['ymin'] = math.pow(10., math.floor(math.log10(utils.getNonZeroMinimum(hist_root)))-1)
                if yMax < 0:
                    cfgPlot['ymax'] = math.pow(10., math.ceil(math.log10(hist_root.GetMaximum()))+3)
            else:
                cfgPlot['ymin'] = 0
                cfgPlot['ymax'] = 1.75*hist_root.GetMaximum()

            if not prefit and False:
                hist_root_tot_unc_ = hist_root.Clone("hist_root_tot_unc_%d"%qTbin)
                hist_root_tot_unc_.SetFillColor(18)
                hist_root_tot_unc_.SetMarkerSize(0)
                hist_root_tot_unc_.SetLineWidth(0)
                hist_root_tot_unc_.Reset("ICE")
                hist_root_tot_unc_.ResetStats() 
                
                hist_root_tot_unc_ratio_ = hist_root.Clone("hist_root_tot_unc_ratio_%d"%qTbin)
                hist_root_tot_unc_ratio_.SetFillColor(18)
                hist_root_tot_unc_ratio_.SetMarkerSize(0)
                hist_root_tot_unc_ratio_.SetLineWidth(0)
                hist_root_tot_unc_ratio_.Reset("ICE") # reset errors
                hist_root_tot_unc_ratio_.ResetStats() 
                
                pdf_pert_tot = None
                for nStat in range(0, nStatVars):
                
                    pdf_values = tf.math.pow(pdf_stat[nStat][qTbin,:], 2)
                    if pdf_pert_tot == None:
                        pdf_pert_tot = pdf_values
                    else:
                        pdf_pert_tot += pdf_values
                
                
                quit()
                if True:
                    ### PLUS VARIATION
                    pdf_tot_var_up = [0]*len(centers_recoil)
                    nStatName = "stat%d_p" % nStat
                    func_parms_p = []
                    for k in range(0, fitCfg['nParams']):
                        for l in range(0, fitCfg["p%d"%k]['nParams']):
                            func_parms_p.append(fitCfg[nStatName]["p%d"%k]["p%d"%l])
                           
                    if average_eval_PDF:
                        pdf_values_var_bins = func_model([qT_tf, bins_recoil_tf], func_parms_p, *args).numpy()
                        pdf_values_var = [0.5*(pdf_values_var_bins[i-1]+pdf_values_var_bins[i]) for i in range(1, len(pdf_values_var_bins))]
                    else:
                        pdf_values_var = func_model([qT_tf, centers_recoil_tf], func_parms_p, *args).numpy()

                    for i in range(0, len(centers_recoil)): 
                        pdf_tot_var_up[i] += pdf_values_var[i]*yields[qTbin-1]/pdf_fracs[qTbin-1] # iBin starts from 1
                             
                    
                    ### MINUS VARIATION
                    pdf_tot_var_dw = [0]*len(centers_recoil)
                    nStatName = "stat%d_m" % nStat
                    func_parms_m = []
                    for k in range(0, fitCfg['nParams']):
                        for l in range(0, fitCfg["p%d"%k]['nParams']):
                            func_parms_m.append(fitCfg[nStatName]["p%d"%k]["p%d"%l])
                     
                    if average_eval_PDF:
                        pdf_values_var_bins = func_model([qT_tf, bins_recoil_tf], func_parms_m, *args).numpy()
                        pdf_values_var = [0.5*(pdf_values_var_bins[i-1]+pdf_values_var_bins[i]) for i in range(1, len(pdf_values_var_bins))]
                    else:
                        pdf_values_var = func_model([qT_tf, centers_recoil_tf], func_parms_m, *args).numpy()

                    for i in range(0, len(centers_recoil)): 
                        pdf_tot_var_dw[i] += pdf_values_var[i]*yields[qTbin-1]/pdf_fracs[qTbin-1] # iBin starts from 1     
                    
                    # make average of plus and minus variation  
                    for i in range(1, hist_root_tot_unc_ratio_.GetNbinsX()+1):
                        pdf_tot_eval = pdf_values[i-1]*yield_/pdf_frac_range
                        if pdf_tot_eval > 0:
                            err_up = abs(pdf_tot_var_up[i-1] - pdf_tot_eval)
                            err_dw = abs(pdf_tot_eval - pdf_tot_var_dw[i-1])
                            err = abs(0.5*(err_up+err_dw))
                        else:
                            err = 0
                        hist_root_tot_unc_ratio_.SetBinError(i, hist_root_tot_unc_ratio_.GetBinError(i) + err*err) # add in quadrature
                        hist_root_tot_unc_.SetBinError(i, hist_root_tot_unc_.GetBinError(i) + err*err)

                for i in range(1, hist_root_tot_unc_ratio_.GetNbinsX()+1): 
                    pdf_tot_eval = pdf_values[i-1]*yield_/pdf_frac_range
                    if pdf_tot_eval > 0: 
                        hist_root_tot_unc_ratio_.SetBinError(i, (hist_root_tot_unc_ratio_.GetBinError(i)**0.5)/pdf_tot_eval)
                        hist_root_tot_unc_.SetBinError(i, hist_root_tot_unc_.GetBinError(i)**0.5)
                    hist_root_tot_unc_ratio_.SetBinContent(i, 1)
                    hist_root_tot_unc_.SetBinContent(i, pdf_tot_eval)
     

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
            
            hist_root.Draw("PE SAME")
            

            yield_ /= rebin # reduce yield to accomodate for binning
            g_pdf = ROOT.TGraphErrors()
            g_sig_pdf = ROOT.TGraphErrors()
            for i in range(0, len(self.centers_recoil_plot)): 
                #g_pdf.SetPoint(i, self.centers_recoil[i], pdf_values[i]*yield_/pdf_frac_range)
                g_pdf.SetPoint(i, self.centers_recoil_plot[i], pdf_values[i]/(self.bins_recoil_plot[i+1]-self.bins_recoil_plot[i]))
                pdf_tot[i] += pdf_values[i]/(self.bins_recoil_plot[i+1]-self.bins_recoil_plot[i]) #*yield_ #/pdf_frac_range

            
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
            chi2, histRatio = self.ratioHist_tgraph(hist_root, histRatio, g_pdf)
            

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
            latex.DrawLatex(0.20, 0.55, "#chi^{2}/ndof = %.3f" % chi2)

            
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
            dummyL.Draw("SAME")
            histRatio.Draw("SAME E0")
            padB.RedrawAxis()
            padB.RedrawAxis("G")
            canvas.Modify()
            canvas.Update()
            canvas.Draw()
            canvas.SaveAs("%s/%03d_recoil.png" % (outDir, qTbin))
            canvas.SaveAs("%s/%03d_recoil.pdf" % (outDir, qTbin))
            dummyB.Delete()
            dummyT.Delete()
            padT.Delete()
            padB.Delete()
            #g_chi2.SetPoint(iBin-1, qT, chi2)
            

        
        if not prefit:
            hist_root_tot_unc_ratio = ROOT.TH1D("hist_root_tot_unc_ratio", "", nrecbins_stat, min(self.bins_recoil_plot), max(self.bins_recoil_plot))
            hist_root_tot_unc_ratio.SetFillColor(18)
            hist_root_tot_unc_ratio.SetMarkerSize(0)
            hist_root_tot_unc_ratio.SetLineWidth(0)
            for i in range(1, hist_root_tot_unc_ratio.GetNbinsX()+1):
                hist_root_tot_unc_ratio.SetBinError(i, stat_unc[i-1]**0.5)
                hist_root_tot_unc_ratio.SetBinContent(i, 1)
            




        if logY:
            cfgPlot['ymin'] = math.pow(10., math.floor(math.log10(utils.getNonZeroMinimum(hist_root_tot)))-1)
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
        for i in range(0, len(self.centers_recoil_plot)): 
            g_pdf.SetPoint(i, self.centers_recoil_plot[i], pdf_tot[i])
        #    if plotSignal:
        #        g_sig_pdf.SetPoint(i, centers_recoil_plot[i], sig_pdf_tot[i])
        g_pdf.SetLineColor(ROOT.kRed)
        g_pdf.SetLineWidth(3)
        
        #if plotSignal:
        #    g_sig_pdf.SetLineColor(ROOT.kBlue)
        #    g_sig_pdf.SetLineWidth(3)
        #    g_sig_pdf.Draw("L SAME")
        
        g_pdf.Draw("L SAME")
            
        histRatio = hist_root_tot.Clone("ratio")
        histRatio.Reset("ACE")
        histRatio.SetMarkerStyle(8)
        histRatio.SetMarkerSize(0.7)
        histRatio.SetMarkerColor(ROOT.kBlack)
        histRatio.SetLineColor(ROOT.kBlack)
        histRatio.SetLineWidth(1)
        chi2, histRatio = self.ratioHist_tgraph(hist_root_tot, histRatio, g_pdf)
            

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, self.procLabel)
        latex.DrawLatex(0.20, 0.80, self.metLabel)
        latex.DrawLatex(0.20, 0.75, "#chi^{2}/ndof = %.3f" % chi2)

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
        if not prefit:
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
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        #fOut = ROOT.TFile("%s/global.root" % outDir, "RECREATE")
        #histRatio.Write()
        #hist_root_tot.Write()
        #g_pdf.Write()
        #fOut.Close()

    def exportModel(self, mc_in, data_in, out):

        qTMin = tf.constant(0.0, tf.float64)
        qTMax = tf.constant(150.0, tf.float64)
        recoilMin = tf.constant(-100., tf.float64)
        recoilMax = tf.constant(100., tf.float64)
        
        # cdf data
        
        # cdf mc
        handle_mc = open(mc_in, 'rb')
        res_mc = pickle.load(handle)
            
        quant_cdfvals_tf = tf.constant(res_mc['quant_cdfvals'], dtype=self.dtype)
        quant_cdfvals_tf = quant_cdfvals_tf[None, :]
        tfparms = res_mc['x']
        qTMin_mc = jsOut['qTMin'] 
        qTMax_mc = jsOut['qTMax']
        recoilMin_mc = jsOut['recoilMin']
        recoilMax_mc = jsOut['recoilMax']
        fit_args_mc = (quant_cdfvals_tf, res_mc['nParams'], len(res_mc['qTRebin']), False)
        func_model_mc = getattr(self.chebyshev, f"cond_pol{res_mc['order']}")
        
        def func_cdf_mc(pt, ut):
            # protection
            if pt > qTMax_mc:
                pt = qTMax_mc
            if pt < qTMin_mc:
                pt = qTMin_mc
            if ut > recoilMax_mc:
                ut = recoilMax_mc
            if ut < recoilMin_mc:
                ut = recoilMin_mc
                
            pts = tf.reshape(pt, (1,1))
            uts = tf.reshape(ut, (1,1))

            xvals = [pts, None]
            xedges = [None, uts]
            func_cdf_mc = func_model_mc(xvals, xedges, tfparms, *fit_args_mc)
        
        
        # eig vals data
        
        # eig vals mc
        

    
        def get_scaled_eigenvectors(hess, num_null = 0):
            e,v = np.linalg.eigh(hess)

            # remove the null eigenvectors
            e = e[None, num_null:]
            v = v[:, num_null:]

            # scale the eigenvectors
            vscaled = v/np.sqrt(e)

            return vscaled
            
            
            
        def transform_mc_simple(pt, ut):   
            cdf_mc = func_cdf_mc(pt, ut)
            ut_transformed, pdf = func_cdfinv_pdf_data(pt, cdf_mc)
            ut_transformed = tf.reshape(ut_transformed, [])

            return ut_transformed

        scalar_spec = tf.TensorSpec([], tf.float64)

        class TestMod(tf.Module):
            @tf.function(input_signature =  [scalar_spec, scalar_spec])
            def __call__(self, pt, ut):
                return transform_mc_simple(pt, ut)
                
        module = TestMod()
        concrete_function = module.__call__.get_concrete_function()

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function], module)
        converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
          tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]

        tflite_model = converter.convert()

        print(tflite_model)
        with open(out, 'wb') as f:
          f.write(tflite_model)