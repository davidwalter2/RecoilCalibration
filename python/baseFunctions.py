
import math
import tensorflow as tf
import narf
import wums.fitutils
import wums.tfutils
import numpy as np




class CubicSpline:

    def __init__(self):
        pass


    def func_constraint_for_quantile_fit(self, xvals, xedges, qparms, quant_cdfvals, axis=-1):
        # constraint on the sum
        constraints = 0.5*tf.math.square(tf.math.reduce_sum(tf.exp(qparms), axis=axis) - 1.) # require sum to be zero
        constraint = tf.math.reduce_sum(constraints)

        #deltax = tf.exp(qparms)
        #constraints = 10*tf.math.square(deltax - tf.experimental.numpy.diff(quant_cdfvals))
        #constraint = 0.5*tf.math.reduce_sum(constraints) # 0.2 def

        #deltax = tf.exp(qparms)
        #deltas = tf.experimental.numpy.diff(quant_cdfvals)
        #constraints = 0.5/deltas*tf.math.square(deltax - deltas) # or divide
        #constraint = 1e4*tf.math.reduce_sum(constraints) # 0.2 def # start without first to see the scale

        return constraint

    def cond_spline(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        extrpl = args[4]
        bkg_cdf = args[5]
        scale_sf_sig = args[6]
        scale_sf_bkg = args[7]

        # unpack parms_postfit to (nKnots, nQuants)
        parms_2d = tf.reshape(parms, (-1, order))
        parms_2d = tf.transpose(parms_2d)
        parms_ = wums.fitutils.cubic_spline_interpolate(qT_vals, parms_2d, edges_qT, axis=0, extrpl=extrpl)

        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig

    def cond_spline_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        extrpl = args[4]
        bkg_cdf = args[5]
        scale_sf_sig = args[6]
        scale_sf_bkg = args[7]

        # unpack parms_postfit to (nKnots, nQuants)
        parms_2d = tf.reshape(parms, (-1, order))
        parms_2d = tf.transpose(parms_2d)
        parms_ = wums.fitutils.cubic_spline_interpolate(qT_vals, parms_2d, edges_qT, axis=0, extrpl=extrpl)
        return self.func_constraint_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1)

