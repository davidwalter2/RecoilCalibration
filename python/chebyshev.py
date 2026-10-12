
import math
import tensorflow as tf
import narf
import wums.fitutils
import wums.tfutils
import numpy as np


class Chebyshev:

    def __init__(self, order, qTmin, qTmax):
        self.order = order
        self.min_ = tf.constant(qTmin, dtype=tf.float64)
        self.max_ = tf.constant(qTmax, dtype=tf.float64)

    def tf(self, qT):
        return ((qT-self.min_)-(self.max_-qT))/(self.max_-self.min_)

    def pol0(self, qT, p0, p1, p2):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.)   

    def pol1(self, qT, p0, p1):
        qT_ = self.tf(qT)
        return p0 + p1*qT

    def pol2(self, qT, p0, p1, p2):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.)

    def pol3(self, qT, p0, p1, p2, p3):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.) + p3*(4.*tf.math.pow(qT_,3)-3.*qT_)

    def pol4(self, qT, p0, p1, p2, p3, p4):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.) + p3*(4.*tf.math.pow(qT_,3)-3.*qT_) + p4*(8.*tf.math.pow(qT_,4)-8.*tf.math.pow(qT_,2)+1.)

    def pol4_prime(self, qT, p0, p1, p2, p3, p4):
        qT_ = self.tf(qT)
        return p1 + p2*4.*qT_ + p3*(12.*tf.math.pow(qT_,2)-3.) + p4*(32.*tf.math.pow(qT_,3)-16.*qT_)

    def pol4_lin(self, qT, p0, p1, p2, p3, p4):
        qB = tf.constant(100, dtype=tf.float64)
        return self.pol4_prime(qB, p0, p1, p2, p3, p4)*(qT-qB) + self.pol4(qB, p0, p1, p2, p3, p4)

    def pol4_pw(self, qT, p0, p1, p2, p3, p4):
        qB = tf.constant([100], dtype=tf.float64)
        if tf.is_tensor(qT):
            qT__ = qT
        else: 
            qT__ = tf.constant(qT, dtype=tf.float64)
        return tf.where(tf.math.greater(qT__, qB), self.pol4_lin(qT, p0, p1, p2, p3, p4), self.pol4(qT, p0, p1, p2, p3, p4))

    def pol5(self, qT, p0, p1, p2, p3, p4, p5):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.) + p3*(4.*tf.math.pow(qT_,3)-3.*qT_) + p4*(8.*tf.math.pow(qT_,4)-8.*tf.math.pow(qT_,2)+1.) + p5*(16.*tf.math.pow(qT_,5)-20.*tf.math.pow(qT_,3)+5.*qT_)
        
    
    def pol6(self, qT, p0, p1, p2, p3, p4, p5, p6):
        qT_ = self.tf(qT)
        return p0 + p1*qT_ + p2*(2.*tf.math.pow(qT_,2)-1.) + p3*(4.*tf.math.pow(qT_,3)-3.*qT_) + p4*(8.*tf.math.pow(qT_,4)-8.*tf.math.pow(qT_,2)+1.) + p5*(16.*tf.math.pow(qT_,5)-20.*tf.math.pow(qT_,3)+5.*qT_) + p6*(32.*tf.math.pow(qT_,6)-48.*tf.math.pow(qT_,4)+18.*tf.math.pow(qT_,2)-1)
        
        
    
    def func_constraint_for_quantile_fit(self, xvals, xedges, qparms, quant_cdfvals, axis=-1):
        # constraint on the sum
        constraints = 0.5*tf.math.square(tf.math.reduce_sum(tf.exp(qparms), axis=axis) - 1.) # require sum to be zero
        constraint = tf.math.reduce_sum(constraints)
       
        #deltax = tf.exp(qparms)
        #constraints = 0.2*tf.math.square(deltax - tf.experimental.numpy.diff(quant_cdfvals))
        #constraint = 0.5*tf.math.reduce_sum(constraints) # 0.2 def

        #deltax = tf.exp(qparms)
        #deltas = tf.experimental.numpy.diff(quant_cdfvals)
        #constraints = 0.5/deltas*tf.math.square(deltax - deltas) # or divide
        #constraint = 1e4*tf.math.reduce_sum(constraints) # 0.2 def # start without first to see the scale

        return constraint
     
     
    def func_transform_cdf(self, quantile):
        # cauchy
        const_pi = tf.constant(math.pi, quantile.dtype)
        return tf.math.atan(quantile)/const_pi + 0.5
        
        # sigmoid
        return tf.math.sigmoid(quantile)
        
        # gauss
        const_sqrt2 = tf.constant(math.sqrt(2.), quantile.dtype)
        return 0.5*(1. + tf.math.erf(quantile/const_sqrt2))

    def func_transform_quantile(self, cdf):
        # cauchy
        const_pi = tf.constant(math.pi, cdf.dtype)
        return tf.math.tan(const_pi*(cdf - 0.5))
    
        # sigmoid
        return tf.math.log(cdf/(1.-cdf))
        
        # gauss
        const_sqrt2 = tf.constant(math.sqrt(2.), cdf.dtype)
        const_2 = tf.constant(2, cdf.dtype)
        const_1 = tf.constant(1, cdf.dtype)
        return const_sqrt2*tf.math.erfinv(const_2*cdf - const_1)

    def cond_pol1(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        bkg_cdf = args[1]
        scale_sf_sig = args[2]
        scale_sf_bkg = args[3]

        parms_2d = tf.reshape(parms, (-1, 2))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        parms_ = self.pol1(xvals[0], p0, p1)
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig
    
    def cond_pol1_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        
        parms_2d = tf.reshape(parms, (-1, 2))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        parms_ = self.pol2(xvals[0], p0, p1)
        return wums.fitutils.func_constraint_for_quantile_fit(xvals, xedges, parms_, axis=1)

    def cond_pol2(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        
        bkg_cdf = args[4]
        scale_sf_sig = args[5]
        scale_sf_bkg = args[6]
        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 3))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]

        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        #lin12 = lin1_2d[:,2]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        #lin22 = lin2_2d[:,2]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol2(xvals[0], p0, p1, p2) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig

    def cond_pol2_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 3))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]

        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        #lin12 = lin1_2d[:,2]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        #lin22 = lin2_2d[:,2]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol2(xvals[0], p0, p1, p2) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        return wums.fitutils.func_constraint_for_quantile_fit(xvals, xedges, parms_, axis=1)

    def cond_pol3(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        
        bkg_cdf = args[4]
        scale_sf_sig = args[5]
        scale_sf_bkg = args[6]
        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 4))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol3(xvals[0], p0, p1, p2, p3) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig

    def cond_pol3_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]

        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 4))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol3(xvals[0], p0, p1, p2, p3) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        return wums.fitutils.func_constraint_for_quantile_fit(xvals, xedges, parms_, axis=1)

      
    def cond_pol4(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        
        bkg_cdf = args[4]
        scale_sf_sig = args[5]
        scale_sf_bkg = args[6]
        
        '''
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 5))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol4(xvals[0], p0, p1, p2, p3, p4) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        '''
        

        parms_2d = tf.reshape(parms, (-1, 5))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        parms_ = self.pol4(xvals[0], p0, p1, p2, p3, p4) 
  
        
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig

    def cond_pol4_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]

        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 5))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol4(xvals[0], p0, p1, p2, p3, p4) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        return self.func_constraint_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1)
        #return wums.fitutils.func_constraint_for_quantile_fit(xvals, xedges, parms_, axis=1)
      
    def cond_pol5(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        
        bkg_cdf = args[4]
        scale_sf_sig = args[5]
        scale_sf_bkg = args[6]
        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 6))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        p5 = parms_2d[:,5]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol5(xvals[0], p0, p1, p2, p3, p4, p5) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        
        
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig
    
    def cond_pol5_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]

        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 6))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        p5 = parms_2d[:,5]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol5(xvals[0], p0, p1, p2, p3, p4, p5) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        return self.func_constraint_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1)
          

    def cond_pol6(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]
        qT_vals = args[1]
        edges_qT = args[2]
        order = args[3]
        
        bkg_cdf = args[4]
        scale_sf_sig = args[5]
        scale_sf_bkg = args[6]
        
        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 7))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        p5 = parms_2d[:,5]
        p6 = parms_2d[:,6]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol6(xvals[0], p0, p1, p2, p3, p4, p5, p6) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        
        
        cdf_sig = wums.fitutils.func_cdf_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1) # , transform = (self.func_transform_cdf, self.func_transform_quantile)
        if bkg_cdf != None:
            return tf.matmul(scale_sf_sig, cdf_sig) + tf.matmul(scale_sf_bkg, bkg_cdf)
        else:
            return cdf_sig
    
    def cond_pol6_constraint(self, xvals, xedges, parms, *args):
        quant_cdfvals = args[0]

        nlin = 0
        nprams_tot = parms.shape[0]
        nparms_lin = 2*nlin
        nparms_poly = nprams_tot - nparms_lin
        
        lin1 = parms[0:nparms_lin]
        poly = parms[nparms_lin:nparms_poly]
        lin2 = parms[nprams_tot-nparms_lin:nprams_tot]

        lin1_2d = tf.reshape(lin1, (-1, 2))
        lin2_2d = tf.reshape(lin2, (-1, 2))
        
        parms_2d = tf.reshape(poly, (-1, 7))
        p0 = parms_2d[:,0]
        p1 = parms_2d[:,1]
        p2 = parms_2d[:,2]
        p3 = parms_2d[:,3]
        p4 = parms_2d[:,4]
        p5 = parms_2d[:,5]
        p6 = parms_2d[:,6]
        
        lin10 = lin1_2d[:,0]
        lin11 = lin1_2d[:,1]
        
        lin20 = lin2_2d[:,0]
        lin21 = lin2_2d[:,1]
        
        parms_lin1 = self.pol1(xvals[0], lin10, lin11) 
        parms_poly = self.pol6(xvals[0], p0, p1, p2, p3, p4, p5, p6) 
        parms_lin2 = self.pol1(xvals[0], lin20, lin21)
        parms_ = tf.concat([parms_lin1, parms_poly, parms_lin2], 1)
        
        return self.func_constraint_for_quantile_fit(xvals, xedges, parms_, quant_cdfvals, axis=1)
          
