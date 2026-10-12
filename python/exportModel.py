
import sys,array,math,os
import pickle
import numpy as np
import tensorflow as tf
import narf
import narf.fitutils
import narf.tfutils


class Export:

    scalar_spec = tf.TensorSpec([], tf.float64)

    def __init__(self):
    
        self.funcs = []
        self.func_names = []
        self.func_inputs = []
        self.nstat = 0
        self.nparms_para, self.nparms_perp = 0, 0
        

    def get_scaled_eigenvectors(self, hess, num_null = 2):
        e,v = np.linalg.eigh(hess)

        # remove the null eigenvectors
        e = e[None, num_null:]
        v = v[:, num_null:]

        # scale the eigenvectors
        vscaled = v/np.sqrt(e)
        return vscaled

    def get_invcdf(self, input_file):

        ut_axis = 1
        qt_axis = 0

        handle = open(input_file, 'rb')
        res = pickle.load(handle)
        quant_cdfvals = res['quant_cdfvals_tf']
        knots_qt = res['knots_qt']
        order = res['order']
        extrpl = res['extrpl']
        parms = tf.constant(res['x'], tf.float64)
        vscaled = tf.constant(self.get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
        ut_low = tf.cast(res['recoilMin'], tf.float64)
        ut_high = tf.cast(res['recoilMax'], tf.float64)
        def func_invcdf(pt, quant):
            pts = tf.reshape(pt, (1,1))
            quant_outs = tf.reshape(quant, (1,1))

            parms_2d = tf.reshape(parms, (-1, order))
            parms_2d = tf.transpose(parms_2d)
            qparms = narf.fitutils.cubic_spline_interpolate(knots_qt, parms_2d, pts, axis=qt_axis, extrpl=extrpl)
            quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low, x_high=ut_high, axis=ut_axis)
            cdfinvvals = narf.fitutils.pchip_interpolate(quant_cdfvals, quants, quant_outs, axis=ut_axis)
            return cdfinvvals

        return parms, vscaled, func_invcdf

    def get_cdf(self, input_file):

        ut_axis = 1
        qt_axis = 0

        handle = open(input_file, 'rb')
        res = pickle.load(handle)
        quant_cdfvals = res['quant_cdfvals_tf']
        knots_qt = res['knots_qt']
        order = res['order']
        extrpl = res['extrpl']
        parms = tf.constant(res['x'], tf.float64)
        vscaled = tf.constant(self.get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
        ut_low = tf.cast(res['recoilMin'], tf.float64)
        ut_high = tf.cast(res['recoilMax'], tf.float64)
        def func_cdf(pt, ut):
            pts = tf.reshape(pt, (1,1))
            uts = tf.reshape(ut, (1,1))

            parms_2d = tf.reshape(parms, (-1, order))
            parms_2d = tf.transpose(parms_2d)
            qparms = narf.fitutils.cubic_spline_interpolate(knots_qt, parms_2d, pts, axis=qt_axis, extrpl=extrpl)
            quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low, x_high=ut_high, axis=ut_axis)
            cdfvals = narf.fitutils.pchip_interpolate(quants, quant_cdfvals, uts, axis=ut_axis)
            return cdfvals

        return parms, vscaled, func_cdf

    def set_limits(self, pt_max, ut_min, ut_max):
        self.ut_min = tf.cast(ut_min, tf.float64)
        self.ut_max = tf.cast(ut_max, tf.float64)
        self.ut_min_para = tf.cast(ut_min, tf.float64)
        self.ut_max_para = tf.cast(ut_max, tf.float64)
        self.ut_min_perp = tf.cast(ut_min, tf.float64)
        self.ut_max_perp = tf.cast(ut_max, tf.float64)
        self.pt_max = tf.cast(pt_max, tf.float64)

    def set_limits_para(self, ut_min, ut_max):
        self.ut_min_para = tf.cast(ut_min, tf.float64)
        self.ut_max_para = tf.cast(ut_max, tf.float64)

    def set_limits_perp(self, ut_min, ut_max):
        self.ut_min_perp = tf.cast(ut_min, tf.float64)
        self.ut_max_perp = tf.cast(ut_max, tf.float64)

    def add_base_transform(self, source_para, target_para, source_perp, target_perp):

        parms_target_para, vscaled_target_para, func_cdfinv_target_para = self.get_invcdf(target_para)
        parms_source_para, vscaled_source_para, func_cdf_source_para = self.get_cdf(source_para)

        parms_target_perp, vscaled_target_perp, func_cdfinv_target_perp = self.get_invcdf(target_perp)
        parms_source_perp, vscaled_source_perp, func_cdf_source_perp = self.get_cdf(source_perp)

        self.nstat = parms_target_para.shape[0] + parms_source_para.shape[0] + parms_target_perp.shape[0] + parms_source_perp.shape[0]

        def base_transform(pt, ut_para, ut_perp):
            ret_para, ret_grad_para = self.add_transform_single(parms_source_para, vscaled_source_para, func_cdf_source_para, parms_target_para, vscaled_target_para, func_cdfinv_target_para, self.ut_min_para, self.ut_max_para)(pt, ut_para)
            ret_perp, ret_grad_perp = self.add_transform_single(parms_source_perp, vscaled_source_perp, func_cdf_source_perp, parms_target_perp, vscaled_target_perp, func_cdfinv_target_perp, self.ut_min_perp, self.ut_max_perp)(pt, ut_perp)
            ret_grad = tf.concat([ret_grad_para, ret_grad_perp], 0)
            return ret_para, ret_perp, ret_grad

        self.funcs.append(base_transform)
        self.func_names.append(f"base_transform")
        self.func_inputs.append([self.scalar_spec, self.scalar_spec, self.scalar_spec])
        
        def base_transform_no_unc(pt, ut_para, ut_perp):
            ret_para = self.add_transform_single_no_unc(func_cdf_source_para, func_cdfinv_target_para, self.ut_min_para, self.ut_max_para)(pt, ut_para)
            ret_perp = self.add_transform_single_no_unc(func_cdf_source_perp, func_cdfinv_target_perp, self.ut_min_perp, self.ut_max_perp)(pt, ut_perp)
            return ret_para, ret_perp

        self.funcs.append(base_transform_no_unc)
        self.func_names.append(f"base_transform_no_unc")
        self.func_inputs.append([self.scalar_spec, self.scalar_spec, self.scalar_spec])

    def add_transform_single(self, parms_source, vscaled_source, func_cdf_source, parms_target, vscaled_target, func_cdfinv_target, ut_min, ut_max):
 
        nparms = parms_source.shape[0] + parms_target.shape[0]
        def func_cdfinv_pdf_target(pt, quant):
            with tf.GradientTape() as t:
                t.watch(quant)
                cdfinv = func_cdfinv_target(pt, quant)
            pdfreciprocal = t.gradient(cdfinv, quant)
            pdf = 1./pdfreciprocal
            return cdfinv, pdf

        def transform(pt, ut):
            pt = tf.where(tf.math.greater(pt, self.pt_max), self.pt_max, pt)
            #ut = tf.where(tf.math.greater(ut, ut_max), ut_max, ut)
            #ut = tf.where(tf.math.less(ut, ut_min), ut_min, ut)
            with tf.GradientTape(persistent=True) as t:
                t.watch(parms_source)
                t.watch(parms_target)

                cdf_source = func_cdf_source(pt, ut)
                ut_transformed, pdf = func_cdfinv_pdf_target(pt, cdf_source)

                ut_transformed = tf.reshape(ut_transformed, [])
                pdf = tf.reshape(pdf, [])

            pdf_grad_source = t.gradient(pdf, parms_source)
            pdf_grad_target = t.gradient(pdf, parms_target)

            del t

            weight_grad_source = pdf_grad_source/pdf
            weight_grad_target = pdf_grad_target/pdf

            weight_grad_source = weight_grad_source[None, :]
            weight_grad_target = weight_grad_target[None, :]

            weight_grad_mc_eig = weight_grad_source @ vscaled_source
            weight_grad_data_eig = weight_grad_target @ vscaled_target

            weight_grad_mc_eig = tf.reshape(weight_grad_mc_eig, [-1])
            weight_grad_data_eig = tf.reshape(weight_grad_data_eig, [-1])

            weight_grad_eig = tf.concat([weight_grad_mc_eig, weight_grad_data_eig], axis=0) + 1.0
            weight_grad_eig_nom = tf.constant([1.]*nparms, dtype=tf.float64)

            ret = tf.where(tf.math.logical_or(tf.math.greater_equal(ut, self.ut_max), tf.math.less_equal(ut, self.ut_min)), ut, ut_transformed)
            ret_grad = tf.where(tf.math.logical_or(tf.math.greater_equal(ut, self.ut_max), tf.math.less_equal(ut, self.ut_min)), weight_grad_eig_nom, weight_grad_eig)
            return ret, ret_grad
        
        return transform

    def add_transform_single_no_unc(self, func_cdf_source, func_cdfinv_target, ut_min, ut_max):

        def func_cdfinv_pdf_target(pt, quant):
            with tf.GradientTape() as t:
                t.watch(quant)
                cdfinv = func_cdfinv_target(pt, quant)
            pdfreciprocal = t.gradient(cdfinv, quant)
            pdf = 1./pdfreciprocal
            return cdfinv, pdf

        def transform(pt, ut):
            pt = tf.where(tf.math.greater(pt, self.pt_max), self.pt_max, pt)
            cdf_mc = func_cdf_source(pt, ut)
            ut_transformed, pdf = func_cdfinv_pdf_target(pt, cdf_mc)
            ut_transformed = tf.reshape(ut_transformed, [])
            return tf.where(tf.math.logical_or(tf.math.greater_equal(ut, ut_max), tf.math.less_equal(ut, ut_min)), ut, ut_transformed)

        return transform

    def add_transform_single_(self, name, source, target):

        parms_target, vscaled_target, func_cdfinv_target = self.get_invcdf(target)
        parms_source, vscaled_source, func_cdf_source = self.get_cdf(source)
        nparms = parms_target.shape[0] + parms_source.shape[0]

        def func_cdfinv_pdf_target(pt, quant):
            with tf.GradientTape() as t:
                t.watch(quant)
                cdfinv = func_cdfinv_target(pt, quant)
            pdfreciprocal = t.gradient(cdfinv, quant)
            pdf = 1./pdfreciprocal
            return cdfinv, pdf

        def transform_stat_unc(pt, ut):

            with tf.GradientTape(persistent=True) as t:
                t.watch(parms_source)
                t.watch(parms_target)

                cdf_source = func_cdf_source(pt, ut)
                ut_transformed, pdf = func_cdfinv_pdf_target(pt, cdf_source)

                ut_transformed = tf.reshape(ut_transformed, [])
                pdf = tf.reshape(pdf, [])

            pdf_grad_source = t.gradient(pdf, parms_source)
            pdf_grad_target = t.gradient(pdf, parms_target)

            del t

            weight_grad_source = pdf_grad_source/pdf
            weight_grad_target = pdf_grad_target/pdf

            weight_grad_source = weight_grad_source[None, :]
            weight_grad_target = weight_grad_target[None, :]

            weight_grad_mc_eig = weight_grad_source @ vscaled_source
            weight_grad_data_eig = weight_grad_target @ vscaled_target

            weight_grad_mc_eig = tf.reshape(weight_grad_mc_eig, [-1])
            weight_grad_data_eig = tf.reshape(weight_grad_data_eig, [-1])

            weight_grad_eig = tf.concat([weight_grad_mc_eig, weight_grad_data_eig], axis=0) + 1.0
            weight_grad_eig_nom = tf.constant([1.]*nparms, dtype=tf.float64)

            ret = tf.where(tf.math.logical_or(tf.math.greater(ut, self.ut_max), tf.math.less(ut, self.ut_min)), ut, ut_transformed)
            ret_grad = tf.where(tf.math.logical_or(tf.math.greater(ut, self.ut_max), tf.math.less(ut, self.ut_min)), weight_grad_eig_nom, weight_grad_eig)
            return ret, ret_grad
        
        self.funcs.append(transform_stat_unc)
        self.func_names.append(f"transform_{name}")

        def transform(pt, ut):
            cdf_mc = func_cdf_source(pt, ut)
            ut_transformed, pdf = func_cdfinv_pdf_target(pt, cdf_mc)
            ut_transformed = tf.reshape(ut_transformed, [])
            return tf.where(tf.math.logical_or(tf.math.greater(ut, self.ut_max), tf.math.less(ut, self.ut_min)), ut, ut_transformed)

        self.funcs.append(transform)
        self.func_names.append(f"transform_{name}_no_unc")
        return nparms

    def add_pdf(self, name, nom, para=True):
        if para:
            ut_min, ut_max = self.ut_min_para, self.ut_max_para
        else:
            ut_min, ut_max = self.ut_min_perp, self.ut_max_perp

        # returns the pdf value 
        parms_nom, vscaled_nom, func_cdf_nom = self.get_cdf(nom)

        def calc_pdf(pt, ut):
            ut = tf.where(tf.math.greater(ut, ut_max), ut_max, ut)
            ut = tf.where(tf.math.less(ut, ut_min), ut_min, ut)
            pt = tf.where(tf.math.greater(pt, self.pt_max), self.pt_max, pt)
            with tf.GradientTape() as t:
                t.watch(ut)
                pdf = func_cdf_nom(pt, ut)
            pdfval_nom = t.gradient(pdf, ut)
            return pdfval_nom
 
        self.funcs.append(calc_pdf)
        self.func_names.append(f"pdf_{name}")
        self.func_inputs.append([self.scalar_spec, self.scalar_spec])

    def add_systematic(self, name, nom, pert):
        # returns the weight as ratio of two PDFs
        parms_nom, vscaled_nom, func_cdf_nom = self.get_cdf(nom)
        parms_pert, vscaled_pert, func_cdf_pert = self.get_cdf(pert)

        def calc_weight(pt, ut):
            ut = tf.where(tf.math.greater(ut, self.ut_max), self.ut_max, ut)
            ut = tf.where(tf.math.less(ut, self.ut_min), self.ut_min, ut)
            pt = tf.where(tf.math.greater(pt, self.pt_max), self.pt_max, pt)
            with tf.GradientTape() as t:
                t.watch(ut)
                pdf = func_cdf_nom(pt, ut)
            pdfval_nom = t.gradient(pdf, ut)

            with tf.GradientTape() as t:
                t.watch(ut)
                pdf = func_cdf_pert(pt, ut)
            pdfval_pert = t.gradient(pdf, ut)
            
            return tf.math.divide(pdfval_pert, pdfval_nom)
 
        self.funcs.append(calc_weight)
        self.func_names.append(f"syst_{name}")
        self.func_inputs.append([self.scalar_spec, self.scalar_spec])

    def export(self, out):
        def meta():
            nstat = tf.constant([self.nstat], dtype=tf.int64)
            return nstat

        self.funcs.insert(0, meta)
        self.func_names.insert(0, "meta")
        self.func_inputs.insert(0, [])
        model = narf.tfutils.function_to_tflite(self.funcs, self.func_inputs, self.func_names)

        with open(out, 'wb') as f:
            f.write(model)


    def test(self, out):
        with open(out, 'rb') as f:
            model = f.read()
            interpreter = tf.lite.Interpreter(model_content=model)
        

        pt = tf.constant(10, dtype=tf.float64)
        ut_para = tf.constant(5, dtype=tf.float64)
        ut_perp = tf.constant(5, dtype=tf.float64)

        testval = interpreter.get_signature_runner('base_transform_no_unc')(input_00002_00000=pt, input_00002_00001=ut_para, input_00002_00002=ut_perp)
        print(testval)
        print(interpreter.get_tensor_details())