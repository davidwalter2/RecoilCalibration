
import sys,array,math,os
import pickle
import numpy as np
import tensorflow as tf
import narf
import narf.fitutils
import narf.tfutils


class Export:

    def __init__(self, target_para, target_perp, source_para, source_perp):
    
        self.funcs = []

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
        vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
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
        vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
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

    def setup_transform(self, name, source, target, ut_min, ut_max):

        parms_target, vscaled_target, func_cdfinv_target = self.get_invcdf(target)
        parms_source, vscaled_source, func_cdf_source = self.get_cdf(source)

        nparms = parms_target.shape[0] + parms_source.shape[0]
        ut_min = tf.cast(ut_min, tf.float64)
        ut_max = tf.cast(ut_max, tf.float64)

        def tf(pt, ut):

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

            ret = tf.where(tf.math.logical_or(tf.math.greater(ut, ut_max), tf.math.less(ut, ut_min)), ut, ut_transformed)
            ret_grad = tf.where(tf.math.logical_or(tf.math.greater(ut, ut_max), tf.math.less(ut, ut_min)), weight_grad_eig_nom, weight_grad_eig)
            return ret, ret_grad

        def tf_no_unc(pt, ut):
            cdf_mc = func_cdf_source(pt, ut)
            ut_transformed, pdf = func_cdfinv_pdf_target(pt, cdf_mc)
            ut_transformed = tf.reshape(ut_transformed, [])
            return tf.where(tf.math.logical_or(tf.math.greater(ut, ut_max), tf.math.less(ut, ut_min)), ut, ut_transformed)

        if name == "para":
            transform_para = tf
            transform_para_no_unc = tf_no_unc
            self.funcs.append(transform_para)
            self.funcs.append(transform_para_no_unc)
        if name == "perp":
            transform_perp = tf
            transform_perp_no_unc = tf_no_unc
            self.funcs.append(transform_perp)
            self.funcs.append(transform_perp_no_unc)

    def export(self):

        scalar_spec = tf.TensorSpec([], tf.float64)
        input_signature = [scalar_spec, scalar_spec]
        model = narf.tfutils.function_to_tflite([meta, transform_para, transform_perp, transform_para_no_unc, transform_perp_no_unc], [[], input_signature, input_signature, input_signature, input_signature])
        #model = narf.tfutils.function_to_tflite([transform_perp], [input_signature])

        with open(out, 'wb') as f:
            f.write(model)


def get_scaled_eigenvectors(hess, num_null = 2):
    e,v = np.linalg.eigh(hess)

    # remove the null eigenvectors
    e = e[None, num_null:]
    v = v[:, num_null:]

    # scale the eigenvectors
    vscaled = v/np.sqrt(e)

    return vscaled

def get_invcdf(input_file):

    ut_axis = 1
    qt_axis = 0

    handle = open(input_file, 'rb')
    res = pickle.load(handle)
    quant_cdfvals = res['quant_cdfvals_tf']
    knots_qt = res['knots_qt']
    order = res['order']
    extrpl = res['extrpl']
    parms = tf.constant(res['x'], tf.float64)
    vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
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

def get_cdf(input_file):

    ut_axis = 1
    qt_axis = 0

    handle = open(input_file, 'rb')
    res = pickle.load(handle)
    quant_cdfvals = res['quant_cdfvals_tf']
    knots_qt = res['knots_qt']
    order = res['order']
    extrpl = res['extrpl']
    parms = tf.constant(res['x'], tf.float64)
    vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
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


def exportModel(target_para, target_perp, source_para, source_perp, ut_min, ut_max, out):

    parms_target_para, vscaled_target_para, func_cdfinv_target_para = get_invcdf(target_para)
    parms_target_perp, vscaled_target_perp, func_cdfinv_target_perp = get_invcdf(target_perp)

    parms_source_para, vscaled_source_para, func_cdf_source_para = get_cdf(source_para)
    parms_source_perp, vscaled_source_perp, func_cdf_source_perp = get_cdf(source_perp)

    nparms_para = parms_target_para.shape[0] + parms_source_para.shape[0]
    nparms_perp = parms_target_perp.shape[0] + parms_source_perp.shape[0]

    ut_min = tf.cast(ut_min, tf.float64)
    ut_max = tf.cast(ut_max, tf.float64)

    def func_cdfinv_pdf_target_para(pt, quant):
        with tf.GradientTape() as t:
            t.watch(quant)
            cdfinv = func_cdfinv_target_para(pt, quant)
        pdfreciprocal = t.gradient(cdfinv, quant)
        pdf = 1./pdfreciprocal
        return cdfinv, pdf

    def func_cdfinv_pdf_target_perp(pt, quant):
        with tf.GradientTape() as t:
            t.watch(quant)
            cdfinv = func_cdfinv_target_perp(pt, quant)
        pdfreciprocal = t.gradient(cdfinv, quant)
        pdf = 1./pdfreciprocal
        return cdfinv, pdf

    def transform_para_no_unc(pt, ut_para):
        cdf_mc = func_cdf_source_para(pt, ut_para)
        ut_transformed, pdf = func_cdfinv_pdf_target_para(pt, cdf_mc)
        ut_transformed = tf.reshape(ut_transformed, [])
        return tf.where(tf.math.logical_or(tf.math.greater(ut_para, ut_max), tf.math.less(ut_para, ut_min)), ut_para, ut_transformed)
        
    def transform_perp_no_unc(pt, ut_perp):
        cdf_mc = func_cdf_source_perp(pt, ut_perp)
        ut_transformed, pdf = func_cdfinv_pdf_target_perp(pt, cdf_mc)
        ut_transformed = tf.reshape(ut_transformed, [])
        return tf.where(tf.math.logical_or(tf.math.greater(ut_perp, ut_max), tf.math.less(ut_perp, ut_min)), ut_perp, ut_transformed)

    def transform_para(pt, ut_para):

        with tf.GradientTape(persistent=True) as t:
            t.watch(parms_source_para)
            t.watch(parms_target_para)

            cdf_source_para = func_cdf_source_para(pt, ut_para)
            ut_transformed_para, pdf_para = func_cdfinv_pdf_target_para(pt, cdf_source_para)

            ut_transformed_para = tf.reshape(ut_transformed_para, [])
            pdf_para = tf.reshape(pdf_para, [])

        pdf_grad_source_para = t.gradient(pdf_para, parms_source_para)
        pdf_grad_target_para = t.gradient(pdf_para, parms_target_para)

        del t

        weight_grad_source_para = pdf_grad_source_para/pdf_para
        weight_grad_target_para = pdf_grad_target_para/pdf_para

        weight_grad_source_para = weight_grad_source_para[None, :]
        weight_grad_target_para = weight_grad_target_para[None, :]

        weight_grad_mc_eig_para = weight_grad_source_para @ vscaled_source_para
        weight_grad_data_eig_para = weight_grad_target_para @ vscaled_target_para

        weight_grad_mc_eig_para = tf.reshape(weight_grad_mc_eig_para, [-1])
        weight_grad_data_eig_para = tf.reshape(weight_grad_data_eig_para, [-1])

        weight_grad_eig_para = tf.concat([weight_grad_mc_eig_para, weight_grad_data_eig_para], axis=0) + 1.0
        weight_grad_eig_para_nom = tf.constant([1.]*nparms_para, dtype=tf.float64)

        ret = tf.where(tf.math.logical_or(tf.math.greater(ut_para, ut_max), tf.math.less(ut_para, ut_min)), ut_para, ut_transformed_para)
        ret_grad = tf.where(tf.math.logical_or(tf.math.greater(ut_para, ut_max), tf.math.less(ut_para, ut_min)), weight_grad_eig_para_nom, weight_grad_eig_para)
        return ret, ret_grad
        
    def transform_perp(pt, ut_perp):

        with tf.GradientTape(persistent=True) as t:
            t.watch(parms_source_perp)
            t.watch(parms_target_perp)

            cdf_source_perp = func_cdf_source_perp(pt, ut_perp)
            ut_transformed_perp, pdf_perp = func_cdfinv_pdf_target_perp(pt, cdf_source_perp)

            ut_transformed_perp = tf.reshape(ut_transformed_perp, [])
            pdf_perp = tf.reshape(pdf_perp, [])

        pdf_grad_source_perp = t.gradient(pdf_perp, parms_source_perp)
        pdf_grad_target_perp = t.gradient(pdf_perp, parms_target_perp)

        del t

        weight_grad_source_perp = pdf_grad_source_perp/pdf_perp
        weight_grad_target_perp = pdf_grad_target_perp/pdf_perp

        weight_grad_source_perp = weight_grad_source_perp[None, :]
        weight_grad_target_perp = weight_grad_target_perp[None, :]

        weight_grad_mc_eig_perp = weight_grad_source_perp @ vscaled_source_perp
        weight_grad_data_eig_perp = weight_grad_target_perp @ vscaled_target_perp

        weight_grad_mc_eig_perp = tf.reshape(weight_grad_mc_eig_perp, [-1])
        weight_grad_data_eig_perp = tf.reshape(weight_grad_data_eig_perp, [-1])

        weight_grad_eig_perp = tf.concat([weight_grad_mc_eig_perp, weight_grad_data_eig_perp], axis=0) + 1.0
        weight_grad_eig_perp_nom = tf.constant([1.]*nparms_perp, dtype=tf.float64)

        ret = tf.where(tf.math.logical_or(tf.math.greater(ut_perp, ut_max), tf.math.less(ut_perp, ut_min)), ut_perp, ut_transformed_perp)
        ret_grad = tf.where(tf.math.logical_or(tf.math.greater(ut_perp, ut_max), tf.math.less(ut_perp, ut_min)), weight_grad_eig_perp_nom, weight_grad_eig_perp)
        return ret, ret_grad

    def meta():
        n_unc = tf.constant([nparms_para, nparms_perp], dtype=tf.int64)
        return n_unc

    scalar_spec = tf.TensorSpec([], tf.float64)
    input_signature = [scalar_spec, scalar_spec]
    model = narf.tfutils.function_to_tflite([meta, transform_para, transform_perp, transform_para_no_unc, transform_perp_no_unc], [[], input_signature, input_signature, input_signature, input_signature])
    #model = narf.tfutils.function_to_tflite([transform_perp], [input_signature])

    with open(out, 'wb') as f:
        f.write(model)


    test_interp = tf.lite.Interpreter(model_content=model)
    #test_interp.allocate_tensors()
    #print()
    #print("get_signature_list")
    #print(test_interp.get_signature_list())
    
    #print("get_tensor_details")
    #print(test_interp.get_tensor_details())
    #quit()
    #print()
    #pt = tf.constant(10, dtype=tf.float64)
    #ut = tf.constant(0, dtype=tf.float64)
    #para = test_interp.get_signature_runner('transform_perp')(input_00001_00000=pt, input_00001_00001=ut)
    #print(para)
    #print(para['output_00001_00001'].shape)
    n_unc = test_interp.get_signature_runner('meta')()
    print(n_unc['output_00000_00000'][0])
    


if __name__ == "__main__":

    tag = "highPU_DeepMETReso" # highPU_RawPFMET highPU_DeepMETReso lowPU_mumu_DeepMETReso
    ut_min, ut_max = -80, 80

    target_para = f"data/{tag}/data_para_postfit.pkl"
    target_perp = f"data/{tag}/data_perp_postfit.pkl"
    source_para = f"data/{tag}/z_para_postfit.pkl"
    source_perp = f"data/{tag}/z_perp_postfit.pkl"
    out = f"data/{tag}/model_mc_data.tflite"
    systs = []
    exportModel(target_para, target_perp, source_para, source_perp, ut_min, ut_max, out)

    target_para = f"data/{tag}/zmumu_para_postfit.pkl"
    target_perp = f"data/{tag}/zmumu_perp_postfit.pkl"
    source_para = f"data/{tag}/zmumu_gen_para_postfit.pkl"
    source_perp = f"data/{tag}/zmumu_gen_perp_postfit.pkl"
    out = f"data/{tag}/model_mc_gen.tflite"
    #exportModel(target_para, target_perp, source_para, source_perp, ut_min, ut_max, out)


    target_para = f"data/{tag}/data_para_postfit.pkl"
    target_perp = f"data/{tag}/data_perp_postfit.pkl"
    source_para = f"data/{tag}/z_gen_para_postfit.pkl"
    source_perp = f"data/{tag}/z_gen_perp_postfit.pkl"
    out = f"data/{tag}/model_gen_data.tflite"
    #exportModel(target_para, target_perp, source_para, source_perp, ut_min, ut_max, out)