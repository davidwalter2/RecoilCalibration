"""Compare two recoil calibration tflite models through the narf C++ tflite helper used by WRemnants.

usage: python reproduce/compare_models.py <reference.tflite> <test.tflite>
"""

import sys

import numpy as np

import narf  # noqa: F401
import narf.tfliteutils  # noqa: F401
import ROOT
from wremnants.production import recoil_tools  # noqa: F401  declares recoil_helper.hpp

ROOT.gInterpreter.Declare(
    r"""
namespace cmpmodel {
std::vector<double> base(wrem::RecoilCalibrationHelper<1100> &h, double pt, double upara, double uperp) {
  auto r = h(pt, upara, uperp);
  std::vector<double> out{r.ut_para_corr(0), r.ut_perp_corr(0)};
  for (int k = 0; k < 1100; ++k) out.push_back(r.unc_weights(k));
  return out;
}
double f2(wrem::RecoilCalibrationUncertaintyHelper &h, double pt, double u) { return h(pt, u); }
}
"""
)
names = ["syst_bkg_para", "syst_bkg_perp", "pdf_data_para", "pdf_data_perp", "pdf_mc_para", "pdf_mc_perp"]
models = {}
for path in sys.argv[1:3]:
    models[path] = dict(
        base=ROOT.wrem.RecoilCalibrationHelper[1100](path, "base_transform", True),
        f2={n: ROOT.wrem.RecoilCalibrationUncertaintyHelper(path, n) for n in names},
    )
ref, test = [models[p] for p in sys.argv[1:3]]
qts = [0.5, 2, 5, 10, 20, 30, 50, 70, 90, 100, 150]
us = np.arange(-79.5, 80, 2.5)
dmax = dict(para=0.0, perp=0.0, w=0.0)
for qt in qts:
    for u in us:
        for comp, args in (("para", (u, 0.0)), ("perp", (0.0, u))):
            a = np.array(list(ROOT.cmpmodel.base(ref["base"], qt, *args)))
            b = np.array(list(ROOT.cmpmodel.base(test["base"], qt, *args)))
            i = 0 if comp == "para" else 1
            dmax[comp] = max(dmax[comp], abs(a[i] - b[i]))
            dmax["w"] = max(dmax["w"], np.max(np.abs(a[2:] - b[2:])))
print(f"max |delta corrected u| para {dmax['para']:.2e} GeV, perp {dmax['perp']:.2e} GeV; max |delta weight| {dmax['w']:.2e}")
for n in names:
    d = max(abs(ROOT.cmpmodel.f2(ref["f2"][n], qt, u) - ROOT.cmpmodel.f2(test["f2"][n], qt, u)) for qt in qts for u in us)
    print(f"max |delta {n}| {d:.2e}")
