"""
Summarize a recoil calibration model on a grid of qT and recoil, through the C++ helpers used by WRemnants: the
calibrated recoil, the total statistical uncertainty (sum in quadrature of the variations of the stat. nuisances)
and the systematic variations. With a reference model, also the difference of the calibrated recoil.

usage: python reproduce/summarize_model.py <model.tflite> [--ref <reference.tflite>]
"""

import argparse

import numpy as np

import narf  # noqa: F401
import narf.tfliteutils  # noqa: F401
import ROOT
from wremnants.production import recoil_tools

ROOT.gInterpreter.Declare(
    r"""
namespace summodel {
template <typename H>
std::vector<double> base(H &h, double pt, double upara, double uperp) {
  auto r = h(pt, upara, uperp);
  std::vector<double> out{r.ut_para_corr(0), r.ut_perp_corr(0)};
  for (int k = 0; k < r.unc_weights.size(); ++k) out.push_back(r.unc_weights(k));
  return out;
}
double syst2(wrem::RecoilCalibrationUncertaintyHelper &h, double pt, double u) { return h(pt, u); }
double syst3(wrem::RecoilCalibrationSystematicHelper &h, double pt, double u, double ucorr) { return h(pt, u, ucorr); }
}
"""
)

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--ref", default=None, help="Reference model")
parser.add_argument("--qts", type=float, nargs="+", default=[2, 10, 30, 60, 100, 150])
parser.add_argument("--us", type=float, nargs="+", default=[-60, -40, -20, -10, 0, 10, 20, 40, 60])
args = parser.parse_args()


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            interpreter = recoil_tools.Interpreter(model_content=f.read())
        self.nstat = int(interpreter.get_signature_runner("meta")()["output_00000_00000"][0])
        self.base = ROOT.wrem.RecoilCalibrationHelper[self.nstat](path, "base_transform", True)
        self.systs = {}
        for name, signature in sorted(interpreter.get_signature_list().items()):
            if name.startswith("syst_"):
                if len(signature["inputs"]) == 3:
                    self.systs[name] = (ROOT.wrem.RecoilCalibrationSystematicHelper(path, name), 3)
                else:
                    self.systs[name] = (ROOT.wrem.RecoilCalibrationUncertaintyHelper(path, name), 2)

    def transform(self, qt, u_para, u_perp):
        out = np.array(list(ROOT.summodel.base(self.base, qt, u_para, u_perp)))
        return out[0], out[1], out[2:]

    def syst(self, name, qt, u, ucorr):
        helper, ninputs = self.systs[name]
        # the old (pt, u) functions are ratios of data pdfs, evaluated at the calibrated recoil
        return ROOT.summodel.syst3(helper, qt, u, ucorr) if ninputs == 3 else ROOT.summodel.syst2(helper, qt, ucorr)


model = Model(args.model)
ref = Model(args.ref) if args.ref else None
print(f"{model.nstat} stat. nuisances, systematics: {list(model.systs.keys())}")

for comp in ("para", "perp"):
    print(f"\n=== {comp}: u' - u (GeV), stat. uncertainty of the log of the event density (x1e3), systematic weights - 1 (x1e3)")
    systs = [n for n in model.systs if comp in n]
    for qt in args.qts:
        print(f"qT = {qt} GeV")
        header = f"  {'u':>6s} {'du':>7s}"
        if ref:
            header += f" {'du-ref':>7s} {'stat-ref':>8s}"
        header += f" {'stat':>7s}" + "".join(f" {n.removeprefix('syst_'):>16s}" for n in systs)
        print(header)
        for u in args.us:
            # the other component outside of the calibrated range, where its stat. weights are 1
            u_para, u_perp = (u, 100.0) if comp == "para" else (100.0, u)
            para, perp, w = model.transform(qt, u_para, u_perp)
            up = para if comp == "para" else perp
            line = f"  {u:6.1f} {up - u:7.2f}"
            if ref:
                rpara, rperp, rw = ref.transform(qt, u_para, u_perp)
                line += f" {up - (rpara if comp == 'para' else rperp):7.2f} {1e3 * np.sqrt(np.sum(np.log(np.clip(rw, 1e-6, None)) ** 2)):8.2f}"
            line += f" {1e3 * np.sqrt(np.sum(np.log(np.clip(w, 1e-6, None)) ** 2)):7.2f}"
            for n in systs:
                line += f" {1e3 * (model.syst(n, qt, u, up) - 1):16.2f}"
            print(line)
