"""Re-export the recoil calibration model from existing fit results (same steps as export() in
calibration/highPU_DeepMETPVRobust.py), writing to a new file.

usage: python reproduce/reexport.py <dataDir> <output.tflite>
"""

import logging
import sys

import exportModel

logging.basicConfig(level=logging.INFO)
dataDir, out = sys.argv[1], sys.argv[2]

exp = exportModel.Export()
exp.set_limits(500, -80, 80)
exp.add_base_transform(
    f"{dataDir}/z_para_postfit.pkl",
    f"{dataDir}/data_para_postfit.pkl",
    f"{dataDir}/z_perp_postfit.pkl",
    f"{dataDir}/data_perp_postfit.pkl",
)
exp.add_systematic("bkg_para", f"{dataDir}/data_para_postfit.pkl", f"{dataDir}/data_para_bkg_postfit.pkl")
exp.add_systematic("bkg_perp", f"{dataDir}/data_perp_postfit.pkl", f"{dataDir}/data_perp_bkg_postfit.pkl")
exp.add_pdf("data_para", f"{dataDir}/data_para_postfit.pkl")
exp.add_pdf("data_perp", f"{dataDir}/data_perp_postfit.pkl")
exp.add_pdf("mc_para", f"{dataDir}/z_para_postfit.pkl")
exp.add_pdf("mc_perp", f"{dataDir}/z_perp_postfit.pkl")
exp.export(out)
