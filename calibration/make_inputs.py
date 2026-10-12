"""
Input histograms of the recoil calibration from a W-like histmaker output of WRemnants with --recoilHists: the recoil
components (u_para + qT and u_perp) vs the reconstructed dimuon qT for the calibrated Z sample, the backgrounds and
data, and the recoil along the generator boson proxy vs its pt for the calibrated Z sample.

The calibrated Z sample are the datasets the calibration is applied to in WRemnants (samples.zprocs_recoil), all
other simulated processes form the background. Replaces scripts/recoil/exportHists.py of WRemnants, which takes the
Z sample from the Zmumu group and fails since it also contains Zmumu10to50.

usage: python calibration/make_inputs.py -i <histmaker output>.hdf5 -o <dataDir>/input_mumu.pkl
"""

import argparse
import os
import pickle

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wremnants.utilities import samples

HISTS = {
    "para": "recoil_corr_xy_para_qt_v_pt",
    "perp": "recoil_corr_xy_perp_v_pt",
}
HISTS_GEN = {
    "para_gen": "recoil_corr_xy_para_gen_v_gen_pt",
    "para_qt_gen": "recoil_corr_xy_para_qt_gen_v_gen_pt",
    "perp_gen": "recoil_corr_xy_perp_gen_v_gen_pt",
}


def read(groups, hname, procs):
    groups.setNominalName(hname)
    groups.loadHistsForDatagroups(hname, syst="", procsToRead=procs)
    h = sum([groups.groups[p].hists[hname] for p in procs])
    vals = h.values()
    vals[vals < 0] = 0  # remove negative values
    h.view().value = vals
    return h


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Histmaker output (W-like with --recoilHists)")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output pkl file")
    args = parser.parse_args()

    groups = Datagroups(args.input)
    calibrated = set(samples.zprocs_recoil)

    # split the groups into the calibrated Z datasets and the rest
    data, z, bkg = [], [], []
    for name in list(groups.getNames()):
        members = groups.groups[name].members
        if len(members) == 0:
            continue
        if all(m.is_data for m in members):
            data.append(name)
            continue
        for suffix, keep, target in (("_calibrated", True, z), ("_other", False, bkg)):
            if any((m.name in calibrated) == keep for m in members):
                groups.copyGroup(name, f"{name}{suffix}", member_filter=lambda m, keep=keep: (m.name in calibrated) == keep)
                target.append(f"{name}{suffix}")

    print(f"data: {data}")
    print(f"calibrated Z: {[(g, [m.name for m in groups.groups[g].members]) for g in z]}")
    print(f"background: {[(g, [m.name for m in groups.groups[g].members]) for g in bkg]}")

    out = {"lumi_header": "{:.1f} fb^{{#minus1}} (13 TeV)".format(groups.lumi)}
    for comp, hname in HISTS.items():
        out[f"z_{comp}"] = read(groups, hname, z)
        out[f"bkg_{comp}"] = read(groups, hname, bkg)
        out[f"data_{comp}"] = read(groups, hname, data)
    for comp, hname in HISTS_GEN.items():
        out[f"z_{comp}"] = read(groups, hname, z)

    for key, h in out.items():
        if key != "lumi_header":
            print(f"{key:15s} {str(h.axes.name):35s} {h.sum(flow=True).value:14.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"Written {args.output}")
