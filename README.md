# RecoilCalibration

Clone the repo:

```shell
git clone git@github.com:davidwalter2/RecoilCalibration.git
```

Need to run in singularity. The fit and export utilities (`fitutils`, `tfutils`) now live in `wums`, so
`setup.sh` sources a WRemnants checkout (default `/work/submit/david_w/WRemnants`, override with `WREM_DIR`):

```shell
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
source setup.sh
```

## Calibration of DeepMETPVRobust (2026 version)

Recoil fits (parallel/perpendicular, data/MC/background, and the variations) and export to a TFLite model. Independent
fits run in parallel (each uses ~5 cores, run on a large node):

```shell
calibration/run_highPU_DeepMETPVRobust.sh [dataDir] [plotDir]
```

or step by step with `python calibration/highPU_DeepMETPVRobust.py --steps <steps>`. The inputs are
`<dataDir>/input_mumu.pkl`, produced with `scripts/recoil/exportHists.py` of WRemnants from a W-like histmaker output
with `--recoilHists`. `data/highPU_DeepMETPVRobust_v2/` uses the same inputs as the May 2024 calibration in
`data/highPU_DeepMETPVRobust/`.

Changes with respect to the May 2024 calibration:
- Weights for the statistical uncertainties: a variation of the data fit parameters changes the calibrated distribution
  through `p_data(u')` at fixed calibrated recoil `u'`, a variation of the MC fit parameters through `1/p_mc(u)` at
  fixed `u`. The previous weights differentiated `p_data(F_data^-1(q))` at fixed quantile `q`, which gives the wrong
  shape, and the wrong sign for the MC parameters (`reproduce/check_stat_weights.py`). The weights are exponentiated
  so that they stay positive.
- The background fits are refitted with the constraint, like the other fits, and fits that end unconverged are
  continued with trust-exact (dense hessian), see `polish_fit`. The fits are only once differentiable in the
  parameters (pchip), so the minimization stalls around edm ~ 0.1.
- The background systematic uses data fits with the background normalisation scaled by 1.2 and 0.8 from the same
  inputs (the May 2024 model used stale fits from April 2024 for the variation), as a two-sided variation.
- New systematic for the qT parametrisation, from MC and data fits with an additional spline knot at 90 GeV (`*_ext`
  fits), which mostly matters where the calibration is extrapolated beyond the last knot at 60 GeV.

The systematics are exported as `syst_<name>(qT, u, u')`, with `u` and `u'` the recoil component before and after the
calibration: `syst_bkg_{para,perp}{Up,Down}` and `syst_qtParam_{para,perp}`. WRemnants evaluates all `syst_*`
functions of the model (`--recoilModel` to use a model outside of wremnants-data).

## Validation

```shell
# compare the fit results with a reference (status, edm, shifts of the quantiles)
python reproduce/compare_fits.py data/highPU_DeepMETPVRobust data/highPU_DeepMETPVRobust_v2
# calibrated recoil, stat. uncertainty and systematic variations on a grid, compared to a reference model
python reproduce/summarize_model.py data/highPU_DeepMETPVRobust_v2/model_mc_data.tflite --ref data/highPU_DeepMETPVRobust/model_mc_data.tflite
# check of the weights for the statistical uncertainties against finite differences and with a closure test
python reproduce/check_stat_weights.py data/highPU_DeepMETPVRobust_v2 --comp para
```

Re-export a model from existing fit results and compare it with a reference model (through the C++ TFLite helper used
by WRemnants):

```shell
python reproduce/reexport.py data/highPU_DeepMETPVRobust data/highPU_DeepMETPVRobust/model_mc_data_reexport.tflite
python reproduce/compare_models.py <reference.tflite> data/highPU_DeepMETPVRobust/model_mc_data_reexport.tflite
```

Re-exporting the May 2024 fit results with the export code of commit 576888c (before the changes above) reproduces the
model deployed in wremnants-data (commit 38416e6) to machine precision.
