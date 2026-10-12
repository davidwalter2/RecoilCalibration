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
`<dataDir>/input_mumu.pkl`, produced from a W-like histmaker output of WRemnants with `--recoilHists`:

```shell
python calibration/make_inputs.py -i <mz_wlike_with_mu_eta_pt_*.hdf5> -o <dataDir>/input_mumu.pkl
```

The Z sample is the one the calibration is applied to in WRemnants (`samples.zprocs_recoil`), all other simulated
processes are background. Calibrations:
- `data/highPU_DeepMETPVRobust/`: May 2024 calibration (deployed in wremnants-data), from Jan's inputs (W-like
  histmaker of May 2024 without theory corrections, converted with `scripts/recoil/exportHists.py`).
- `data/highPU_DeepMETPVRobust_v2/`: fits redone from the same inputs with the changes below.
- `data/highPU_DeepMETPVRobust_v3/`: new inputs from the W-like histmaker of WRemnants 3fec7dd2 with the default
  theory corrections (`mz_wlike_with_mu_eta_pt.py --recoilHists`,
  `/ceph/submit/data/user/d/david_w/WMassAnalysis/histmaker_results/261012_wlike_lowMt/mz_wlike_with_mu_eta_pt_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr_maxFiles_m1_recoilInputs.hdf5`),
  also containing the recoil along the generator boson proxy.
- `data/highPU_DeepMETPVRobust_v3_noTheoryCorr/`: only the inputs of the same histmaker without theory corrections
  (`--theoryCorr --ewTheoryCorr`), for the comparison with the May 2024 inputs, which had none
  (`inputs_May2024_vs_noTheoryCorr.txt`), and with the inputs of v3 (`inputs_noTheoryCorr_vs_theoryCorr.txt`).
- `data/highPU_DeepMETPVRobust_v4_dilepton/`: inputs from the dilepton histmaker with the dilepton trigger selection and
  a wider muon acceptance (`mz_dilepton.py --recoilHists --useDileptonTriggerSelection --pt 174 26 200`,
  `/ceph/submit/data/user/d/david_w/WMassAnalysis/histmaker_results/261012_recoilInputs_dilepton/mz_dilepton_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr_maxFiles_m1_wideAcceptance.hdf5`):
  9.24M instead of 7.52M data events, 168k instead of about 2k above qT = 100 GeV. Fitted with the default configuration,
  compared with v3 for the dependence on the selection (`compare_fits_vs_v3.txt`, `summary_vs_v3.txt`).
- `data/highPU_DeepMETPVRobust_v4_dilepton_wide/`: the same inputs fitted up to qT = 200 GeV, with knots up to 150 GeV
  and an extra knot at 200 GeV for the qtParam variation, to measure the calibration where the default configuration
  extrapolates (`summary_vs_v3.txt`, `summary_vs_v4_dilepton.txt`):

  ```shell
  calibration/run_highPU_DeepMETPVRobust.sh data/highPU_DeepMETPVRobust_v4_dilepton_wide <plotDir> \
      --qtBins $(seq 0 30) $(seq 32 2 50) $(seq 55 5 80) 90 100 110 120 135 150 175 200 \
      --knotsPara 1 4 8 15 30 60 100 150 --knotsPerp 1 4 8 20 60 100 150 \
      --knotsParaExt 1 4 8 15 30 60 100 150 200 --knotsPerpExt 1 4 8 20 60 100 150 200
  ```

  The fit `data_para_ext` ends at edm = 0.46 (continuing it with `--polish` does not improve it); it only enters the
  qtParam systematic.

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
