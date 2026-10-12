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

Recoil fits (for parallel/perpendicular data/mc/bkg) and export to a TFLite model:

```shell
python calibration/highPU_DeepMETPVRobust.py
```

Re-export a model from existing fit results and compare it with a reference model (through the C++
TFLite helper used by WRemnants):

```shell
python reproduce/reexport.py data/highPU_DeepMETPVRobust data/highPU_DeepMETPVRobust/model_mc_data_reexport.tflite
python reproduce/compare_models.py <reference.tflite> data/highPU_DeepMETPVRobust/model_mc_data_reexport.tflite
```

Re-exporting the May 2024 fit results reproduces the model deployed in wremnants-data (commit 38416e6)
to machine precision.
