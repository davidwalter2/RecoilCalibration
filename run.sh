#!/bin/bash

python calibration/highPU_RawPFMET.py
python calibration/highPU_DeepMETReso.py
python calibration/highPU_DeepMETPVRobust.py
python calibration/highPU_DeepMETPVRobustNoPUPPI.py

#python calibration/lowPU_DeepMETReso.py
#python calibration/lowPU_RawPFMET.py