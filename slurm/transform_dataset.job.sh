#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate xray
cd /media/compute/homes/dmindlin/xray

python3 -m src.data.transform_dataset