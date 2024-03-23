#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate InvestoGPT
python data_pipeline/pipeline_update/overall_pipeline.py