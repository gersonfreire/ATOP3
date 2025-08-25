#!/bin/bash
#DSUB -n zhy
#DSUB -N 1
#DSUB -A root.qrn8y2ug
#DSUB -R "cpu=16;gpu=0;mem=40000"
#DSUB -oo %J.out
#DSUB -eo %J.err

python cli.py