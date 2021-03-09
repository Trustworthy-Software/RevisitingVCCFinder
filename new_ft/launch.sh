#!/bin/bash

set -e
set -x

###################
#
#	
#
#
###################

./prepare_input.sh
#Generate graphs to 4_Results/2_Plots/
cd 2_Tools/
python3 python_sklearn.py
#Gives the Precision and Thresholdd for the recall of 0.24
python3 const_recall.py
cd ..
