#!/bin/bash
# TLDR; : This script will perform all experiments.
#
#	For documented explanation, see Documentation.txt
#Implemented and tested with python 3.7 

set -e
set -x

### To import DB see Documentation.txt
echo '#####################################__Replication__#####################################'
echo 'Replication and generating the ct_input will take several hours'
#Replication
cd replicate
./Launch_Instructions.sh
#Generate Data for co-Training in replicate/4_Results/1_Data/2_CT_Input
./launch_ct_input_generation.sh
cd ..

echo '#####################################__Co-Training with New Features__#####################################'
# cotraining with NEW features:
mkdir -p exp_new_features
# commits_extraction.py will Extract both the two new feature set:
# 1) commits_msg
# 2) Code metrics
./commits_extraction.py exp_new_features/

./cotraining_new_features.py exp_new_features/ 1000
./cotraining_new_features.py exp_new_features/ 5000
./cotraining_new_features.py exp_new_features/ 10000
# Writing result files. These are easily readable with pandas. See exp-dir_to_graph.py for example
# Results will be in exp_new_features/test_predictions_cotraining_new_features_1000.txt  , 5000, 10000
# Those are CSVs that contain prediction,actuall_class rows, for the test set (without the unlabeled)
# and exp_new_features/unlabel_predictions_cotraining_new_features_1000.txt, 5000, 10000
# Those are CSVs that contain prediction,actuall_class rows, for the unlabeled test
# Get a graph pdf with :
./exp-dir_to_graph.py exp_new_features/ new_features "CoTraining - NEW_features" # last field is the title of the graph
# now there should be a graph PDF at exp_new_features/recall_precision.pdf

#
# run 
#

echo '#####################################__New Features without Co-Training__#####################################'
#New Features CoT-less takes data from exp_new_features
cd new_ft
#launch
./launch.sh
cd ..


echo '#####################################__Co-Training with VCCFeatures__#####################################'
# cotraining with VCC features
# data needed should be in replicate/4_Results/1_Data/2_CT_Input/
mkdir exp_vcc_features/
./cotraining_vcc_features.py replicate/4_Results/1_Data/2_ct_input/ 1000
./cotraining_vcc_features.py replicate/4_Results/1_Data/2_ct_input/ 5000
./cotraining_vcc_features.py replicate/4_Results/1_Data/2_ct_input/ 10000
# Results will be in exp_vcc_features/predictions_cotraining_new_features_1000.txt  , 5000, 10000
./exp-dir_to_graph.py replicate/4_Results/1_Data/2_ct_input/ vcc_features "CoTraining - VCC_features"
# now there should be a graph PDF at replicate/4_Results/1_Data/2_CT_Input/recall_precision.pdf
