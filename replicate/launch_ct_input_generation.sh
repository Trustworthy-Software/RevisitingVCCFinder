#!/bin/bash
#################
# This chain of commands enable to recreate the inputs for the 
# cotraining using NewFeatures
#################
set -e
set -x

cd 2_Tools

# If this script was launched by do_everything.sh, this is already done
#python3 vcc2.py

#
python3 organizer.py

#
python3 ct_run_sally_tracks_id.py

#
./ct_recompose.sh

cd ..
#Results are in 4_Results/1_Data/2_ct_input
