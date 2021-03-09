#!/bin/bash
set -e
set -x

cd 2_Tools

#
python3 vcc2.py

#
#To launch replication of VCCFinder
python3 run_sally_tracks_id.py

#
./recompose.sh

#To generate data to be used for data generated adding cotraining
python3 sklearn_drawNt10.py

cd ..

