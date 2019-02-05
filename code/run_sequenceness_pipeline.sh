#!/bin/bash -l

#$ -S /bin/bash

#$ -l mem=25G
#$ -l h_rt=2:30:00
#$ -l tmpfs=20G

#$ -wd ~/Scratch/replay_aversive_learning

#$ -o ~/Scratch/replay_aversive_learning
#$ -e ~/Scratch/replay_aversive_learning


# Arguments = data directory, subject ID, blink component id
# cp -a $1/sub-$2 $TMPDIR

source activate mne
which python
python code/run_sequenceness_pipeline.py data $2 $3
