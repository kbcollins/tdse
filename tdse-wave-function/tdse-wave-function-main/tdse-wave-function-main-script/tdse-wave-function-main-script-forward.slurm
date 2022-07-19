#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=forward        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=02:00:00           # set total run time limit for job


################## BASH VARIABLES ##################
# Define Bash variables used to define where data
# is stored and saved, as well as, test parameters

# directory set up like: ./"$parentdir"/v"$potential"/
# - Inside v"$potential" is the training data and result folders
# - Result folders are named like:
#   ./"$parentdir"/v"$potential"/results-"$SLURM_JOB_NAME"
# - Result folders contain:
#   - graphs
#   - any output that might be used by other scripts
#     (e.g., results-converge would contain bestthetav.npy
#     and bestthetaprop.npy which are used by the propagation
#     test)
parentdir="dt"

period="0.2"  # seconds


################## EXPERIMENT SETUP ##################
# The Bash script that calls the desired Pyscript(s)
# and feeds it(them) the test parameters (i.e.,
# the Pyscript defines what test procedure,
# running the Pyscript is conducting one trial,
# each experiment consists of many trials where one
# or a few parameters are changed each trial

# run pyscript
for thisv in {0..6}; do
  for j in {0..20}; do
    # scale tells bc how many decimals it should
    # return, the '|' is a "pipe" -it directs the
    # output of the echo command as the input of
    # the bc command
    thisdt=`echo "scale=10; 0.001+(0.0001*"$j")" | bc`
    thisnumts=`echo "scale=0; "$period"/"$thisdt"" | bc`
    # ./parent/potential/experiment/test-results
    thispath=./"$parentdir"/v"$thisv"/numts-"$thisnumts"_dt-"$thisdt"
    mkdir -p "$thispath"
    python tdse-wave-function-main-script-"$SLURM_JOB_NAME"-kbc.py "$thispath" "$thisv" "$thisnumts" "$thisdt"
#    python tdse-wave-function-main-script-forward-kbc.py ./"$parentdir"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
  done
done


## Define Bash variables
#parentdir="07-11-2022"
#numtsargs=(20 50 80 110 140)
#dtargs=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6")
#
## run pyscript
#for i in {0..6}; do
#  for j in "${!dtargs[@]}"; do
#    thisdt="${dtargs[j]}"
#    for k in "${!numtsargs[@]}"; do
#      thisnumts="${numtsargs[k]}"
#      mkdir -p ./"$parentdir"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
#      python tdse-wave-function-main-script-forward-kbc.py ./"$parentdir"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
#    done
#  done
#done

