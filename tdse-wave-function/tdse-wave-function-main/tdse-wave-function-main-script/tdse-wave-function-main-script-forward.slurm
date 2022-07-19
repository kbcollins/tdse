#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=forward        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=02:00:00           # set total run time limit for job


######### HELPFUL ENVIRONMENT VARIABLES #########

# $SLURM_JOB_NAME - short name of job set by --job-name


################## BASH VARIABLES ##################
# These variables define where data is loaded
# from and/or saved to, as well as, define the
# test parameters

# directory set up like:
# ./"$parentdir"/v"$thisv"/"$thistrial"/"$thismodel"/results-"$SLURM_JOB_NAME"
# - The "$parentdir" directory can be called w/e is most
#   convenient. I like the idea of experiment name.
# - numts-"$thisnumts"_dt-"$thisdt" is the directory
#   containing trial specific results and data
#   - A trial is identified by the specific computational
#     parameters used (like: L, numx, numfour, model, etc.)
#   - Trial directories will contain a result directory,
#     named like results-"$SLURM_JOB_NAME"
#     - results folders may contain: graphs and any output
#       that might be used by other scripts (e.g.,
#       results-converge would contain bestthetav.npy and
#       bestthetaprop.npy which are used by the propagation test)
# - Each v# directory will contain CSV files that aggregate
#   important results from each trial
parentdir="dt"

period="0.2"  # seconds


################## BASH SCRIPT ##################
# The Bash script that calls the desired Pyscript(s)
# and feeds it(them) the test parameters (i.e.,
# the Pyscript defines what test procedure,
# running the Pyscript is conducting one trial,
# each experiment consists of many trials where one
# or a few parameters are changed each trial

# run pyscript
for thisv in {0..6}; do
  # store path for current potential
  thispath=./"$parentdir"/v"$thisv"

  for j in {0..20}; do
    # scale tells bc how many decimals it should
    # return, the '|' is a "pipe" -it directs the
    # output of the echo command as the input of
    # the bc command
    thisdt=`echo "scale=10; 0.001+(0.0001*"$j")" | bc`
    thisnumts=`echo "scale=0; "$period"/"$thisdt"" | bc`

    # ./parent/potential/trial/
    thistrialpath=./"$parentdir"/v"$thisv"/trial_nts-"$thisnumts"_dt-"$thisdt"
    mkdir -p "$thistrialpath"

    # run the Python scrip
    python tdse-wave-function-main-script-"$SLURM_JOB_NAME"-kbc.py "$thistrialpath" "$thisv" "$thisnumts" "$thisdt"
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

