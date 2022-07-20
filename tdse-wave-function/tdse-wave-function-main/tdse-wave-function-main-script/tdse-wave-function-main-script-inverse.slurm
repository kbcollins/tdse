#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=inverse        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=12:00:00           # set total run time limit for job


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
# - Each v# directory will contain .txt files that aggregate
#   important results from each trial
parentdir="dt"

modellist=("fourier" "cheby")


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
#  echo "$thispath"

  # store the directory contents of ./"$parentdir"/v"$thisv"
  # as a bash array
  # the -d flag tells ls to only give directories
  filelist=( $(ls "$thispath" | grep "trial*") )
#  echo "${filelist[@]}"

  # make/clear the agg-results-"$SLURM_JOB_NAME".txt file (the : is
  # the Bash noop or null command and the > redirects the
  # output of the noop to the file and overwrites its contents)
  : > "$thispath"/agg-results-"$SLURM_JOB_NAME".txt
  echo "L, numx, numfour, numts, dt, num a0, model, opt success, val obj, num opt itr, opt time, l2 err, l-inf err, tot time" > "$thispath"/agg-results-"$SLURM_JOB_NAME".txt

  # this loops through all possible models
  for thismodel in "${modellist[@]}"; do
    # this loops through all of the files inside of
    # ./"$parentdir"/v"$thisv"/"$thistrial"
    for thistrial in "${filelist[@]}"; do
      # ./"$parentdir"/v"$thisv"/"$thistrial"
      thistrialpath="$thispath"/"$thistrial"
#      echo "$thistrialpath"
      # make the results directory
      # ./"$parentdir"/v"$thisv"/"$thistrial"/"$thismodel"/results-"$SLURM_JOB_NAME"
      mkdir -p "$thistrialpath"/"$thismodel"/results-"$SLURM_JOB_NAME"
    python tdse-wave-function-main-script-"$SLURM_JOB_NAME"-kbc.py "$thismodel" "$thistrialpath"
    done

  done

done