#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=forward        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=02:00:00           # set total run time limit for job


################## BASH SCRIPT ##################

# Define Bash variables
period="0.2"  # seconds
parentdir="07-16-2022"

# run pyscript
for i in {0..6}; do
  for j in {0..9}; do
    thisdt=`echo "scale=10; 0.01-(0.001*"$j")" | bc`
    thisnumts=`echo "scale=0; "$period"/"$thisdt"" | bc`
    mkdir -p ./"$parentdir"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
    python tdse-wave-function-main-script-"$SLURM_JOB_NAME"-kbc.py ./"$parentdir"/numts-"$thisnumts"_dt-"$thisdt" "$i" "$thisnumts" "$thisdt"
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

