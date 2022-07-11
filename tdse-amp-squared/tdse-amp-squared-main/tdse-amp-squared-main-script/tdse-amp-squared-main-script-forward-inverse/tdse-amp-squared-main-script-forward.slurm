#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=forward        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=02:00:00           # set total run time limit for job


################## BASH SCRIPT ##################

# Normal run of forward script
#for i in {0..6}
#do
##	echo $i
#	mkdir -p ./07-08-2022_numts-20_dt-1e-2/v$i
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022_numts-20_dt-1e-2 $i 20 1e-2
#done


# Define Bash variables
date="07-11-2022"
period="0.2"  # seconds

# run pyscript
for i in {0..6}; do
  for j in {0..10}; do
    thisdt=`echo "scale=10; 1/(2^"$j")" | bc`
    thisnumts=`echo "scale=0; "$period"/"$thisdt"" | bc`
    mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
    python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
  done
done


## Define Bash variables
#date="07-11-2022"
#numtsargs=(20 50 80 110 140)
#dtargs=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6")
#
## run pyscript
#for i in {0..6}; do
#  for j in "${!dtargs[@]}"; do
#    thisdt="${dtargs[j]}"
#    for k in "${!numtsargs[@]}"; do
#      thisnumts="${numtsargs[k]}"
#      mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
#      python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
#    done
#  done
#done

