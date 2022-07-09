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
date="07-08-2022"
numtsargs=(20 50 80 110 140)
dtargs=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6")

# run pyscript
for i in {0..6}; do
  # test: keep numts constant, change dt
  for thisdt in "${dtargs[@]}"; do
    thisnumts=20
    mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
    python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
  done

  # test: keep dt constant, change numts
  for thisdt in "${numtsargs[@]:1}"; do
    thisdt="1e-2"
    mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
    python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
  done

  # test: change numts and dt
  for j in "${!dtargs[@]:1}"; do
    thisdt="${dtargs[j]}"
    thisnumts="${numtsargs[j]}"
    mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$i"
    python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$i" $thisnumts $thisdt
  done
done
