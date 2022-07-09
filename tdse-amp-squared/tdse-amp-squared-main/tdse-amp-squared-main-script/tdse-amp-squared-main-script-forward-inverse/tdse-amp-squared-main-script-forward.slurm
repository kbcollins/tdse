#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=forward        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=02:00:00           # set total run time limit for job
#SBATCH --array=0-6               # job array with index values 0, 1, 2, 3, 4, 5, 6
                                  # use $SLURM_ARRAY_TASK_ID to pass the array number as an argument


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

# test: keep numts constant, change dt
for thisdt in "${dtargs[@]}"; do
  thisnumts=20
  mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$SLURM_ARRAY_TASK_ID"
  python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$SLURM_ARRAY_TASK_ID" $thisnumts $thisdt
done

# test: keep dt constant, change numts
for thisdt in "${numtsargs[@]:1}"; do
  thisdt="1e-2"
  mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$SLURM_ARRAY_TASK_ID"
  python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$SLURM_ARRAY_TASK_ID" $thisnumts $thisdt
done

# test: change numts and dt
for i in "${!dtargs[@]:1}"; do
  thisdt="${dtargs[i]}"
  thisnumts="${numtsargs[i]}"
  mkdir -p ./"$date"/numts-"$thisnumts"_dt-"$thisdt"/v"$SLURM_ARRAY_TASK_ID"
  python tdse-amp-squared-main-script-forward-kbc.py ./"$date"/numts-"$thisnumts"_dt-"$thisdt" "$SLURM_ARRAY_TASK_ID" $thisnumts $thisdt
done


#for i in {0..6}
#do
#  dt=
#	mkdir -p ./07-08-2022/numts-20_dt-"$dt"/v"$i"
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-"$dt" $i 20 $dt
#
#	dt=
#	mkdir -p ./07-08-2022/numts-20_dt-"$dt"/v"$i"
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-"$dt" $i 20 $dt
#
#	dt=
#	mkdir -p ./07-08-2022/numts-20_dt-"$dt"/v"$i"
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-"$dt" $i 20 $dt
#
#  dt=
#	mkdir -p ./07-08-2022/numts-20_dt-"$dt"/v"$i"
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-"$dt" $i 20 $dt
#
#	dt=
#	mkdir -p ./07-08-2022/numts-20_dt-"$dt"/v"$i"
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-"$dt" $i 20 $dt
#done


# test: keep time-step size constant, change number of time steps
for i in {0..6}
do
  numts=50
	mkdir -p ./07-08-2022/numts_numts-"$numts"_dt-1e-2/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-"$numts"_dt-1e-2 $i $numts 1e-2

	numts=80
	mkdir -p ./07-08-2022/numts_numts-"$numts"_dt-1e-2/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-"$numts"_dt-1e-2 $i $numts 1e-2

	numts=110
	mkdir -p ./07-08-2022/numts_numts-"$numts"_dt-1e-2/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-"$numts"_dt-1e-2 $i $numts 1e-2

	numts=140
	mkdir -p ./07-08-2022/numts_numts-"$numts"_dt-1e-2/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-"$numts"_dt-1e-2 $i $numts 1e-2
done


# test: change both the number of time steps and time-step size
for i in {0..6}
do
  dt="1e-2"
  numts=20
	mkdir -p ./07-08-2022/numts-"$numts"_dt-"$dt"/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-"$numts"_dt-"$dt" $i $numts $dt

  dt="1e-3"
  numts=50
	mkdir -p ./07-08-2022/numts-"$numts"_dt-"$dt"/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-"$numts"_dt-"$dt" $i $numts $dt

  dt="1e-4"
  numts=80
	mkdir -p ./07-08-2022/numts-"$numts"_dt-"$dt"/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-"$numts"_dt-"$dt" $i $numts $dt

  dt="1e-5"
  numts=110
	mkdir -p ./07-08-2022/numts-"$numts"_dt-"$dt"/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-"$numts"_dt-"$dt" $i $numts $dt

	dt="1e-6"
  numts=140
	mkdir -p ./07-08-2022/numts-"$numts"_dt-"$dt"/v"$i"
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-"$numts"_dt-"$dt" $i $numts $dt
done