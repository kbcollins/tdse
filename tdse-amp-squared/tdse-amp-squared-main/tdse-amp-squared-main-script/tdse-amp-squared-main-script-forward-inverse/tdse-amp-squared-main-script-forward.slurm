#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --output='out-forward'
#SBATCH --time=00:35:00


#mkdir -p v0
#python tdse-amp-squared-main-script-forward-kbc.py 0


# Normal run of forward script
#for i in {0..6}
#do
##	echo $i
#	mkdir -p ./07-08-2022_numts-20_dt-1e-2/v$i
#	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022_numts-20_dt-1e-2 $i 20 1e-2
#done


# test: keep number of time-steps constant change time-step size
for i in {0..6}
do
  dt='1e-2'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-$dt $i 20 $dt

	dt='1e-3'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-$dt $i 20 $dt

	dt='1e-4'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-$dt $i 20 $dt

  dt='1e-5'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-$dt $i 20 $dt

	dt='1e-6'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-20_dt-$dt $i 20 $dt
done


# test: keep time-step size constant, change number of time steps
for i in {0..6}
do
  numts=50
	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2 $i $numts 1e-2

	numts=80
	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2 $i $numts 1e-2

	numts=110
  mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2 $i $numts 1e-2

	numts=140
  mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2 $i $numts 1e-2
done


# test: change both the number of time steps and time-step size
for i in {0..6}
do
  dt='1e-2'
  numts=20
	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-$numts_dt-$dt $i $numts $dt

  dt='1e-3'
  numts=50
	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-$numts_dt-$dt $i $numts $dt

  dt='1e-4'
  numts=80
	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-$numts_dt-$dt $i $numts $dt

  dt='1e-5'
  numts=110
	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-$numts_dt-$dt $i $numts $dt

	dt='1e-6'
  numts=140
	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022/numts-$numts_dt-$dt $i $numts $dt
done