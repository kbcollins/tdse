#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-inverse
#SBATCH --output='out-inverse'
#SBATCH --time=02:35:00

#python tdse-amp-squared-main-script-inverse-kbc.py ./v0

#for i in {0..6}
#do
##	echo $i
#	python tdse-amp-squared-main-script-inverse-kbc.py ./v$i
#done


for i in {0..6}
do
  dt='1e-2'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i/results-inverse
	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-20_dt-$dt/v$i

	dt='1e-3'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i/results-inverse
	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-20_dt-$dt/v$i

	dt='1e-4'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i/results-inverse
	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-20_dt-$dt/v$i

  dt='1e-5'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i/results-inverse
	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-20_dt-$dt/v$i

	dt='1e-6'
	mkdir -p ./07-08-2022/numts-20_dt-$dt/v$i/results-inverse
	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-20_dt-$dt/v$i
done


## test: keep time-step size constant, change number of time steps
#for i in {0..6}
#do
#  numts=50
#	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
#
#	numts=80
#	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
#
#	numts=110
#	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
#
#	numts=140
#	mkdir -p ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts_numts-$numts_dt-1e-2/v$i
#done
#
#
## test: change both the number of time steps and time-step size
#for i in {0..6}
#do
#  dt='1e-2'
#  numts=20
#	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-$numts_dt-$dt/v$i
#
#  dt='1e-3'
#  numts=50
#	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-$numts_dt-$dt/v$i
#
#  dt='1e-4'
#  numts=80
#	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-$numts_dt-$dt/v$i
#
#  dt='1e-5'
#  numts=110
#	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-$numts_dt-$dt/v$i
#
#	dt='1e-6'
#  numts=140
#	mkdir -p ./07-08-2022/numts-$numts_dt-$dt/v$i/results-inverse
#	python tdse-amp-squared-main-script-inverse-kbc.py ./07-08-2022/numts-$numts_dt-$dt/v$i
#done
