#qsub -q all.q -cwd -t 1-180 experiment.sh

python2.6 all_runs.py $SGE_TASK_ID
