
#qsub -q all.q -cwd -t 1-3 test.sh
python2.6 clustertest.py $SGE_TASK_ID
