@ECHO OFF
PUSHD %~DP0

TITLE "Tensorflow distribute program"

SET WORKER_NUM=3
SET FOR_NUM=2

start python 7.1_a3c_atari_distibute.py --job_name=ps --worker_hosts_num=%WORKER_NUM%
sleep 10

for /l %%a in (0,1,%FOR_NUM%) do start python 7.1_a3c_atari_distibute.py --job_name=worker --worker_hosts_num=%WORKER_NUM% --task_index=%%a
