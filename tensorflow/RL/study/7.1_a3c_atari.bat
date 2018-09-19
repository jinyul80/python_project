@ECHO OFF
PUSHD %~DP0

TITLE "Tensorflow distribute program"

SET FOR_NUM=6

for /l %%a in (0,1,0) do start python 7.1_a3c_atari_distribute.py --job_name=ps --task_index=%%a

timeout /t 10

for /l %%a in (0,1,%FOR_NUM%) do start python 7.1_a3c_atari_distribute.py --job_name=worker --task_index=%%a && timeout /t 1