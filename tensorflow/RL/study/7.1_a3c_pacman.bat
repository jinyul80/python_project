@ECHO OFF
PUSHD %~DP0

TITLE "Tensorflow distribute program"

SET FOR_NUM=5

start python 7.1_a3c_pacman_distribute.py --job_name=ps

timeout /t 10

for /l %%a in (0,1,%FOR_NUM%) do start python 7.1_a3c_pacman_distribute.py --job_name=worker --task_index=%%a && timeout /t 1

