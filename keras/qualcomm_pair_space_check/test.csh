#!/bin/csh

set start_time = `date +%s`
set start_time_string = `date`

set maxCount = 10
set idx = 0

set pyPath = "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/pair_space_check_server.py"

echo "Normal images"
set normal_img_list = `ls dataset/normal/*.jpg`
while ( $idx < $maxCount )
    @ idx++

    set predict = `python3 $pyPath \
                --ip "cam_storage" \
                --img "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/$normal_img_list[$idx]"`

    echo "idx: $idx, predict: $predict"
end

echo ""
set idx = 0

echo "Abnormal images"
set abnormal_img_list = `ls dataset/abnormal/*.jpg`
while ( $idx < $maxCount )
    @ idx++

    set predict = `python3 $pyPath \
                --ip "cam_storage" \
                --img "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/$abnormal_img_list[$idx]"`

    echo "idx: $idx, predict: $predict"
end

echo "Program Success..."

# 실행 시간 계산
set end_time = `date +%s`
set end_time_string = `date`
set elapsed_time = `echo "$end_time - $start_time" | bc`
set htime = `echo "$elapsed_time / 3600" | bc`
set mtime = `echo "($elapsed_time / 60) - ($htime  * 60)" | bc`
set stime = `echo "$elapsed_time - (($elapsed_time/60) * 60)" | bc`

echo "==============================================================="
echo "Start Time : $start_time_string"
echo "End Time : $end_time_string"
echo "Total Time : ${htime} H ${mtime} M ${stime} S"
echo "==============================================================="

exit