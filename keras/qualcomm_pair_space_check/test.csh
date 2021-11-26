#!/bin/csh

set maxCount = 5
set idx = 0

set pyPath = "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/pair_space_check_server.py"

echo "Normal images"
while ( $idx < $maxCount )
    @ idx++

    set predict = `python3 $pyPath \
                --img "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/dataset/normal/${idx}.jpg"`

    echo "idx: $idx, predict: $predict"
end

echo ""
echo "Abnormal images"
set idx = 0
while ( $idx < $maxCount )
    @ idx++

    set predict = `python3 $pyPath \
                --img "/workfile/jinyul/python_project/keras/qualcomm_pair_space_check/dataset/abnormal/${idx}.jpg"`

    echo "idx: $idx, predict: $predict"
end
