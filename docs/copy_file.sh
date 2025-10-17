#!/usr/bin/env bash
# /nasdata/atp/data/panjinquan/dataset-dmai
dst=/edudata/dataset/AIJE
#scp -r $src $dst

src=(
/nasdata/atp/data/panjinquan/dataset-dmai/AIJE/【TOP】技能人才系统_数据集管理
)

for ((i=0; i<${#src[@]}; i++)); do
    printf "copy %20s  to  %s\n" "${src[$i]}" "${dst}"
	scp -r "${src[$i]}" "${dst}"
done

