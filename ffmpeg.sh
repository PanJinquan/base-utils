#!/usr/bin/env bash
# ffmpeg -loglevel 0 -threads 1 -re -ss 0 -i path/to/video.mp4 -vf scale=-1:-1, fps=1 -q:v 20 -vcodec mjpeg -an -vsync 0
video_file=/media/PKing/新加卷1/SDK/base-utils/data/video/kunkun_cut.mp4
outputs=outputs/dataset/image_%04d.jpg
parent_dir=$(dirname $outputs)
echo $parent_dir
mkdir -p $parent_dir
ffmpeg -i  $video_file -r 1 -q:v 2  -vcodec mjpeg $outputs
