# FFmpeg 常用的工具使用方法

- 安装FFmpeg

```bash
# 安装ffmpeg
sudo apt update && sudo apt install ffmpeg
# 验证安装
ffmpeg -version
```

- 视频抽帧

```bash
# ffmpeg -loglevel 0 -threads 1 -re -ss 0 -i path/to/video.mp4 -vf scale=-1:-1, fps=1 -q:v 20 -vcodec mjpeg -an -vsync 0
video_file=/media/PKing/新加卷1/SDK/base-utils/data/video/kunkun_cut.mp4
outputs=outputs/dataset/image_%04d.jpg
parent_dir=$(dirname $outputs)
echo $parent_dir
mkdir -p $parent_dir
ffmpeg -i  $video_file -r 1 -q:v 2  -vcodec mjpeg $outputs

```

- USB摄像头录制

```bash

# TODO 安装ffmpeg
sudo apt update && sudo apt install ffmpeg 

# ls /dev/video* # 命令查看所有视频设备
# -f v4l2  -i /dev/video0    # 指定视频输入源，default是系统默认的V4L2设备，通常是/dev/video0
# -f pulse -i default        # 指定音频输入源,从PulseAudio默认设备捕获音频
# -s  1920x1080              # 设置视频分辨率(size)为 1920×1080（全高清）
# -c:v libx264               # 指定视频编码器(codec:video)，libx264通用性好；h264_v4l2m2m用于树莓派硬件加速
# -c:a aac                   # 指定音频编码器(codec:audio)，aac通用性好；opus用于低延迟
# -r 25                      # 设置帧率(25fps)
# -preset medium             # 调节编码速度与压缩率的平衡(ultrafast/medium/veryslow压缩率最高)
# -f segment                 # 启用分段输出
# -segment_time 1800         # 每个分段1800秒（30分钟）
# -reset_timestamps 1        # 每个分段重置时间戳，避免播放问题
# -segment_format mp4        # 输出格式为MP4
# -strftime 1                # 在文件名中使用strftime格式
# "%Y%m%d_%H%M%S.mp4"        # 分段文件名格式

# TODO 使用ffmpeg录制视频，mp4中断后，会导致文件损坏无法播放,建议先保存为mkv和flv格式，再转换为mp4格式
ffmpeg -f v4l2 -i /dev/video0 -f pulse -i default \
-s 1920x1080 -c:v libx264 -c:a aac -r 25 \
-preset medium -f segment -segment_time 1800 \
-reset_timestamps 1 -segment_format matroska \
-strftime 1 \
"%Y%m%d_%H%M%S.mkv"

# TODO 批量将MKV文件转为MP4文件
for file in *.mkv; do
    ffmpeg -i "$file" -c:v libx264 -crf 24 -preset medium -c:a aac -b:a 128k  -movflags +faststart "${file%.mkv}.mp4"
done

for file in *.avi; do
    ffmpeg -i "$file" -c:v libx264 -crf 24 -preset medium -c:a aac -b:a 128k  -movflags +faststart "${file%.avi}.mp4"
done
# TODO mp4强行中断，会导致文件损坏无法播放,建议MKV和flv格式
ffmpeg -f v4l2  -i /dev/video0 -f pulse -i default  \
-s  1920x1080 -c:v libx264 -c:a aac -r 25 \
-preset medium -f segment -segment_time 60 \
-reset_timestamps 1 \
-segment_format mp4 -strftime 1 \
"%Y%m%d_%H%M%S.mp4"


ffmpeg -f v4l2 -i /dev/video0 -f pulse -i default \
-s 1920x1080 -c:v libx264 -c:a aac -r 25 \
-preset medium -f segment -segment_time 1800 \
-reset_timestamps 1 -segment_format flv -strftime 1 \
"%Y%m%d_%H%M%S.flv"

```