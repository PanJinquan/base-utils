# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-11 10:36:28
    @Brief  : pip install ffmpy -i https://pypi.douban.com/simple
"""
import os
from IPython.display import Audio, display
from pybaseutils.base_audio import ffmpeg_utils
from pybaseutils import file_utils, image_utils


def audio_list_dir(audio_dir, video_file):
    """
    特征提取模型
    :param audio_dir:
    :param video_file:
    :return:
    """
    audio_list = file_utils.get_files_list(audio_dir, postfix=["*.mp3"])
    for audio_file in audio_list:
        video_out = audio_file.replace(".mp3", ".mp4")
        ffmpeg_utils.merge_video_audio(video_file, audio_file, video_out)


if __name__ == '__main__':
    video_file = "../data/video/kunkun_cut.mp4"
    audio_file = "../data/video/kunkun_cut.mp3"
    # ffmpeg_utils.extract_video_audio(video_file, audio_file)

    video_file = "../data/video/test-video.mp4"
    audio_file = "../data/video/kunkun_cut.mp3"
    video_out = "../data/video/test-video-result.mp4"
    # ffmpeg_utils.merge_video_audio(video_file, audio_file, video_out)
    # display(Audio(filename=audio_file))
    # print("OK")

    audio_dir = "/home/dm/nasdata/Project/SadTalker/TTS-Demo/azure/"
    video_file = "/home/dm/nasdata/Project/SadTalker/TTS-Demo/轮.mp4"
    audio_list_dir(audio_dir, video_file)
