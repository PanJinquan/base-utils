# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-07 16:45:52
    @Brief  :
"""
import os
from pydub import AudioSegment
import math


class SplitWavAudioMubin():
    def __init__(self, filename):
        self.filename = filename
        self.audio = AudioSegment.from_wav(self.filename)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(split_filename, format="wav")

    def multiple_split(self, sec_per_split):
        total_sec = math.ceil(self.get_duration())
        filename = os.path.basename(self.filename)
        for i in range(0, total_sec, sec_per_split):
            split_fn = 'split_' + str(i) + '_' + filename
            self.single_split(i, i + sec_per_split, split_fn)

            print(str(i) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splitted successfully')


if __name__ == "__main__":
    filename = '/media/PKing/新加卷1/SDK/base-utils/data/audio/long_audio.wav'
    split_wav = SplitWavAudioMubin(filename)
    split_wav.multiple_split(sec_per_split=10)
