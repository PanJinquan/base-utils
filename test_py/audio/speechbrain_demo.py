# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-07 15:21:52
    @Brief  :
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import interfaces
from pybaseutils.audio import audio_utils, vad_utils


class SpeechBrain(object):
    def __init__(self, source,
                 hparams_file="hyperparams.yaml",
                 pymodule_file="custom.py",
                 savedir="./savedir",
                 use_ctc=False,
                 device="cpu"):
        """
        :param source:
        :param hparams_file:
        :param pymodule_file:
        :param savedir:
        :param use_ctc:
        # """
        self.use_ctc = use_ctc
        if use_ctc:
            # asr-wav2vec2-ctc-aishell/custom.py
            self.asr_model = interfaces.foreign_class(source=source,
                                                      hparams_file=hparams_file,
                                                      pymodule_file=pymodule_file,
                                                      savedir=savedir,
                                                      classname="CustomEncoderDecoderASR",
                                                      run_opts={"device": "cuda"})
        else:
            # "speechbrain/pretrained/interfaces.py"
            self.asr_model = EncoderDecoderASR.from_hparams(source=source,
                                                            hparams_file=hparams_file,
                                                            pymodule_file=pymodule_file,
                                                            savedir=savedir,
                                                            run_opts={"device": "cuda"})

    def inference(self, waveform: np.ndarray, sample_rate=16000):
        """Run inference w/ ASR model
        Args:
            waves (torch.Tensor): tensor representation of speech segment
        Returns:
            str: ASR result
        """
        # waveform = waveform.transpose(1,0)
        waveform = torch.from_numpy(waveform)  # (nums,channel)
        waveform = self.asr_model.audio_normalizer(waveform, sample_rate)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        if self.use_ctc:
            predicted_words = self.asr_model.transcribe_batch(batch, rel_length)
            pred = predicted_words[0]
        else:
            predicted_words, predicted_tokens = self.asr_model.transcribe_batch(batch, rel_length)
            pred = predicted_words[0]
        if isinstance(pred, list): pred = "".join(pred)
        pred = pred.replace(' ', '')
        return pred

    def inference_file(self, file: str):
        """Run inference w/ ASR model
        Args:
            waves (torch.Tensor): tensor representation of speech segment
        Returns:
            str: ASR result
        """
        pred = self.asr_model.transcribe_file(file)
        if isinstance(pred, list): pred = "".join(pred)
        pred = pred.replace(' ', '')
        return pred


if __name__ == '__main__':
    """
    source需要提供model.ckpt，normalizer.ckpt，tokenizer.ckpt和hyperparams.yaml四个文件
    """
    audio_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/long_audio.wav"
    audio_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/bus_chinese.wav"
    # source = "/home/PKing/nasdata/dataset-dmai/AIJE/audio/model/asr-wav2vec2-transformer-aishell"
    # source = "/home/PKing/nasdata/dataset-dmai/AIJE/audio/model/asr-transformer-aishell"
    source = "/home/PKing/nasdata/dataset-dmai/AIJE/audio/model/asr-wav2vec2-ctc-aishell"  # use_ctc=True
    sr = 16000
    asr_model = SpeechBrain(source=source, use_ctc=True)
    audio_data, sample_rate = audio_utils.read_audio(audio_file, sr=sr, mono=True)
    vad = vad_utils.AudioVAD()
    segments = vad.vad_from_audio_data(audio_data, sample_rate, buffer=False)
    for i, segment in enumerate(segments):
        sample_rate = segment["sample_rate"]
        seg_data = segment["data"]
        time_beg = segment["start"]
        time_end = segment["end"]
        idx_beg = int(time_beg * sample_rate)
        idx_end = int(time_end * sample_rate)
        print(f"{time_beg}({idx_beg}), {time_end}({idx_end})")
        print(asr_model.inference(seg_data, sample_rate=sample_rate))
        tmp_file = "seg_data.wav"
        audio_utils.audio_write(tmp_file, audio_data=seg_data, sr=sample_rate)
        print(asr_model.inference_file(tmp_file))
        audio_utils.sound_audio(seg_data)
