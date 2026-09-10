# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://zhuanlan.zhihu.com/p/437580699
"""
import os
from tqdm import tqdm
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from pybaseutils.base_audio import audio_utils
import soundfile as sf

# key = "50314f04a9ec4b0aa3241691c5d67d12"
# region = "westus"
key = "d0ab1c2e99074582b54d9ee8f0e81fca"
region = "eastus"


class AzureService(object):
    def __init__(self, key=key, region=region, audio_format="Riff24Khz16BitMonoPcm", language="1", speaker=False):
        """
        :param key:
        :param region:
        :param language: (1 for Chinese, 2 for English)
        :param speaker:
        """
        # Convert language option to language code
        self.language_code = "zh-CN" if language == "1" else "en-US"
        self.voice = "zh-CN-XiaochenNeural" if language == "1" else "en-US-JennyNeural"
        self.audio_format = audio_format
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.set_speech_synthesis_output_format(
            format_id=speechsdk.SpeechSynthesisOutputFormat[self.audio_format])
        if speaker:
            audio_config = AudioOutputConfig(use_default_speaker=True)
        else:
            audio_config = None
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    def synthesizer_request(self, text, rate=1.0):
        """
        :param text: 文本
        :param rate: 语速
        :return:
        """
        # 语速不普通的慢20%
        ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{self.voice}">
                <mstts:express-as style="chat">
                    <prosody rate="{rate}">
                    {text}
                    </prosody>
                </mstts:express-as>
            </voice>
            </speak>
            """
        # 将SSML文本合成为语音
        result = self.synthesizer.speak_ssml_async(ssml).get()
        return result

    def get_tts_result(self, text, rate=1.0):
        """
        :param text: 文本
        :param rate: 语速
        :return:
        """
        result = self.synthesizer_request(text, rate=rate)
        audio_data, samplerate = audio_utils.audio_bytes2array(result.audio_data1)
        audio_buffer = audio_utils.audio_write("", audio_data, samplerate, format="wav", buffer=True)
        return audio_buffer

    def tts_file(self, filename, output=None):
        if not output: output = os.path.dirname(filename)
        # Open the input file and read its contents
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Loop through each line and convert it to an audio file
        for i, line in tqdm(enumerate(lines), total=len(lines), desc="Converting text to speech"):
            # Ignore empty lines
            if not line.strip():
                continue
            # Truncate line if it exceeds the character limit
            line = line[:5000]
            result = self.synthesizer_request(line)
            audio_file = os.path.basename(filename).split(".")[0] + "_{:0=3d}.wav".format(i)
            audio_file = os.path.join(output, audio_file)
            audio_data, samplerate = audio_utils.audio_bytes2array(result.audio_data1)
            audio_utils.audio_write(audio_file, audio_data, samplerate, format="wav", buffer=False)
            audio_utils.write_bin_file(audio_file, result.audio_data1)
            print("save file in :{}".format(audio_file))


azure_service = AzureService(speaker=False)

if __name__ == "__main__":
    filename = "./0001.txt"
    azure_service.tts_file(filename)
