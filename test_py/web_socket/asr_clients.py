import websockets
import asyncio
import wave
import time
import numpy as np
import librosa
from pybaseutils.audio import audio_utils


async def test_asr(url, audio_file, samplerate=16000, chunk_size=9600):
    """测试ASR服务
    Args:
        url: ASR服务的WebSocket地址 (e.g. ws://localhost:8765/asr)
        audio_file: 要测试的音频文件路径
        chunk_size: 每次发送的音频块大小(默认9600采样点，对应16kHz采样率下600ms音频(每帧960采样点 × 10帧)
    """
    # 连接WebSocket服务器
    async with websockets.connect(url) as websocket:
        # 读取WAV文件模拟实时音频流
        try:
            frames, sr = librosa.load(audio_file, sr=samplerate, mono=True)  # 非常耗时
            i = 0
            while True:
                # 将音频数据分割成固定大小的块 (chunk) 模拟实时流式传输
                chunk = frames[i * chunk_size:(i + 1) * chunk_size]
                chunk = audio_utils.audio_data2pcm_bytes(chunk)
                await websocket.send(chunk)
                result = await websocket.recv() # TODO 若没有识别结果，会阻塞
                print(result)
                time.sleep(0.01)
                i = i + 1
        except Exception as e:
            print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    audio_file = "/home/PKing/nasdata/Project/ASR/FunASR/data/list/BAC009S0764W0121.wav"
    url = "ws://localhost:8765/asr"
    asr = test_asr(url, audio_file)
    asyncio.run(asr)
