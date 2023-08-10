# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : VAD
"""

import collections
import contextlib
import io
import wave
import webrtcvad
from pybaseutils.audio import audio_utils


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def write_wave_buffer(audio, sample_rate):
    """Writes a BytesIO object.

    Takes PCM audio data, and sample rate.
    """
    audio_buffer = io.BytesIO()
    with contextlib.closing(wave.open(audio_buffer, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    audio_buffer.seek(0)
    return audio_buffer


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(vad, frames, sample_rate,
                  frame_duration_ms, padding_duration_ms):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    t_begin, t_end = -1, -1
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                t_begin = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                t_end = frame.timestamp + frame.duration
                triggered = False
                yield [b''.join([f.bytes for f in voiced_frames]), t_begin, t_end]
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        t_end = frame.timestamp + frame.duration
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield [b''.join([f.bytes for f in voiced_frames]), t_begin, t_end]


class AudioVAD(object):
    """端点检测,语音活动检测器,能够将一段语音分割中的静音帧和非静音"""

    def __init__(self, mode=0, frame_duration_ms=30, padding_duration_ms=300):
        """
        语音活动检测器
        get audio chunks
        :param mode: or aggressiveness 敏感系数，取值0-3，越大表示越敏感，越激进，对细微的声音频段都可以分割出来
        :param frame_duration_ms: The frame duration in milliseconds.
        :param padding_duration_ms:  The amount to pad the window, in milliseconds.
        """
        self.mode = mode
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = webrtcvad.Vad(mode=self.mode)

    def vad_from_audio_pcm(self, audio_pcm, sample_rate, buffer=True):
        """
        :param audio_pcm:  PCM audio data,
        :param sample_rate: sample rate
        :param buffer:  True: 'data' is a BytesIO object.
                        False: 'data' is a audio data(ndarray)
        :return:  audio chunks
        """
        frames = frame_generator(frame_duration_ms=self.frame_duration_ms, audio=audio_pcm, sample_rate=sample_rate)
        frames = list(frames)
        chunks = vad_collector(self.vad, frames=frames, sample_rate=sample_rate,
                               frame_duration_ms=self.frame_duration_ms,
                               padding_duration_ms=self.padding_duration_ms)
        for i, chunk in enumerate(chunks):
            if buffer:
                data = write_wave_buffer(chunk[0], sample_rate)
            else:
                data = audio_utils.pcm_bytes2audio_data(chunk[0])
            yield {"data": data, "start": chunk[1], "end": chunk[2], "sample_rate": sample_rate}

    def vad_from_file_mono(self, audio_file, buffer=False):
        """
        :param audio_file:
        :param buffer:  True: 'data' is a BytesIO object.
                        False: 'data' is a audio data(ndarray)
        :return:
        """
        audio_pcm, sample_rate = read_wave(audio_file)
        return self.vad_from_audio_pcm(audio_pcm, sample_rate, buffer=buffer)

    def vad_from_file(self, audio_file, buffer=False):
        """
        :param audio_file:
        :param buffer:  True: 'data' is a BytesIO object.
                        False: 'data' is a audio data(ndarray)
        :return:
        """
        audio_data, sample_rate = audio_utils.read_audio(audio_file, mono=True)
        return self.vad_from_audio_data(audio_data, sample_rate, buffer=buffer)

    def vad_from_audio_data(self, audio_data, sample_rate, buffer=False):
        """
        :param audio_data:
        :param sample_rate:
        :param buffer:  True: 'data' is a BytesIO object.
                        False: 'data' is a audio data(ndarray)
        :return:
        """
        audio_pcm = audio_utils.audio_data2pcm_bytes(audio_data)
        return self.vad_from_audio_pcm(audio_pcm, sample_rate, buffer=buffer)


def example1():
    """暂时不支持单音道"""
    from pybaseutils.audio import audio_utils
    audio_file = "../../data/audio/long_audio.wav"
    mode = 0  # "aggressiveness"
    audio, sample_rate = read_wave(audio_file)
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(vad, frames, sample_rate, 30, 300)
    for i, segment in enumerate(segments):
        seg_data = write_wave_buffer(segment[0], sample_rate)
        time_beg = segment[1]
        time_end = segment[2]
        print("({},{})".format(time_beg, time_end))
        idx_beg = int(time_beg * sample_rate)
        idx_end = int(time_end * sample_rate)
        audio_utils.sound_audio(seg_data)


def example2():
    from pybaseutils.audio import audio_utils

    audio_file = "../../data/audio/bus_chinese.wav"
    # audio_file = "../../data/audio/long_audio.wav"
    vad = AudioVAD()
    segments = vad.vad_from_file(audio_file)
    for i, segment in enumerate(segments):
        sample_rate = segment["sample_rate"]
        seg_data = segment["data"]
        time_beg = segment["start"]
        time_end = segment["end"]
        idx_beg = int(time_beg * sample_rate)
        idx_end = int(time_end * sample_rate)
        print(f"{time_beg}({idx_beg}), {time_end}({idx_end})")
        audio_utils.sound_audio(seg_data)


def example3():
    from pybaseutils.audio import audio_utils
    audio_file = "../../data/audio/bus_chinese.wav"
    # audio_file = "../../data/audio/long_audio.wav"
    vad = AudioVAD()
    audio_data, sample_rate = audio_utils.read_audio(audio_file)
    segments = vad.vad_from_audio_data(audio_data, sample_rate)
    for i, segment in enumerate(segments):
        sample_rate = segment["sample_rate"]
        seg_data = segment["data"]
        time_beg = segment["start"]
        time_end = segment["end"]
        idx_beg = int(time_beg * sample_rate)
        idx_end = int(time_end * sample_rate)
        print(f"{time_beg}({idx_beg}), {time_end}({idx_end})")
        audio_utils.sound_audio(seg_data)


if __name__ == '__main__':
    # example1()
    example2()
    # example3()
