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
import sys
import wave
import webrtcvad


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


class AudioProcess:
    def __init__(self, aggressiveness, frame_duration_ms, padding_duration_ms) -> None:
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.nums_segments = 0

    def __call__(self, wav_path):
        audio, sample_rate = read_wave(wav_path)
        vad = webrtcvad.Vad(self.aggressiveness)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(vad, frames, sample_rate, frame_duration_ms=30, padding_duration_ms=300)
        for i, segment in enumerate(segments):
            yield [write_wave_buffer(segment[0], sample_rate), segment[1], segment[2]]

    def get_segments(self, wav_path):
        """
        src_data = segment[0]
        time_beg = segment[1]
        time_end = segment[2]
        :param wav_path:
        :return:
        """
        audio, sample_rate = read_wave(wav_path)
        vad = webrtcvad.Vad(self.aggressiveness)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(vad, frames, sample_rate, frame_duration_ms=30, padding_duration_ms=300)
        result = []
        for i, segment in enumerate(segments):
            result.append([write_wave_buffer(segment[0], sample_rate), segment[1], segment[2]])
        return result


def example1():
    """暂时不支持单音道"""
    from pybaseutils.base_audio import audio_utils
    audio_file = "../../data/audio/bus_chinese.wav"
    # audio_file = "/home/PKing/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-maker/modules/tts/data/轮-test.wav"
    mode = 0  # "aggressiveness"
    audio, sample_rate = read_wave(audio_file)
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    target_sr = 48000
    segments = vad_collector(vad, frames, sample_rate, 30, 300)
    for i, segment in enumerate(segments):
        seg_data = write_wave_buffer(segment[0], sample_rate)
        time_beg = segment[1]
        time_end = segment[2]
        print("({},{})".format(time_beg, time_end))
        idx_beg = int(time_beg * target_sr)
        idx_end = int(time_end * target_sr)
        audio_utils.sound_audio(seg_data)


def example2():
    """暂时不支持单音道"""
    from pybaseutils.base_audio import audio_utils

    audio_file = "../../data/audio/bus_chinese.wav"
    # audio_file = "/home/PKing/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-maker/modules/tts/data/轮-test.wav"
    target_sr = 48000
    audio_seg = AudioProcess(aggressiveness=3, frame_duration_ms=30, padding_duration_ms=30)
    segments = audio_seg.get_segments(audio_file)
    for i, segment in enumerate(segments):
        seg_data = segment[0]
        time_beg = segment[1]
        time_end = segment[2]
        print("({},{})".format(time_beg, time_end))
        idx_beg = int(time_beg * target_sr)
        idx_end = int(time_end * target_sr)
        audio_utils.sound_audio(seg_data)


if __name__ == '__main__':
    example1()
    # example2()
