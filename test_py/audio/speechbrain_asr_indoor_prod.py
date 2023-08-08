import os
import sys
import ffmpeg
import torch
import queue
import webrtcvad
import pyaudio
import time
import halo
import pinyin
import random
import subprocess
import numpy as np
import speechbrain as sb
from copy import deepcopy
from hyperpyyaml import load_hyperpyyaml
from pprint import pprint
from datetime import datetime
from multiprocessing import Process, Value
from threading import Thread
from collections import deque
from speechbrain.pretrained import EncoderDecoderASR
from rapidfuzz import process, fuzz

import pika
import json
import redis
import traceback
from logger import custom_log

log = custom_log(logging_file_dir='logs/asr_indoor.log')


class PikaMessenger:
    """
    Pika w/ multithreading

    Args:
        cfg (dict): a dictionary containing all configurations

    Attributes:
        cfg (dict): a dictionary containing all configurations
        user_pwd: pika credentials containing username and password
        con: pika connection
        ch: pika connection channel
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.user_pwd = pika.PlainCredentials(self.cfg['username'],
                                              self.cfg['pwd'])
        self.con = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.cfg['ip'],
                                      port=self.cfg['port'],
                                      credentials=self.user_pwd,
                                      heartbeat=0))
        self.ch = self.con.channel()

    def consume(self, queue_name, callback):
        """Declare a RabbitMQ queue and start consuming in a child thread

        Args:
            queue_name (str): name of the queue
            callback (function): a callback function to grab info from inbound messages
        """
        try:
            q = self.ch.queue_declare(queue=queue_name,
                                      durable=False,
                                      auto_delete=True)
            self.ch.basic_consume(queue=queue_name,
                                  on_message_callback=callback,
                                  auto_ack=True,
                                  exclusive=False,
                                  consumer_tag=None,
                                  arguments=None)
            log.debug(f"{queue_name}: start_consuming()")
            self.ch.start_consuming()
        except SystemExit:
            log.debug(f"{queue_name}: stop_consuming()")
            self.ch.stop_consuming()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.con.close()


class ASRIndoorService:
    """
    The ASR service for indoor scene

    Args:
        cfg (dict): a dictionary contains all configurations

    Attributes:
        cfg (dict): a dictionary contains all configurations
        asr_model: an asr model object
        vad: a voice activity detection object used to check whether a chunk is speech or not
        spinner: a spinner object that spins when there is a ongoing speech
        pa: a PyAudio object
        stream: a stream object in PyAudio
        stream_callback: a callback function which fetches audio chunk in real time
        audio_queue: a queue that stores audio chunk
        audio_pipe: a buffer that stores audio data fetched by ffmpeg
        examId: a exam ID that will be fetched from inbound message
        examineeId: a examinee ID that will be fetched from inbound message
        url: rtsp/rtmp url that will be fetched from inbound message
        state: a signal - 0: start the exam; 1: end the exam; -1: restart the script
        previous_state: a varible stores previous state
        stage: a stage indicator that will be fetched from inbound message, initial value is 'S1'
        rtk_out: a routing key for exchange that sends out messages
        ex_out: a exchange name that sends out messages
        user_pwd: pika credentials containing username and password
        con: pika connection
        ch: pika connection channel
        in_message_consumer_thread: a child thread that fetches messages from 'voice_tipmessage'
        in_stage_consumer_thread: a child thread that fetches messages from 'voice_tipstage'
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.add_pys_to_cfg()
        self.asr_model = self.load_model()  # 语音识别模型
        self.vad = webrtcvad.Vad(self.cfg['vad_mode'])
        if self.cfg['activate_spinner']:
            self.spinner = halo.Halo(spinner='noise')
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.stream_callback = None
        self.audio_queue = queue.Queue()
        self.audio_pipe = None

        # Init variables to be fetched from Queue
        self.examId = 0
        self.examineeId = 0
        self.url = ''
        self.state = ''
        self.previous_state = ''
        self.stage = self.cfg['stage']

        # RabbitMQ stuffs
        self.rtk_out = self.cfg['out_contentmessage']
        self.ex_out = self.cfg['out_contentmessage']
        user_pwd = pika.PlainCredentials(self.cfg['username'], self.cfg['pwd'])
        self.con = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.cfg['ip'],
                                      port=self.cfg['port'],
                                      credentials=user_pwd,
                                      heartbeat=0))
        self.ch = self.con.channel()
        self.in_message_consumer_thread = None
        self.in_stage_consumer_thread = None

    def run(self):
        """Run ASR service

        Raises:
            SystemExit: Restart the script
        """
        # Running two queue_in consumers in two separate threads
        self.in_message_consumer_thread = Thread(
            target=self.start_in_message_consumer)
        self.in_message_consumer_thread.daemon = True
        self.in_stage_consumer_thread = Thread(
            target=self.start_in_stage_consumer)
        self.in_stage_consumer_thread.daemon = True

        log.debug("Starting in_message_consumer_thread")
        self.in_message_consumer_thread.start()
        log.debug("Starting in_stage_consumer_thread")
        self.in_stage_consumer_thread.start()

        log.info(' [*] Waiting for messages. To exit press CTRL+C')

        # Block the rest of the process until receiving the first non-empty url
        # and check the url type
        while True:
            if self.url:
                log.debug(f"Got url: {self.url}")
                if self.url.startswith('rtmp'):
                    self.cfg['input_choice'] = 'rtmp'
                    self.cfg['rtmp_url'] = self.url
                elif self.url.startswith('rtsp'):
                    self.cfg['input_choice'] = 'rtsp'
                    self.cfg['rtsp_url'] = self.url
                self.previous_state = self.state
                break

        # Block the rest of the process again until receiving the first non-empty stage
        while True:
            if self.stage:
                log.debug(f"Got stage: {self.stage}")
                break

        # Build audio_pipe to fetch audio data using ffmpeg
        self.audio_pipe = self.build_audio_pipe()

        # Build PyAudio stream to fetch audio chunk from audio_pipe
        self.stream_callback = self.setup_stream_callback()
        self.stream = self.setup_stream()
        self.stream.start_stream()
        log.info(
            f"=== Listening to {self.cfg['input_choice']} (ctrl-C to exit)... ==="
        )

        # Instantiate vad and chunk generator
        speech_chunks = self.vad_collector()
        speech_segment = bytearray()
        just_ended = True
        exam_ended = False
        start_time, end_time = None, None
        current_stage = self.stage  # use dynamic stage

        # For each chunk generated by vad and chunk generator
        for chunk in speech_chunks:
            # Hard restart code when self.state == '-1'
            if self.state == '-1':
                raise SystemExit
            # Bypass asr service when self.state == '1'
            if self.state == '1' and not exam_ended:
                log.info("=== Dump unfinished speech ===")
                speech_segment = bytearray()
                just_ended = True
                exam_ended = True
                self.stage = None
                continue
            # Resume asr service when self.state == '0'
            if self.state == '0' and exam_ended:
                log.debug("Set exam_ended to False and wait for stage")
                exam_ended = False
                while True:
                    if self.stage:
                        log.debug(f"Got stage: {self.stage}")
                        break

            # At the starting chunk of the speech_segment
            if just_ended:
                # Update current stage
                current_stage = self.stage  # use dynamic stage
                start_time = datetime.now()
                just_ended = False
            # If chunk exists
            if chunk is not None:
                if self.cfg['activate_spinner']:
                    self.spinner.start()
                speech_segment.extend(chunk)
            # At the end of the speech_segment
            else:
                just_ended = True
                if self.cfg['activate_spinner']:
                    self.spinner.stop()
                # Added length constrain to speech_segment
                if len(speech_segment) >= self.cfg['speech_max_length']:
                    log.debug("!" * 100)
                    log.debug(
                        f"speech_segment ({len(speech_segment)} bytes) exceeds speech_max_length ({self.cfg['speech_max_length']})"
                    )
                    log.debug("Skip recognizing")
                    log.debug("!" * 100)
                    speech_segment = bytearray()
                    continue
                # Convert bytesarray to tensor
                waves = self.audio_preprocess(speech_segment)
                # ASR model inference
                pred = self.audio_inference(waves)
                # Filter out very short predictions
                if len(pred) == 0 or len(pred) == 1:
                    continue

                # Run RapidFuzz matching algorithm
                matched_targets, matched_algo_signs, matched_keywords, keywords, similarities = self.fuzz_match(
                    pred, current_stage)
                end_time = datetime.now()

                # Display results
                log.debug("-" * 100)
                log.debug(f"Current stage: {current_stage}")
                log.debug(f"Raw ASR Model Output: {pred}")
                log.debug("Keywords Detected by RapidFuzz:")
                for keyword, similarity in zip(keywords, similarities):
                    log.debug(f"\t- {keyword} ({similarity:.2f}%)")
                log.debug(f"Matched Keywords: {', '.join(matched_keywords)}")
                log.debug(f"Matched Target: {' | '.join(matched_targets)}")
                log.debug(
                    f"Matched Algorithm Sign: {' | '.join(matched_algo_signs)}")
                log.debug(
                    f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                log.debug(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                log.debug("-" * 100)

                # Clear speech_segment
                speech_segment = bytearray()

                # Publish response json
                response_dict = {
                    'examId': self.examId,
                    'examineeId': self.examineeId,
                    'stage': self.stage,
                    'matchedKeywords': ",".join(matched_keywords),
                    'algorithmSign': ",".join(matched_algo_signs),
                    'isCheckout': True if matched_targets else False,
                    'audioResult': pred,
                    'startTime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'endTime': end_time.strftime('%Y-%m-%d %H:%M:%S')
                }

                response_json = json.dumps(response_dict)

                self.ch.basic_publish(
                    exchange=self.ex_out,
                    routing_key=self.rtk_out,
                    body=response_json,
                    properties=pika.BasicProperties(delivery_mode=1))
                log.info('返回成功： {}'.format(matched_algo_signs))
                log.info(response_dict)

    def start_in_message_consumer(self):
        """Runs callback function that fetches messages from 'voice_tipmessage'"""

        def callback_in_message(ch, method, properties, body):
            log.debug('inside callback_in_message')

            # 需要返回的参数
            obj_json = self.getJsonObj(body=body)

            # 获取消息字段
            self.examId = int(obj_json['examId'])
            self.examineeId = int(obj_json['examineeId'])
            self.url = str(obj_json['address'])  # renamed to 'address'
            log.debug(f'url in callback_in_message: {self.url}')
            self.state = str(obj_json['state'])

        with PikaMessenger(self.cfg) as consumer:
            consumer.consume(queue_name=self.cfg['in_tipmessage'],
                             callback=callback_in_message)
        log.debug('exit start_in_message_consumer')

    def start_in_stage_consumer(self):
        """Runs callback function that fetches messages from 'voice_tipstage'"""

        def callback_in_stage(ch, method, properties, body):
            log.debug('inside callback_in_stage')

            # 需要返回的参数
            obj_json = self.getJsonObj(body=body)

            # 获取消息字段
            self.examId = int(obj_json['examId'])
            self.examineeId = int(obj_json['examineeId'])
            self.stage = str(obj_json['stage'])
            if self.stage not in ['S1', 'S2', 'S3']:
                log.warning(
                    f'stage: {self.stage} is not in valid format, redirect to None'
                )
                self.stage = None
            log.debug(f'stage in callback_in_stage: {self.stage}')
            self.url = str(obj_json['address'])  # duplicate url for resuming asr service after abnormal restart
            log.debug(f'url in callback_in_stage: {self.url}')

        with PikaMessenger(self.cfg) as consumer:
            consumer.consume(queue_name=self.cfg['in_tipstage'],
                             callback=callback_in_stage)
        log.debug('exit start_in_stage_consumer')

    def getJsonObj(self, body):
        """Get json string from binary body"""

        data_string = bytes.decode(body)
        # load to json obj
        log.debug(f'data_string: {data_string}')
        obj_json = json.loads(data_string)
        return obj_json

    def add_pys_to_cfg(self):
        """Assign pinyin to all keywords"""

        self.cfg['keyword_pys'] = deepcopy(self.cfg['keywords'])
        for stage in self.cfg['keyword_pys'].keys():
            for keyword in self.cfg['keyword_pys'][stage].keys():
                self.cfg['keyword_pys'][stage][keyword] = pinyin.get(
                    keyword, format='strip', delimiter=' ')
        log.info('Added keyword_pys to cfg')

    def load_model(self):
        """Load ASR model"""

        return EncoderDecoderASR.from_hparams(
            source=self.cfg['source'],
            savedir=self.cfg['savedir'],
            run_opts={'device': self.cfg['device']})

    def build_audio_pipe(self):
        """Use ffmpeg to fetch audio data into audio pipe"""

        if self.cfg['input_choice'] == 'video':
            # Run local audio pipe in child process
            log.debug("Constructing audio_pipe for video")
            if self.cfg['activate_rnnoise']:
                log.debug(f"Applying arnndn filter w/ rnnoise model: {self.cfg['rnnoise_model']}")
                audio_pipe = subprocess.Popen(
                    f"ffmpeg -loglevel {self.cfg['ffmpeg_loglevel']} -i {self.cfg['video_file']} -af arnndn=m={self.cfg['rnnoise_model']} -map 0:a -f s16le -ac {self.cfg['num_audio_channels']} -acodec pcm_s16le -ar {self.cfg['audio_sample_rate']} -loglevel {self.cfg['ffmpeg_loglevel']} pipe:",
                    shell=True,
                    stdout=subprocess.PIPE
                )
            else:
                audio_pipe = (
                    ffmpeg
                        .input(self.cfg['video_file'], loglevel=self.cfg['ffmpeg_loglevel'])['a']
                        .output('pipe:',
                                format='s16le',
                                acodec='pcm_s16le',
                                ac=self.cfg['num_audio_channels'],
                                ar=self.cfg['audio_sample_rate'],
                                loglevel=self.cfg['ffmpeg_loglevel'])
                        # .global_args('-report')
                        .run_async(pipe_stdout=True)
                )
        elif self.cfg['input_choice'] == 'rtsp':
            # Run RTSP audio pipe in child process
            log.debug("Constructing audio_pipe for rtsp")
            if self.cfg['activate_rnnoise']:
                log.debug(f"Applying arnndn filter w/ rnnoise model: {self.cfg['rnnoise_model']}")
                audio_pipe = subprocess.Popen(
                    f"ffmpeg -loglevel {self.cfg['ffmpeg_loglevel']} -allowed_media_types audio -rtsp_transport tcp -i {self.cfg['rtsp_url']} -af arnndn=m={self.cfg['rnnoise_model']} -f s16le -ac {self.cfg['num_audio_channels']} -acodec pcm_s16le -ar {self.cfg['audio_sample_rate']} -loglevel {self.cfg['ffmpeg_loglevel']} pipe:",
                    shell=True,
                    stdout=subprocess.PIPE
                )
            else:
                audio_pipe = (
                    ffmpeg
                        .input(self.cfg['rtsp_url'],
                               rtsp_transport='tcp',
                               allowed_media_types='audio',
                               loglevel=self.cfg['ffmpeg_loglevel'])
                        .output('pipe:',
                                format='s16le',
                                acodec='pcm_s16le',
                                ac=self.cfg['num_audio_channels'],
                                ar=self.cfg['audio_sample_rate'],
                                loglevel=self.cfg['ffmpeg_loglevel'])
                        # .global_args('-report')
                        .run_async(pipe_stdout=True)
                )
        elif self.cfg['input_choice'] == 'rtmp':
            # Run RTMP audio pipe in child process
            log.debug("Constructing audio_pipe for rtmp")
            if self.cfg['activate_rnnoise']:
                log.debug(f"Applying arnndn filter w/ rnnoise model: {self.cfg['rnnoise_model']}")
                audio_pipe = subprocess.Popen(
                    f"ffmpeg -loglevel {self.cfg['ffmpeg_loglevel']} -i {self.cfg['rtmp_url']} -map 0:a -f s16le -ac {self.cfg['num_audio_channels']} -acodec pcm_s16le -ar {self.cfg['audio_sample_rate']} -loglevel {self.cfg['ffmpeg_loglevel']} pipe:",
                    shell=True,
                    stdout=subprocess.PIPE
                )
            else:
                audio_pipe = (
                    ffmpeg
                        .input(self.cfg['rtmp_url'], loglevel=self.cfg['ffmpeg_loglevel'])['a']
                        .output('pipe:',
                                format='s16le',
                                acodec='pcm_s16le',
                                ac=self.cfg['num_audio_channels'],
                                ar=self.cfg['audio_sample_rate'],
                                loglevel=self.cfg['ffmpeg_loglevel'])
                        # .global_args('-report')
                        .run_async(pipe_stdout=True)
                )
        else:
            log.debug("Using mic")
            audio_pipe = None
        return audio_pipe

    def setup_stream_callback(self):
        """Define callback for PyAudio stream instance"""

        if self.audio_pipe:

            def callback(in_data, frame_count, time_info, status):
                data = self.audio_pipe.stdout.read(2 * frame_count)
                if not data:
                    self.state = '-1'
                self.audio_queue.put(data)
                return (data, pyaudio.paContinue)
        else:

            def callback(in_data, frame_count, time_info, status):
                self.audio_queue.put(in_data)
                return (None, pyaudio.paContinue)

        return callback

    def setup_stream(self):
        """Define PyAudio stream instance"""

        if self.audio_pipe:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.cfg['num_audio_channels'],
                rate=self.cfg['audio_sample_rate'],
                output=True,
                frames_per_buffer=self.cfg['chunk_size'],
                stream_callback=self.stream_callback,
                output_device_index=self.cfg['output_device_index'])
        else:
            stream = self.pa.open(format=pyaudio.paInt16,
                                  channels=self.cfg['num_audio_channels'],
                                  rate=self.cfg['audio_sample_rate'],
                                  input=True,
                                  frames_per_buffer=self.cfg['chunk_size'],
                                  stream_callback=self.stream_callback)
        return stream

    def vad_collector(self):
        """
        Pull audio chunks from audio_queue, tag them w/ VAD and yield continuous speech chunks

        Raises:
            SystemExit: restart the script if audio queue is empty for 20 secs

        Yields:
            byte: speech chunk
        """
        chunks = self.chunk_generator()
        num_padding_frames = self.cfg['vad_pad_duration_ms'] // self.cfg[
            'chunk_duration_ms']
        ring_buffer = deque(maxlen=num_padding_frames)
        triggered = False

        for chunk in chunks:
            # Ignore chunks with smaller chunk size
            if len(chunk) < self.cfg['chunk_size']:
                continue

            # Restart the script if self.state == '-1'
            if self.state == '-1':
                log.debug(
                    "get self.state == '-1' in vad_collector(), raise SystemExit"
                )
                raise SystemExit

            # Ignore chunks when self.state == '1'
            if self.state == '1':
                if self.previous_state != '1':
                    log.info(
                        "=== Exam has ended, waiting for the next exam start... ==="
                    )
                    self.previous_state = self.state
                    ring_buffer.clear()
                    triggered = False
                continue

            # Resume consuming chunks when self.state == '0'
            if self.state == '0':
                if self.previous_state != '0':
                    log.info(
                        f"=== Exam has started, examId: {self.examId} examineeId: {self.examineeId} ==="
                    )
                    self.previous_state = self.state
                    log.debug("Set stage back to 'S1'")
                    self.stage = 'S1'

            # Check whether each audio chunk is speech or not
            try:
                is_speech = self.vad.is_speech(chunk,
                                               self.cfg['audio_sample_rate'])
            except:
                log.debug(
                    "Reach the end of stream (Error in vad.is_speech()), raise SystemExit"
                )
                raise SystemExit

            # Yield continuous speech chunks by ring_buffer
            if not triggered:
                ring_buffer.append((chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > self.cfg[
                    'vad_trigger_ratio'] * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield chunk
                ring_buffer.append((chunk, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > self.cfg[
                    'vad_trigger_ratio'] * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

    def chunk_generator(self):
        """Get audio chunks from audio_queue"""

        while True:
            if self.state == '-1':
                log.debug("raise SystemExit in chunk_generator")
                raise SystemExit
            try:
                yield self.audio_queue.get(
                    timeout=self.cfg['audio_queue_timeout'])
            except queue.Empty:
                log.debug("audio_queue.get() timeout, raise SystemExit")
                raise SystemExit

    @staticmethod
    def audio_preprocess(segment_bytes):
        """Convert speech segment from bytearray to torch tensor

        Args:
            segment_bytes (bytearray): a bytearray of continuous speech chunks

        Returns:
            torch.Tensor: tensor representation of speech segment
        """
        wave_np = np.frombuffer(segment_bytes, np.int16)
        wave_np = wave_np.astype(np.float32, order='C') / 32768.0
        wave_batch = torch.from_numpy(wave_np).unsqueeze(0)
        return wave_batch

    def audio_inference(self, waves):
        """Run inference w/ ASR model

        Args:
            waves (torch.Tensor): tensor representation of speech segment

        Returns:
            str: ASR result
        """
        wave_lens = torch.tensor([1.0])
        preds, pred_tokens = self.asr_model.transcribe_batch(waves, wave_lens)  # 声波的转录文件，每个转录的token令牌
        pred = preds[0].replace(' ', '')
        return pred

    def fuzz_match(self, pred, stage):  # 对ASR结果执行RapidFuzz算法，得到关键词然后用它们来搜索目标句子

        """Perform RapidFuzz algorithm on ASR result to get the keywords
        then use them to search for the target sentence

        Args:
            pred (str): ASR result
            stage (str): current stage ID

        Returns:
            tuple: matching results
        """
        # Get predefined keywords matching library for the current stage
        stage_keywords = self.cfg['keywords'][stage]
        # Convert asr result to pinyin
        pred_py = pinyin.get(pred, format='strip', delimiter=' ')

        # Use RapidFuzz algorithm to extract keywords from ASR result
        candidates = process.extract(pred_py,
                                     self.cfg['keyword_pys'][stage],
                                     scorer=eval(self.cfg['scorer']),
                                     limit=self.cfg['limit'],
                                     score_cutoff=self.cfg['score_cutoff'])
        keywords = [candidate[2] for candidate in candidates]
        similarities = [candidate[1] for candidate in candidates]

        # Query keywords in predefined keywords matching library to find the target sentence
        matched_targets, matched_algo_signs, matched_keywords = [], [], []
        for keyword in keywords:
            if keyword not in stage_keywords:
                continue
            linked_obj = stage_keywords[keyword]
            if isinstance(linked_obj, str):
                matched_keywords.append(keyword)
                matched_targets.append(self.cfg['targets'][linked_obj])
                matched_algo_signs.append(linked_obj)
            elif isinstance(linked_obj, dict):
                for sub_keyword in keywords:
                    if sub_keyword == keyword:
                        continue
                    if sub_keyword in linked_obj:
                        matched_keywords += [keyword, sub_keyword]
                        if isinstance(linked_obj[sub_keyword], str):
                            matched_targets.append(
                                self.cfg['targets'][linked_obj[sub_keyword]])
                            matched_algo_signs.append(linked_obj[sub_keyword])
                            break
                        elif isinstance(linked_obj[sub_keyword], list):
                            matched_targets += [
                                self.cfg['targets'][sign]
                                for sign in linked_obj[sub_keyword]
                            ]
                            matched_algo_signs += linked_obj[sub_keyword]
                            break
            elif isinstance(linked_obj, list):
                matched_keywords.append(keyword)
                matched_targets += [
                    self.cfg['targets'][sign] for sign in linked_obj
                ]
                matched_algo_signs += linked_obj

            if matched_targets and matched_algo_signs and matched_keywords:
                break

        return matched_targets, matched_algo_signs, matched_keywords, keywords, similarities


def main():
    # Load config file with command-line overrides
    cfg_file, _, overrides = sb.parse_arguments(sys.argv[1:])
    with open(cfg_file) as fin:
        cfg = load_hyperpyyaml(fin, overrides)
    log.info(cfg)
    try:
        log.debug("init asr_service")
        # Instantiate asr_indoor_service
        asr_indoor_service = ASRIndoorService(cfg)
        # Run asr_indoor_service
        asr_indoor_service.run()
        log.debug("out of asr_service.run()")
    except KeyboardInterrupt:
        log.debug("Interrupted")
        log.debug("sys.exit()")
        sys.exit()
    except SystemExit:
        log.debug("inside SystemExit exception")
        log.debug("restart the script")
        os.execv(sys.executable, ['python'] + sys.argv)
    except RuntimeError as re:
        log.debug(f"RuntimeError encountered: {re}")
        log.debug("restart the script")
        os.execv(sys.executable, ['python'] + sys.argv)
    except Exception as e:
        log.error(traceback.format_exc())
        print(traceback.format_exc())
        log.debug("restart the script")
        os.execv(sys.executable, ['python'] + sys.argv)


if __name__ == '__main__':
    main()
