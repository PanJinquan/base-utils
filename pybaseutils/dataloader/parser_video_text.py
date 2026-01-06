# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import PIL.Image as Image
import numpy as np
import random
import cv2
import torch
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import video_utils
from classifier.dataset import parser_image_text


class VideoTextDataset(parser_image_text.TextDataset):
    """Pytorch Dataset"""

    def __init__(self, data_file, data_root=None, class_name=None, transform=None, shuffle=False, use_rgb=False,
                 phase="test", disp=False, check=False, **kwargs):
        super(VideoTextDataset, self).__init__(data_file=data_file, data_root=data_root, class_name=class_name,
                                               transform=transform, shuffle=shuffle, use_rgb=use_rgb,
                                               phase=phase, disp=disp, check=check, **kwargs)
        self.tag = self.__class__.__name__
        self.cfg: dict = kwargs.get("cfg", {})
        self.duration = self.cfg.get("duration", 3)
        self.freq = self.cfg.get("freq", 4)
        self.seq_len = int(self.duration * self.freq)  # 32=4*8,64=4*16
        self.input_size = self.cfg.get("input_size", kwargs.get("input_size", (112, 112)))
        self.log("{:15s} duration        :{}".format(self.tag, self.duration))
        self.log("{:15s} freq            :{}".format(self.tag, self.freq))
        self.log("{:15s} seq_len         :{}".format(self.tag, self.seq_len))
        self.log("{:15s} input_size      :{}".format(self.tag, self.input_size))
        self.log("------------------------------------------------------------------")

    def load_dataset(self, data_file, data_root="", **kwargs):
        """
        txt保存格式：[path,name] 或者 [path,name,xmin,ymin,xmax,ymax]
        :param data_file:
        :param data_root:
        :return: item_list [{"file","label","name","bbox"}],bbox非必须
        """
        if isinstance(data_file, str): data_file = [data_file]
        item_list = []
        for file in data_file:
            root = data_root if data_root else os.path.dirname(file)
            content = file_utils.read_data(file, split=" ")
            data = []
            for line in content:
                item = {"file": os.path.join(root, line[0]), "label": line[1], 'name': line[1]}
                if len(line) == 6: item['bbox'] = line[2:]  # (xmin,ymin,xmax,ymax)
                data.append(item)
            self.log("{:15s} loading data from:{},have {}".format(self.tag, file, len(data)))
            item_list += data
        return item_list

    def __getitem__(self, index):
        """
        :param index:
        :return: {"image": image, "label": label}
        """
        item = self.item_list[index]
        file, label, bbox = item["file"], item[self.label_index], item.get("bbox", [])
        image = self.read_video(file, size=self.input_size, use_rgb=self.use_rgb, duration=self.duration,
                                freq=self.freq, shuffle=self.shuffle)
        label = np.asarray(label, dtype=np.int64)
        if len(image) > 0:
            if self.transform: image = self.transform(images=image)
            image = self.normalize(image)
            image = self.to_tensor(image)  # (C,D,H,W)
        if image is None or len(image) == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        return {"image": image, "label": label, "file": file}

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame = (frame / 255.0 - 0.5) / 0.5
            buffer[i] = frame
        buffer = np.asarray(buffer, dtype=np.float32)
        return buffer

    def read_video(self, video_file, size=(224, 224), use_rgb=False, shuffle=False, **kwargs):
        """
        :param video_file:
        :param size:
        :param use_rgb:
        :param freq: 抽帧频率
        :param use_cut:
        :return:
        """
        video_cap = video_utils.get_video_capture(video_file)
        width, height, numFrames, fps = video_utils.get_video_info(video_cap, disp=False)
        time = (0, numFrames / fps)
        video_time, video_idx = video_utils.get_video_sampling(self.freq, time, fps, random=shuffle)
        # TODO 居中裁剪
        clip = (int(len(video_idx) / 2 - self.seq_len / 2), int(len(video_idx) / 2 + self.seq_len / 2))
        clip = (max(clip[0], 0), min(clip[1], len(video_idx)))
        video_idx = video_idx[clip[0]:clip[1]]
        frames = []
        for count in video_idx:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            ret, frame = video_cap.read()
            if not ret: break
            frame = image_utils.resize_image_padding(frame, size=size)
            if use_rgb: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) < 3: return []  # 帧太少，无效视频
        if len(frames) < self.seq_len:
            pad = self.seq_len - len(frames)
            images = [frames[-1]] * pad
            frames += images
        frames = frames[0:self.seq_len]
        assert len(frames) == self.seq_len, "frames size error:{}".format(len(frames))
        return frames


if __name__ == '__main__':
    from classifier.transforms import build_transform
    from classifier.dataset import build_dataset
    from pybaseutils import image_utils

    data_file = ["/home/PKing/nasdata/tmp/UCF101/UCF101/val.txt"]
    batch_size = 1
    input_size = [224, 224]
    # trans_type = "train_video"
    trans_type = "test_video"
    transform = build_transform.image_transform(input_size=input_size, trans_type=trans_type)
    cfg = {"duration": 5, "freq": 2}
    class_name = ["ApplyEyeMakeup", "ApplyLipstick", "Archery"]
    dataset = VideoTextDataset(data_file=data_file,
                               transform=transform,
                               resample=False,
                               shuffle=False,
                               class_name=class_name,
                               input_size=input_size,
                               use_rgb=False,
                               cfg=cfg,
                               disp=True)
    for i in range(len(dataset)):
        data = dataset.__getitem__(0)
        video_file, image, label = data["file"], data["image"], data["label"]
        image = image.transpose((1, 2, 3, 0))
        image = np.asarray((image * 0.5 + 0.5) * 255, dtype=np.uint8)
        images = [image[i] for i in range(len(image))]
        images = image_utils.image_vstack(images=images)
        print(image.shape, label, video_file)
        image_utils.cv_show_image("image", images)
