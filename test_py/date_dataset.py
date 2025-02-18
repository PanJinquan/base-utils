import cv2
import numpy as np
import time
import random
from datetime import datetime
from pybaseutils import file_utils, image_utils, time_utils
from PIL import Image, ImageDraw, ImageFont

# FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-date/fonts/dejavu"
# FONT = "/media/PKing/新加卷/SDK/base-utils/pybaseutils/font_style"
format = '%Y-%m-%d %H:%M:%S'


class DateDataset(object):
    def __init__(self,
                 numbers="0123456789",
                 fg=(0, 0, 0),
                 bg=(128, 128, 128),
                 size=51,
                 format='%Y-%m-%d %H:%M:%S'):
        """
        :param date:
        :param fg:
        :param bg:
        :param size:
        :param format:
        """
        self.size = size
        self.fg = fg
        self.bg = bg
        self.format = format
        self.numbers = numbers
        self.fonts = file_utils.get_files_lists(FONT, postfix=["*.ttf", "*.ttc"])
        print(self.fonts)

    @staticmethod
    def random_chars(char, n):
        """
        从字符串中随机取n个字符（允许重复）
        :param char: 源字符串
        :param n: 需要取的字符数量
        :return: 随机获取的字符串
        """
        return ''.join(random.choices(char, k=n))

    def get_random_date(self, ):
        char = self.random_chars(char=self.numbers, n=14)
        Y, m, d, H, M, S = char[0:4], char[4:6], char[6:8], char[8:10], char[10:12], char[12:14]
        date = f"{Y:4s}-{m:2s}-{d:2s} {H:2s}:{M:2s}:{S:2s}"
        return date

    def __getitem__(self, index):
        date = self.get_random_date()
        font_file = random.choice(self.fonts)
        image = self.add_text_image(point=(0, 0), text=date, font_file=font_file, src=None)
        return dict(image=image, label=date)

    def add_text_image(self, point=(0, 0), text="2025-02-12 16:53:16", font_file=FONT, bold=False, src=None):
        """
        """
        image = src if isinstance(src, np.ndarray) else image_utils.create_image(
            shape=(point[1] + 128, point[0] + 640, 3))
        pilimg = Image.fromarray(image)
        # 创建可以在图片上绘画的对象
        draw = ImageDraw.Draw(pilimg)
        print(font_file)
        # font = ImageFont.truetype(font_file, self.size)  # DejaVu Sans Mono 字体
        font = ImageFont.truetype(font_file, size=self.size, encoding="utf-8")
        # 计算文本大小
        fg_box = draw.textbbox((0, 0), text, font=font)
        fg_w, fg_h = fg_box[2] - fg_box[0], fg_box[3] - fg_box[1]
        # 计算文本位置（放在图片左上角，留出一定边距）
        x, y = point[0], point[1]
        # 先绘制底纹
        # 为文本区域绘制灰色背景
        bg_box = [x - 5, y, x + fg_w + 5, y + fg_h + 15]
        draw.rectangle(bg_box, fill=self.bg)
        # TODO 通过多次绘制文本来实现加粗效果
        if bold:
            for offset in range(-1, 2):
                for offset2 in range(-1, 2):
                    draw.text((x + offset, y + offset2), text, font=font, fill=self.fg)
        # 绘制文本主体
        draw.text((x, y), text, font=font, fill=self.fg)
        image = np.array(pilimg)
        if not isinstance(src, np.ndarray): image = image_utils.get_box_crop(image, bg_box)
        return image

    def get_test_image(self, image, date, point=(0, 100)):
        font_file = random.choice(self.fonts)
        image = self.add_text_image(point=point, text=date, font_file=font_file, src=image)
        return dict(image=image, label=date)


def example1():
    # 示例使用
    image_file = "/home/PKing/Downloads/20250213-085844.png"  # 替换为你的图片路径
    dataset = DateDataset()
    for i in range(100):
        image = cv2.imread(image_file)
        data_info = dataset.__getitem__(i)
        label = data_info["label"]
        print(label)
        data_info = dataset.get_test_image(image, date=label)
        label = data_info["label"]
        image = data_info["image"]
        image_utils.cv_show_image("image", image, delay=0)


def example2():
    dataset = DateDataset()
    for i in range(100):
        data_info = dataset.__getitem__(i)
        label = data_info["label"]
        image = data_info["image"]
        print(label)
        image_utils.cv_show_image("image", image, delay=0)


def example3():
    output = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-date/dataset-v2"
    # output = "/home/PKing/Downloads/dataset-v1"
    dataset = DateDataset()
    for i in range(1000):
        data_info = dataset.__getitem__(i)
        label = data_info["label"]
        image = data_info["image"]
        print(label)
        image_file = file_utils.create_dir(output, "images", f"{label}_{i:0=6d}.jpg")
        cv2.imwrite(image_file, image)
        image_utils.cv_show_image("image", image, delay=10)


if __name__ == "__main__":
    example1()
    # example3()
    # example2()
