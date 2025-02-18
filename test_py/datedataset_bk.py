import cv2
import numpy as np
import time
from datetime import datetime
from pybaseutils import file_utils, image_utils, time_utils
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
format = '%Y-%m-%d %H:%M:%S'


class DateDataset(object):
    def __init__(self, date=("1970-01-01 00:00:00", "9999-12-31 00:00:00"),
                 fg=(0, 0, 0),
                 bg=(128, 128, 128),
                 size=52,
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
        self.date = date
        self.datestamp = (time_utils.date2stamp(self.date[0], self.format),
                          time_utils.date2stamp(self.date[1], self.format)
                          )
        print(self.date)
        print(self.datestamp)

    def get_random_time(self):
        """随机时间"""
        return np.random.randint(int(self.datestamp[0]), int(self.datestamp[1]))

    def __getitem__(self, index):
        stamp = self.get_random_time()
        date = time_utils.stamp2date(stamp, format=self.format)
        image = self.add_text_image(point=(0, 0), text=date, font_file=FONT, src=None)
        return dict(image=image, label=date)

    def add_text_image(self, point=(0, 0), text="2025-02-12 16:53:16", font_file=FONT, src=None):
        """
        """
        image = src if isinstance(src, np.ndarray) else image_utils.create_image(
            shape=(point[1] + 128, point[0] + 640, 3))
        pilimg = Image.fromarray(image)
        # 创建可以在图片上绘画的对象
        draw = ImageDraw.Draw(pilimg)
        font = ImageFont.truetype(font_file, self.size)  # DejaVu Sans Mono 字体
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
        # 先绘制文本轮廓
        for offset in range(-1, 2):
            for offset2 in range(-1, 2):
                draw.text((x + offset, y + offset2), text, font=font, fill=self.fg)
        # 绘制文本主体
        draw.text((x, y), text, font=font, fill=self.fg)
        image = np.array(pilimg)
        if not isinstance(src, np.ndarray): image = image_utils.get_box_crop(image, bg_box)
        return image


def example1():
    # 示例使用
    image_file = "/home/PKing/Downloads/20250213-085844.png"  # 替换为你的图片路径
    image = cv2.imread(image_file)
    stamp = time.time()
    dataset = DateDataset()
    for i in range(100):
        stamp = stamp + 1
        print(f'stamp={stamp}')
        date_str = time_utils.stamp2date(stamp, format=format)
        image = dataset.add_text_image(text=date_str, src=image, point=(0, 100))
        image_utils.cv_show_image("date_img", image, delay=0)


def example2():
    dataset = DateDataset()
    for i in range(100):
        data_info = dataset.__getitem__(i)
        label = data_info["label"]
        image = data_info["image"]
        print(label)
        image_utils.cv_show_image("image", image, delay=0)


if __name__ == "__main__":
    # example1()
    example2()
