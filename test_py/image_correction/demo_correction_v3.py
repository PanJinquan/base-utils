# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
from page_dewarp.image import WarpedImage
# from page_dewarp.options import cfg
# from page_dewarp.pdf import save_pdf
from pybaseutils.cvutils import mouse_utils, corner_utils
from pybaseutils import file_utils, image_utils


def main(imgfile):
    outfiles = []
    # print(imgfile)
    src = cv2.imread(imgfile)
    # print(src.shape[0],src.shape[1])
    src = src[0:src.shape[0] - 40, 0:src.shape[1]]
    cv_img = cv2.copyMakeBorder(src, 120, 0, 120, 120, cv2.BORDER_REPLICATE)
    cv2.imwrite(imgfile, cv_img)

    processed_img = WarpedImage(imgfile)
    # cv2.imwrite("result.png", processed_img)
    # if processed_img.written:
    #     outfiles.append(processed_img.outfile)
    #     print(f"  wrote {processed_img.outfile}", end="\n\n")
    # if cfg.pdf_opts.CONVERT_TO_PDF:
    #     save_pdf(outfiles)


def image_correction_demo(image_dir):
    """
    :param image_dir:
    :return:
    """
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        image_file = "/home/dm/桌面/image/6283f82b05014751878ae1cda4eb5486 (1).png"
        # print(image_file)
        # image = cv2.imread(image_file)
        main(image_file)
        print("倾斜角度：{}".format(""))
        print("--" * 10)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/page-correct/image3"  # 测试图片
    image_correction_demo(image_dir)

