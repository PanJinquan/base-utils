# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
from pybaseutils import file_utils, image_utils, pandas_utils
from pybaseutils.dataloader import parser_voc


def convert_to_class(src_file, bboxes, labels, index, class_name, out_dir="", flag="image2019"):
    labels = labels.reshape(- 1)
    names = [class_name[int(l)] for l in labels]
    if 'closed_eye' in names:
        sub = "drowsy"
    else:
        sub = "undrowsy"
    dst_file = file_utils.create_dir(out_dir, sub, "{}_{:0=6d}.jpg".format(flag, index))
    file_utils.copy_file(src_file, dst_file)
    return sub


if __name__ == "__main__":

    filename = "/home/PKing/nasdata/dataset/tmp/drowsy-driving/drowsy-driving/DDDataset-voc2/src/fdd-dataset/file_list.txt"
    out_dir = "/home/PKing/nasdata/dataset/tmp/drowsy-driving/drowsy-driving/DDDataset-voc2/src/fdd-dataset/class"
    use_rgb = False
    class_name = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=use_rgb,
                                    check=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    for i in range(len(dataset)):
        print(i)
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_ids"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        flag = int(image_id.split(".")[0])
        name = convert_to_class(data["image_file"], bboxes, labels, flag, class_name, out_dir=out_dir)
        print(image_id, name)
        # image = parser_voc.show_target_image(image, bboxes, labels, normal=False, transpose=False,
        #                                      class_name=class_name, use_rgb=use_rgb, thickness=2, fontScale=1)
        # image = show_boxes_image(image, Dataset.cxcywh2xyxy(bboxes, 0, 0), labels, normal=False, transpose=True)
