# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
from pybaseutils.dataloader import parser_voc

if __name__ == "__main__":
    """
    """
    # from models.transforms import data_transforms
    # filename = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/train.txt"
    # class_name = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/class_name.txt"
    filename = "/home/dm/nasdata/dataset/csdn/plate/dataset/CCPD2019-voc/ccpd_base/val.txt"
    class_name=["plate"]
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    check=False,
                                    shuffle=True)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    for i in range(len(dataset)):
        print(i)
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        print(image_id)
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        parser_voc.show_target_image(image, bboxes, labels, normal=False, transpose=False, class_name=class_name,
                                     thickness=4)
        # show_boxes_image(image, Dataset.cxcywh2xyxy(bboxes, 0, 0), labels, normal=False, transpose=True)
