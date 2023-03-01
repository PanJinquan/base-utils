# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
from pybaseutils.dataloader import parser_voc

if __name__ == "__main__":
    # from models.transforms import data_transforms
    # filename = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/train.txt"
    # class_name = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/class_name.txt"
    # filename = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/word-v1/train.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/face_person/MPII/test.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/face-eyeglasses/trainval.txt"
    filename = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-v2/face-eyeglasses/train.txt"
    # filename = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/word-old/train.txt"
    # class_name = ["face", "person"]
    class_name =["unique"]
    # class_name =None
    use_rgb = False
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=use_rgb,
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
        parser_voc.show_target_image(image, bboxes, labels, normal=False, transpose=False,
                                     class_name=class_name, use_rgb=use_rgb, thickness=2)
        # show_boxes_image(image, Dataset.cxcywh2xyxy(bboxes, 0, 0), labels, normal=False, transpose=True)
