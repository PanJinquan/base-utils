# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import xmltodict
import numpy as np
import cv2
import glob
import random
import numbers
from tqdm import tqdm
from pybaseutils.dataloader.dataset import Dataset


class VOCDataset(Dataset):
    """
    VOC数据格式解析器
    数据格式：(xmin,ymin,xmax,ymax)
    输出格式：target is (xmin,ymin,xmax,ymax,label)
    """

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_name=None,
                 transform=None,
                 use_rgb=True,
                 shuffle=False,
                 check=False):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param use_rgb:
        :param shuffle:
        """
        super(VOCDataset, self).__init__()
        self.unique = False  # 是否是单一label，如["BACKGROUND", "unique"]
        self.class_name, self.class_dict = self.parser_classes(class_name)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_id = parser
        self.postfix = self.get_image_postfix(self.image_dir, self.image_id)
        self.transform = transform
        self.use_rgb = use_rgb
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        if check:
            self.image_id = self.checking(self.image_id)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)
        self.num_images = len(self.image_id)
        # print("VOCDataset class_count:{}".format(class_count))
        print("VOCDataset class_name :{}".format(class_name))
        print("VOCDataset class_dict :{}".format(self.class_dict))
        print("VOCDataset num images :{}".format(len(self.image_id)))
        print("VOCDataset num_classes:{}".format(self.num_classes))

    def get_image_postfix(self, image_dir, image_id):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." in image_id[0]:
            postfix = ""
        else:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            postfix = os.path.basename(image_list[0]).split(".")[1]
        return postfix

    def __get_image_anno_file(self, image_dir, anno_dir, image_id: str, img_postfix):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :param img_postfix:
        :return:
        """
        if not img_postfix and "." in image_id:
            img_postfix = image_id.split(".")[-1]
            image_id = image_id[:-len(img_postfix) - 1]
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.xml".format(image_id))
        return image_file, annotation_file

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        print("Please wait, it's in checking")
        dst_ids = []
        # image_ids = image_ids[:100]
        # image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            objects = self.get_annotation(annotation_file)
            bboxes, labels, is_difficult = objects["bboxes"], objects["labels"], objects["is_difficult"]
            if ignore_empty and (len(bboxes) == 0 or len(labels) == 0):
                print("empty annotation:{}".format(annotation_file))
                continue
            dst_ids.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def parser_classes(self, class_name):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        :param class_name:
                    str : class file
                    list: ["face","person"]
                    dict: 可以自定义label的id{'BACKGROUND': 0, 'person': 1, 'person_up': 1, 'person_down': 1}
        :return:
        """
        if isinstance(class_name, str):
            class_name = super().read_files(class_name)
        elif isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
        if isinstance(class_name, list):
            class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        elif isinstance(class_name, dict):
            class_dict = class_name
            class_name = list(class_dict.keys())
        else:
            class_dict = None
        return class_name, class_dict

    def parser_paths(self, filename=None, data_root=None, anno_dir=None, image_dir=None):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :return:
        """
        if isinstance(data_root, str):
            anno_dir = os.path.join(data_root, "Annotations") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "JPEGImages") if not image_dir else image_dir
        image_id = []
        if isinstance(filename, str):
            data_root = os.path.dirname(filename)
            image_id = self.read_files(filename)
        if not anno_dir:
            anno_dir = os.path.join(data_root, "Annotations")
        if not image_dir:
            image_dir = os.path.join(data_root, "JPEGImages")
        if anno_dir and not image_id:
            image_id = self.get_file_list(anno_dir, postfix=["*.xml"], basename=True)
        elif image_dir and not image_id:
            image_id = self.get_file_list(anno_dir, postfix=["*.jpg"], basename=True)
        return data_root, anno_dir, image_dir, image_id

    def convert_target(self, boxes, labels):
        # （xmin,ymin,xmax,ymax,label）
        if len(boxes) == 0:
            target = np.empty(shape=(0, 5), dtype=np.float32)
        else:
            target = np.concatenate([boxes, labels.reshape(-1, 1)], axis=1)
            target = np.asarray(target, dtype=np.float32)
        return target

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.get_image_anno_file(image_id)
        objects = self.get_annotation(annotation_file)
        # labels必须是vector
        bboxes, labels, is_difficult = objects["bboxes"], objects["labels"], objects["is_difficult"]
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        height, width = image.shape[:2]
        num_boxes = len(bboxes)
        if self.transform and len(bboxes) > 0:
            image, bboxes, labels = self.transform(image, bboxes, labels)
        # bboxes, labels = self.target_transform(bboxes, labels)  # torch.Size([29952, 4]),torch.Size([29952])
        target = self.convert_target(bboxes, labels)
        if num_boxes == 0 or len(labels) == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        # return image, bboxes, labels
        # return image, {"target": target, "image_id": image_id, "size": [width, height]}
        data = {"image": image, "target": target, "image_id": image_id,
                "size": [width, height], "image_file": image_file}
        return data

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        return image_file, annotation_file

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, numbers.Number):
            image_id = self.image_id[index]
        else:
            image_id = index
        return image_id

    def __len__(self):
        return len(self.image_id)

    def get_annotation(self, xml_file):
        """
        :param xml_file:
        :param class_dict: class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        :return:
        """
        try:
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])
            filename = annotation["filename"]
            objects = annotation["object"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            objects = []
        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            name = str(object["name"]) if not self.unique else "unique"
            name = "BACKGROUND" if str(object["name"]) == 'BACKGROUND' else name
            if self.class_name and name not in self.class_name:
                continue
            difficult = int(object["difficult"]) if 'difficult' in object else 0
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            # rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            bbox = self.clip_box(bbox, width, height)
            item = {}
            item["bbox"] = bbox
            item["difficult"] = difficult
            if self.class_dict:
                name = self.class_dict[name]
            item["name"] = name
            objects_list.append(item)
        bboxes, labels, is_difficult = self.get_objects_items(objects_list)
        objects = {"bboxes": bboxes,
                   "labels": labels,
                   "is_difficult": is_difficult,
                   "width": width,
                   "height": height,
                   }
        return objects

    def get_objects_items(self, objects_list):
        """
        :param objects_list:
        :return:
        """
        bboxes = []
        labels = []
        is_difficult = []
        for item in objects_list:
            bboxes.append(item["bbox"])
            labels.append(item['name'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels)  # for string
        # labels = np.array(labels, dtype=np.int64)  # for int64
        # labels = np.asarray(labels).reshape(-1, 1)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, is_difficult

    @staticmethod
    def get_files_id(file_list):
        """
        :param file_list:
        :return:
        """
        image_idx = []
        for path in file_list:
            basename = os.path.basename(path)
            id = basename.split(".")[0]
            image_idx.append(id)
        return image_idx

    @staticmethod
    def read_xml2json(xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def read_image(self, image_file: str, use_rgb=True):
        """
        :param image_file:
        :param use_rgb:
        :return:
        """
        # image = cv2.imread(image_file)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        # image = cv2.imread(image_file, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class ConcatDataset(Dataset):
    """ Concat Dataset """

    def __init__(self, datasets, shuffle=False):
        """
        import torch.utils.data as torch_utils
        voc1 = PolygonParser(filename1)
        voc2 = PolygonParser(filename2)
        voc=torch_utils.ConcatDataset([voc1, voc2])
        ====================================
        :param datasets:
        :param shuffle:
        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'dataset should not be an empty iterable'
        # super(ConcatDataset, self).__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.image_id = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_id = dataset.image_id
            image_id = self.add_dataset_id(image_id, dataset_id)
            self.image_id += image_id
            self.classes = dataset.classes
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)

    def add_dataset_id(self, image_id, dataset_id):
        """
        :param image_id:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for id in image_id:
            out_image_id.append({"dataset_id": dataset_id, "image_id": id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        dataset = self.dataset[dataset_id]
        # print(dataset.data_root, image_id)
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, xml_file):
        return self.dataset[0].get_annotation(xml_file)

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, use_rgb=self.dataset[0].use_rgb)

    def __len__(self):
        return len(self.image_id)


def VOCDatasets(filename=None,
                data_root=None,
                anno_dir=None,
                image_dir=None,
                class_name=None,
                transform=None,
                use_rgb=True,
                shuffle=False,
                check=False):
    """
    :param filename:
    :param data_root:
    :param anno_dir:
    :param image_dir:
    :param class_name:
    :param transform:
    :param use_rgb:
    :param shuffle:
    :param check:
    :return:
    """
    if not isinstance(filename, list) and os.path.isfile(filename):
        filename = [filename]
    datasets = []
    for file in filename:
        data = VOCDataset(filename=file,
                          data_root=data_root,
                          anno_dir=anno_dir,
                          image_dir=image_dir,
                          class_name=class_name,
                          transform=transform,
                          use_rgb=use_rgb,
                          shuffle=shuffle,
                          check=check)
        datasets.append(data)
    datasets = ConcatDataset(datasets, shuffle=shuffle)
    return datasets


def show_target_image(image, bboxes, labels, normal=False, transpose=False, class_name=None, use_rgb=True):
    """
    :param image:
    :param targets_t:
                bboxes = targets[idx][:, :4].data
                keypoints = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from pybaseutils import image_utils
    image = np.asarray(image)
    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    # print("image:{}".format(image.shape))
    # print("bboxes:{}".format(bboxes))
    # print("labels:{}".format(labels))
    if transpose:
        image = image_utils.untranspose(image)
    h, w, _ = image.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        bboxes = bboxes * bboxes_scale
    # image = image_processing.untranspose(image)
    # image = image_processing.convert_color_space(image, colorSpace="RGB")
    image = image_utils.draw_image_bboxes_labels(image, bboxes, labels, class_name=class_name)
    image_utils.cv_show_image("image", image, delay=0, use_rgb=use_rgb)
    print("===" * 10)


if __name__ == "__main__":
    # from models.transforms import data_transforms
    # filename = '/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset/test.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/SafetyHelmetWearingDataset/VOC/train.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/Helmet_Dataset(kaggle)/helmet_dataset/train.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/Hard Hat Workers.v2-raw.voc/trainval.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/Helmet-Asian/total.txt'
    filename = '/home/dm/nasdata/dataset/csdn/helmet/Helmet-Europe/trainval.txt'
    # class_dict = ['BACKGROUND', 'unique']
    class_dict = {'head': 0, "helmet": 1}
    class_name = ['head', "helmet"]
    dataset = VOCDatasets(filename=[filename],
                          data_root=None,
                          anno_dir=None,
                          image_dir=None,
                          class_name=class_dict,
                          transform=None,
                          check=False,
                          shuffle=True)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        print(i)
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        print(image_id)
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        show_target_image(image, bboxes, labels, normal=False, transpose=False, class_name=class_name)
        # show_boxes_image(image, Dataset.cxcywh2xyxy(bboxes, 0, 0), labels, normal=False, transpose=True)
