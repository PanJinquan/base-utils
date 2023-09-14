# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
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
from pybaseutils.dataloader.base_dataset import Dataset
from pybaseutils.dataloader import voc_seg_utils
from pybaseutils import file_utils


class VOCDataset(Dataset):
    """
    VOC数据格式解析器
    数据格式：(xmin,ymin,xmax,ymax)
    输出格式：target is (xmin,ymin,xmax,ymax,label)
    """

    def __init__(self,
                 filename=None,
                 data_root=None,
                 image_dir=None,
                 anno_dir=None,
                 seg_dir=None,
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
        self.data_root, self.anno_dir, self.image_dir, self.image_ids = parser
        self.seg_dir = seg_dir
        if check or (self.class_dict is None):
            self.image_ids = self.checking(self.image_ids, class_dict=self.class_dict)
        self.transform = transform
        self.use_rgb = use_rgb
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        self.num_images = len(self.image_ids)
        # print("VOCDataset class_count:{}".format(class_count))
        print("VOCDataset class_name      :{}".format(class_name))
        print("VOCDataset class_dict      :{}".format(self.class_dict))
        print("VOCDataset num images      :{}".format(len(self.image_ids)))
        print("VOCDataset num_classes     :{}".format(self.num_classes))
        print("------" * 10)

    def __get_image_anno_file(self, image_dir, anno_dir, image_name: str):
        """
        :param image_dir:
        :param anno_dir:
        :param image_name:
        :param img_postfix:
        :return:
        """
        img_postfix = image_name.split(".")[-1]
        image_id = image_name[:-len(img_postfix) - 1]
        image_file = os.path.join(image_dir, image_name)
        annotation_file = os.path.join(anno_dir, "{}.xml".format(image_id))
        return image_file, annotation_file

    def checking(self, image_ids: list, class_dict: dict, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        print("Please wait, it's in checking")
        dst_ids = []
        # image_ids = image_ids[:100]
        # image_ids = image_ids[100:]
        class_set = []
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                # print(image_file)
                # os.remove(image_file)
                continue
            if not os.path.exists(image_file):
                continue
            objects = self.get_annotation(annotation_file)
            boxes, labels, is_difficult = objects["boxes"], objects["labels"], objects["is_difficult"]
            class_set = labels.reshape(-1).tolist() + class_set
            class_set = list(set(class_set))
            if ignore_empty and (len(boxes) == 0 or len(labels) == 0):
                print("empty annotation:{}".format(annotation_file))
                continue
            dst_ids.append(image_id)
        class_set = sorted(class_set)
        if not class_dict:
            print("class_name is None, Dataset will auto get class_set :{}".format(class_set))
            self.class_name = class_set
            self.class_dict = {class_name: i for i, class_name in enumerate(class_set)}
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
            class_name = Dataset.read_files(class_name)
        elif isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
        if isinstance(class_name, list) and len(class_name) > 0:
            class_dict = {}
            for i, name in enumerate(class_name):
                name = name.split(",")
                for n in name: class_dict[n] = i
        elif isinstance(class_name, dict) and len(class_name) > 0:
            class_dict = class_name
        else:
            class_dict = None
        if class_dict:
            class_name = {}
            for n, i in class_dict.items():
                class_name[i] = "{},{}".format(class_name[i], n) if i in class_name else n
            class_name = list(class_name.values())
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
        image_ids = []
        if isinstance(filename, str) and filename:
            data_root = os.path.dirname(filename)
            image_ids = self.read_files(filename)
        if not anno_dir:
            anno_dir = os.path.join(data_root, "Annotations")
        if not image_dir:
            image_dir = os.path.join(data_root, "JPEGImages")
        if image_dir and not image_ids:
            image_ids = self.get_file_list(image_dir, postfix=file_utils.IMG_POSTFIX, basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        elif anno_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=["*.xml"], basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        return data_root, anno_dir, image_dir, image_ids

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
        boxes, labels, is_difficult = objects["boxes"], objects["labels"], objects["is_difficult"]
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        height, width = image.shape[:2]
        num_boxes = len(boxes)
        if self.transform and len(boxes) > 0:
            image, boxes, labels = self.transform(image, boxes, labels)
        # boxes, labels = self.target_transform(boxes, labels)  # torch.Size([29952, 4]),torch.Size([29952])
        target = self.convert_target(boxes, labels)
        if num_boxes == 0 or len(labels) == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        # return image, boxes, labels
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
        image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id)
        return image_file, annotation_file

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, numbers.Number):
            image_id = self.image_ids[index]
        else:
            image_id = index
        return image_id

    def __len__(self):
        return len(self.image_ids)

    def get_segment_info(self, filename, bbox):
        """
        :param filename:
        :param bbox:[xmin, ymin, xmax, ymax]
        :return:
        """
        seg = []
        area = 0
        if self.seg_dir:
            # if exist VOC SegmentationObject
            seg_path = os.path.join(self.seg_dir, filename.split('.')[0] + '.png')
            seg, area = voc_seg_utils.get_segment_area(seg_path, bbox)
        if not seg:
            # cal stroke_segs and area by bbox
            xmin, ymin, xmax, ymax = bbox
            seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            area = (xmax - xmin) * (ymax - ymin)
        return seg, area

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
            objects = annotation["object"]
            # filename = annotation["filename"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            objects = []
            width = None
            height = None
        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            name = str(object["name"]) if not self.unique else "unique"
            name = "BACKGROUND" if str(object["name"]) == 'BACKGROUND' else name
            if self.class_dict and name not in self.class_dict:
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
        boxes, labels, is_difficult = self.get_objects_items(objects_list)
        objects = {"boxes": boxes,
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
        boxes = []
        labels = []
        is_difficult = []
        for item in objects_list:
            boxes.append(item["bbox"])
            labels.append(item['name'])
            is_difficult.append(item['difficult'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels)  # for string
        # labels = np.array(labels, dtype=np.int64)  # for int64
        # labels = np.asarray(labels).reshape(-1, 1)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return boxes, labels, is_difficult

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
        image = cv2.imread(image_file)
        # image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
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
        self.image_ids = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_ids = dataset.image_ids
            image_ids = self.add_dataset_id(image_ids, dataset_id)
            self.image_ids += image_ids
            self.classes = dataset.classes
            self.class_name = dataset.class_name
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        print("ConcatDataset total images :{}".format(len(self.image_ids)))
        print("ConcatDataset class_name   :{}".format(self.class_name))
        print("------" * 10)


    def add_dataset_id(self, image_ids, dataset_id):
        """
        :param image_ids:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for image_id in image_ids:
            out_image_id.append({"dataset_id": dataset_id, "image_id": image_id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_ids[index]["dataset_id"]
        image_id = self.image_ids[index]["image_id"]
        dataset = self.dataset[dataset_id]
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = self.image_ids[index]["dataset_id"]
        image_id = self.image_ids[index]["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, xml_file):
        return self.dataset[0].get_annotation(xml_file)

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, use_rgb=self.dataset[0].use_rgb)

    def __len__(self):
        return len(self.image_ids)


def VOCDatasets(filename=None,
                data_root=None,
                image_dir=None,
                anno_dir=None,
                class_name=None,
                transform=None,
                use_rgb=True,
                shuffle=False,
                check=False):
    """
    :param filename:
    :param data_root:
    :param image_dir:
    :param anno_dir:
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
                          image_dir=image_dir,
                          anno_dir=anno_dir,
                          class_name=class_name,
                          transform=transform,
                          use_rgb=use_rgb,
                          shuffle=shuffle,
                          check=check)
        datasets.append(data)
    datasets = ConcatDataset(datasets, shuffle=shuffle)
    return datasets


def show_target_image(image, boxes, labels, normal=False, transpose=False, class_name=None, use_rgb=True,
                      thickness=2, fontScale=1.0):
    """
    :param image:
    :param targets_t:
                boxes = targets[idx][:, :4].data
                keypoints = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from pybaseutils import image_utils
    image = np.asarray(image)
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    # print("image:{}".format(image.shape))
    # print("boxes:{}".format(boxes))
    # print("labels:{}".format(labels))
    if transpose:
        image = image_utils.untranspose(image)
    h, w, _ = image.shape
    landm_scale = np.asarray([w, h] * 5)
    boxes_scale = np.asarray([w, h] * 2)
    if normal:
        boxes = boxes * boxes_scale
    image = image_utils.draw_image_bboxes_labels(image, boxes, labels, class_name=class_name,
                                                 thickness=thickness, fontScale=fontScale, drawType="chinese")
    image_utils.cv_show_image("image", image, delay=0, use_rgb=use_rgb)
    return image


if __name__ == "__main__":
    # from models.transforms import data_transforms
    filename = '/media/PKing/新加卷1/SDK/base-utils/data/coco/file_list.txt'
    filename = '/home/PKing/nasdata/dataset/tmp/gesture/Light-HaGRID/trainval/call/val.txt'
    class_name = None
    dataset = VOCDatasets(filename=[filename, filename],
                          data_root=None,
                          image_dir=None,
                          anno_dir=None,
                          class_name=class_name,
                          transform=None,
                          check=False,
                          shuffle=True)
    class_name = dataset.class_name
    for i in range(len(dataset)):
        print(i)
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        print(image_id)
        boxes, labels = targets[:, 0:4], targets[:, 4:5]
        show_target_image(image, boxes, labels, normal=False, transpose=False, class_name=class_name)
        # show_boxes_image(image, Dataset.cxcywh2xyxy(boxes, 0, 0), labels, normal=False, transpose=True)
        print("===" * 10)
