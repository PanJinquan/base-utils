# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
import numpy as np
import cv2
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pybaseutils.dataloader.base_dataset import Dataset
from pybaseutils import image_utils, file_utils, json_utils


class COCORebuild(COCO):
    def __init__(self, annotation):
        """
        Args:
            annotation:  COCO annotation file(*.json) or COCO dataset
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if isinstance(annotation, str):
            with open(annotation, 'r') as f:
                dataset = json.load(f)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        else:
            dataset = annotation
        self.dataset = dataset
        self.createIndex()
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    @staticmethod
    def coco_categories_remapping(anno_file, class_dict: dict):
        """
        coco类别重新映射
        :param anno_file:  COCO annotation file(*.json).
        :param class_dict: 重新映射的类别
        :return:
        """
        assert isinstance(class_dict, dict)
        if "unique" in class_dict:
            class_dict["unique"] = "unique"
        with open(anno_file, 'r') as f:
            annotation = json.load(f)
        # 原始的categories
        old_categories = {}
        for cat in annotation["categories"]:
            id, name = cat["id"], cat["name"]
            if name in class_dict or "unique" in class_dict:
                old_categories[id] = name
            else:
                raise Exception("Error:{} not in {}".format(name, class_dict))
        class_name = list(set(class_dict.values()))
        id2categories = {id: class_name[id] for id in range(len(class_name))}
        categories2id = {cat: id for id, cat in id2categories.items()}
        # 重新映射新的categories
        new_categories = []
        for name in class_name:
            id = categories2id[name]
            cat = {"supercategory": name, "id": id, "name": name}
            new_categories.append(cat)
        annotation["categories"] = new_categories
        # 重新映射annotations中的category_id
        for anno in annotation['annotations']:
            category_id = anno["category_id"]
            name = old_categories[category_id]
            if "unique" in class_dict:
                name = "unique"
            name = class_dict[name]
            anno["category_id"] = categories2id[name]
        return annotation


def load_coco(annotation, class_name=None) -> COCO:
    """
    :param annotation:
    :param class_name: 重新映射的类别(有BUG，暂时不用)
    :return:
    """
    class_name = None
    if isinstance(class_name, dict):
        annotation = COCORebuild.coco_categories_remapping(annotation, class_name)
    elif isinstance(class_name, list) and "unique" in class_name:
        class_name = {class_name[i]: i for i in range(len(class_name))}
        annotation = COCORebuild.coco_categories_remapping(annotation, class_name)
    coco = COCORebuild(annotation)
    return coco


class CocoDataset(object):
    """Coco dataset."""

    def __init__(self, anno_file, image_dir="", class_name=[], transform=None, target_transform=None, use_rgb=True,
                 shuffle=False, decode=True, **kwargs):
        """
        ├── annotations
        │    ├── instances_train2017.json
        │    └── person_keypoints_train2017.json
        └── images
        :param anno_file: COCO annotation file(*.json).
        :param image_dir: COCO image directory. 如果image_dir为空，则自动搜寻可能存在图片目录
        :param transform:(callable, optional): Optional transform to be applied on a sample.
        :param target_transform:
        :param use_rgb:
        :param shuffle:
        :param decode: 是否对segment进行解码， True:在mask显示分割信息,False：mask为0，无分割信息
        """
        super(CocoDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.image_dir, self.anno_dir = self.parser_paths(anno_file, image_dir)
        self.unique = False
        self.decode = decode
        self.use_rgb = use_rgb
        self.coco = load_coco(anno_file)
        self.category2id = self.load_categories()  # 获得COCO所有种类(类别)category,如{"person":0}
        self.id2category = {i: c for c, i in self.category2id.items()}
        if not class_name:
            class_name = list(self.category2id.keys())
            class_name = ["BACKGROUND"] + class_name
        self.class_name, self.class_dict = self.parser_classes(class_name)
        self.label2category_id = {l: self.category2id[c] for c, l in self.class_dict.items() if c in self.category2id}
        self.category_id2label = {c: l for l, c in self.label2category_id.items()}
        # 所有数据的image id
        self.image_ids, self.class_count = self.get_image_ids(self.class_dict)
        self.files_info = self.get_files_info(self.image_ids)
        self.annos_info = self.get_annos_info(self.image_ids)
        assert self.files_info == self.files_info
        # self.image_ids = self.image_ids[:100]
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        self.num_images = len(self.image_ids)
        self.classes = list(set(self.class_dict.values())) if self.class_dict else []
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else 0
        self.bones = {}
        print("CocoDataset anno_file  :{}".format(anno_file))
        print("CocoDataset image_dir  :{}".format(self.image_dir))
        print("CocoDataset class_count:{}".format(self.class_count))
        print("CocoDataset class_name :{}".format(self.class_name))
        print("CocoDataset class_dict :{}".format(self.class_dict))
        print("CocoDataset num images :{}".format(len(self.image_ids)))
        print("CocoDataset num_classes:{}".format(self.num_classes))
        print("------" * 10)

    def parser_paths(self, filename=None, image_dir=None):
        """
        :param filename:
        :param image_dir:
        :return:
        """
        assert os.path.exists(filename), Exception("no file:{}".format(filename))
        anno_dir = os.path.dirname(filename)
        data_root = os.path.dirname(anno_dir)
        if not image_dir:  # 如果image_dir为空，则自动搜寻可能存在图片目录
            image_sub = ["JPEGImages", "images", "image"]
            for sub in image_sub:
                image_dir = os.path.join(anno_dir, sub)
                if os.path.exists(image_dir):
                    break
                image_dir = os.path.join(data_root, sub)
                if os.path.exists(image_dir):
                    break
        assert os.path.exists(image_dir), Exception("no directory:{}".format(image_dir))
        return image_dir, anno_dir

    def get_image_ids(self, class_dict: dict):
        """过滤符合条件的image id"""
        if "unique" in class_dict:
            CatIds = self.coco.getCatIds(catNms=[])  # catNms is cat names
        else:
            CatIds = self.coco.getCatIds(catNms=list(class_dict.keys()))  # catNms is cat names
        # image_ids = self.coco.getImgIds(catIds=[])  # getImgIds返回的是图片中同时存在class_name的图片
        class_count = {}
        image_ids = set()
        for cat in CatIds:
            id = self.coco.getImgIds(catIds=cat)  # 满足class_name其中一个类别即可
            name = self.coco.loadCats(cat)[0]["name"]
            class_count[name] = len(id)
            image_ids = image_ids | set(id)  # 并集
        image_ids = list(image_ids)
        return image_ids, class_count

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
            class_name = Dataset.read_file(class_name)
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

    def load_categories(self):
        """
        # display COCO categories and supercategories
        catIds = self.coco.getCatIds()  # 获得所有种类的id,
        cats = self.coco.loadCats(catIds)  # 获得所有超类
        :return:
        """
        # COCO数据集catIds有1-90，但coco_name只有80个，因此其下标不一致对应的
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        category2id = {}
        for i, cat in enumerate(categories):
            category2id[cat['name']] = cat['id']
        return category2id

    def load_keypoints_info(self, target=["person"]):
        categories = self.coco.loadCats(self.coco.getCatIds(catNms=target))
        return categories

    def __len__(self):
        return len(self.image_ids)

    def get_object_annotations(self, ids):
        # get ground truth annotations
        annos_ids = self.coco.getAnnIds(imgIds=[ids], iscrowd=False)
        file_info = self.coco.loadImgs(ids=[ids])[0]
        # parse annotations
        anno_info = self.coco.loadAnns(annos_ids)
        return anno_info, file_info

    def get_files_info(self, image_ids: list = None):
        if not image_ids: image_ids = self.image_ids
        file_info = self.coco.loadImgs(ids=image_ids)
        return file_info

    def get_annos_info(self, image_ids: list = None):
        if not image_ids: image_ids = self.image_ids
        anno_info = []
        for ids in image_ids:
            ids = self.coco.getAnnIds(imgIds=[ids], iscrowd=False)
            ann = self.coco.loadAnns(ids)
            anno_info.append(ann)
        return anno_info

    def get_object_image(self, file_info):
        """
        读取图片信息
        :param file_info:
        :return:
        """
        image_file = os.path.join(self.image_dir, file_info['file_name'])
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        height, width = image.shape[:2]
        assert width == file_info['width']
        assert height == file_info['height']
        return image, width, height, image_file

    def get_object_detection(self, annos):
        """
        return: boxes=(num_boxes,5), xmin,ymin,xmax,ymax
        """
        labels, rects = [], []
        for anno in annos:
            # some annotations have basically no width / height, skip them
            if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                continue
            name = self.id2category[anno['category_id']] if not self.unique else "unique"
            if self.class_dict and name not in self.class_dict:
                continue
            bbox = anno['bbox']  # x,y,w,h
            label = self.class_dict[name]
            rects.append(bbox)
            labels.append(label)
        labels = np.asarray(labels)
        boxes = image_utils.xywh2xyxy(np.asarray(rects))
        return boxes, labels

    def get_object_instance(self, anns, h, w, decode=False):
        """
        获得实例分割信息
        :param anns:
        :param h:
        :param w:
        :param decode: 是否对segment进行解码， True:在mask显示分割信息,False：mask为0，无分割信息
        :return:
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        segs, labels, rects = [], [], []
        for ann in anns:
            name = self.id2category[ann['category_id']] if not self.unique else "unique"
            if self.class_dict and name not in self.class_dict:
                continue
            if len(ann['segmentation']) == 0: continue
            seg = ann['segmentation'][0]
            # polygons = image_utils.find_mask_contours(m) # bug：多个实例时，bbox有问题
            # bbox = image_utils.polygons2boxes(polygons)[0]
            label = self.class_dict[name]
            rects.append(ann['bbox'])
            labels.append(label)
            seg = [np.array(seg, dtype=np.int32).reshape(-1, 2)]
            segs.append(seg)
            # parse mask
            if not decode: continue
            # m = coco_mask.decode(coco_mask.frPyObjects(seg, h, w))
            m = self.coco.annToMask(ann)
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * label)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * label).astype(np.uint8)
        labels = np.asarray(labels)
        boxes = image_utils.xywh2xyxy(np.asarray(rects))
        return boxes, labels, mask, segs

    def get_keypoint_info(self, anns, num_joints):
        """
        获得关键点信息
        keypoints=17*3,x,y,visibility
        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        :param anns:
        :return:
        """
        keypoints, labels, rects, boxes = [], [], [], []
        for ann in anns:
            name = self.id2category[ann['category_id']] if not self.unique else "unique"
            if self.class_dict and name not in self.class_dict:
                continue
            keypoint = ann.get('keypoints', [])
            if len(keypoint) == 0: continue
            keypoint = np.asarray(keypoint).reshape(num_joints, 3)
            keypoint = keypoint[:, 0:2]
            keypoints.append(keypoint)
            label = self.class_dict[name]
            rects.append(ann['bbox'])
            labels.append(label)
        if len(labels) > 0:
            labels = np.asarray(labels)
            boxes = image_utils.xywh2xyxy(np.asarray(rects))
        return boxes, labels, keypoints

    def read_image(self, image_file: str, use_rgb=True):
        """
        :param image_file:
        :param use_rgb:
        :return:
        """
        image = cv2.imread(image_file)
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def set_skeleton_keypoints(self, ids, skeleton=[], keypoints=[]):
        """
        设置COCO的skeleton和keypoints
        :param ids: 类别ID
        :param skeleton: COCO数据集的skeleton下标是从1开始的
        :param keypoints:
        :return:
        """
        if isinstance(skeleton, np.ndarray): skeleton = skeleton.tolist()
        if isinstance(keypoints, np.ndarray): keypoints = keypoints.tolist()
        if len(skeleton) > 0: self.coco.loadCats(ids)[0]['skeleton'] = skeleton
        if len(keypoints) > 0: self.coco.loadCats(ids)[0]['keypoints'] = keypoints

    def showAnns(self, image, annotations: dict):
        """
        显示标注信息
        :param image:
        :param annotations:
        :return:
        """
        if len(annotations) == 0: return
        plt.imshow(image), plt.axis('off')
        self.coco.showAnns(annotations), plt.show()


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
        self.classes = []
        self.class_name = []
        self.bones = {}
        for dataset_id, dataset in enumerate(self.dataset):
            # image_ids = dataset.image_ids
            image_ids = list(range(len(dataset.image_ids)))
            image_ids = self.add_dataset_id(image_ids, dataset_id)
            self.image_ids += image_ids
            self.classes = dataset.classes
            self.class_name = dataset.class_name
            self.bones = dataset.bones
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

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, use_rgb=self.dataset[0].use_rgb)

    def __len__(self):
        return len(self.image_ids)
