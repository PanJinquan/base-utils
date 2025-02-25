# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
import numpy as np
import cv2
import glob
import random
import numbers
import json
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, text_utils
from pybaseutils.dataloader.base_dataset import Dataset, ConcatDataset


class LabelMeDataset(Dataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_name=None,
                 use_rgb=False,
                 shuffle=False,
                 check=False,
                 min_points=-1,
                 **kwargs):
        """
        dataset.image_ids
        dataset.classes
        dataset.class_name
        要求该目录下存在images和json
        data_root，anno_dir只要存在一个即可，程序会自动搜索images和json
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param class_name: 当class_name=None且check=True,将自动获取所有class,当class_name=[]会直接返回name
        :param use_rgb:
        :param shuffle:
        :param check: 当class_name=None且check=True,将自动获取所有class
        :param min_points: 当标注的轮廓点的个数小于min_points，会被剔除；负数不剔除
        :param kwargs: read_image: 是否读取图片，否则image=None
        """
        super(LabelMeDataset, self).__init__()
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.min_points = min_points
        self.kwargs = kwargs
        self.class_name, self.class_dict = self.parser_classes(class_name)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_ids = parser
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.class_weights = None
        # self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        if check:
            self.image_ids = self.checking(self.image_ids)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        self.num_images = len(self.image_ids)
        print("LabelMeDataset data_root     :{}".format(self.data_root))
        print("LabelMeDataset anno_dir      :{}".format(self.anno_dir))
        print("LabelMeDataset image_dir     :{}".format(self.image_dir))
        print("LabelMeDataset class_name    :{}".format(self.class_name))
        print("LabelMeDataset class_dict    :{}".format(self.class_dict))
        print("LabelMeDataset num images    :{}".format(len(self.image_ids)))
        # print("LabelMeDataset num_classes   :{}".format(self.num_classes))
        print("------" * 10)

    def __len__(self):
        return len(self.image_ids)

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, anno_file, image_id = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id)
        return image_file, anno_file, image_id

    def __get_image_anno_file(self, image_dir, anno_dir, image_name: str):
        """
        :param image_dir:
        :param anno_dir:
        :param image_name:
        :param img_postfix:
        :return:
        """
        image_file = os.path.join(image_dir, image_name)
        img_postfix = image_name.split(".")[-1]
        image_id = image_name[:-len(img_postfix) - 1]
        anno_file = os.path.join(anno_dir, "{}.json".format(image_id))
        return image_file, anno_file, image_name

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        print("Please wait, it's in checking")
        dst_ids = []
        class_name = []
        for image_id in tqdm(image_ids):
            image_file, anno_file, image_id = self.get_image_anno_file(image_id)
            if not os.path.exists(anno_file):
                continue
            if not os.path.exists(image_file):
                continue
            annotation, width, height = self.load_annotations(anno_file)
            info = self.parser_annotation(annotation, self.class_dict, min_points=self.min_points, unique=self.unique)
            labels = info["labels"]
            if len(labels) == 0:
                continue
            dst_ids.append(image_id)
            class_name += labels
        if self.class_name is None:
            class_name = sorted(list(set(class_name)))
            self.class_name, self.class_dict = self.parser_classes(class_name)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def parser_paths(self, filename=None, data_root=None, anno_dir=None, image_dir=None):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :return:
        """
        if isinstance(data_root, str):
            anno_dir = os.path.join(data_root, "json") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "images") if not image_dir else image_dir
        image_ids = []
        if isinstance(filename, str) and filename:
            image_ids = self.read_file(filename, split=",")
            data_root = os.path.dirname(filename)
        if not anno_dir:  # 如果anno_dir为空，则自动搜寻可能存在图片目录
            anno_dir = self.search_path(data_root, sub_dir=["json"])
        if not data_root and anno_dir:  #
            data_root = os.path.dirname(anno_dir)
            image_dir = self.search_path(data_root, ["images", "JPEGImages"])
        if not image_dir:
            image_dir = self.search_path(data_root, ["images", "JPEGImages"])
        if image_dir and not image_ids:
            image_ids = self.get_file_list(image_dir, postfix=file_utils.IMG_POSTFIX, basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        elif anno_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=["*.json", "*.xml"], basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        assert isinstance(anno_dir, str) and os.path.exists(anno_dir), "no anno_dir :{}".format(anno_dir)
        assert isinstance(image_dir, str) and os.path.exists(image_dir), "no image_dir:{}".format(image_dir)
        assert len(image_ids) > 0, f"image_ids is empty,image_dir={image_dir},anno_dir={anno_dir}"
        return data_root, anno_dir, image_dir, image_ids

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        image_file, anno_file, image_id = self.get_image_anno_file(image_id)
        annotation, width, height = self.load_annotations(anno_file)
        if self.kwargs.get("read_image", True):  # 是否读取图片
            image = self.read_image(image_file, use_rgb=self.use_rgb)
            shape = image.shape
            size = [shape[1], shape[0]]
        else:
            image, shape, size = None, None, [width, height]
        data_info = self.parser_annotation(annotation, self.class_dict, shape=shape,
                                           min_points=self.min_points, unique=self.unique)
        # TODO dict(boxes, labels, points, groups, names, keypoints)
        data_info.update({"image": image, "image_file": image_file, "anno_file": anno_file,
                          "size": size})
        return data_info

    @staticmethod
    def parser_annotation(annotation: dict, class_dict={}, shape=None, min_points=-1, unique=False):
        """
        :param annotation:  labelme标注的数据
        :param class_dict:  label映射
        :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
        :param min_points: 当标注的轮廓点的个数小于等于min_points，会被剔除；负数不剔除
        :return:
        """
        bboxes, labels, points, groups, names, keypoints = [], [], [], [], [], []
        for anno in annotation:
            name = "unique" if unique else anno["label"]
            shape_type = anno.get("shape_type", "polygon")
            label = name
            if class_dict:
                if not name in class_dict:
                    continue
                if isinstance(class_dict, dict):
                    label = class_dict[name]
                    if isinstance(label, str): name = label
            pts = np.asarray(anno["points"], dtype=np.int32)
            if min_points > 0 and len(pts) <= min_points:
                continue
            gid = json_utils.get_value(anno, key=["group_id"], default=0)
            gid = gid if gid else 0
            kpt = json_utils.get_value(anno, key=["keypoints"], default=[])
            if shape:
                h, w = shape[:2]
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            box = image_utils.polygons2boxes([pts])[0]
            if shape_type == "rectangle":
                pts = image_utils.boxes2polygons([box])[0]
            names.append(name)
            labels.append(label)
            bboxes.append(box)
            points.append(pts)
            groups.append(gid)
            keypoints.append(kpt)
        return dict(boxes=bboxes, labels=labels, points=points, groups=groups, names=names, keypoints=keypoints)

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

    def read_image(self, image_file: str, use_rgb=True):
        """
        :param image_file:
        :param use_rgb:
        :return:
        """
        try:
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = image[:, :, 0:3]
            if use_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise Exception("empty image:{}".format(image_file))
        return image

    def get_keypoint_object(self, annotation: list, w, h, class_name=[]):
        """
        获得labelme关键点检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :return:
        """
        objects = {}
        for i, anno in enumerate(annotation):
            label = anno["label"]
            points = np.asarray(anno["points"], dtype=np.int32)
            group_id = anno["group_id"] if "group_id" in anno and anno["group_id"] else 0  # 通过group_id标记同一实例
            if file_utils.is_int(label):
                keypoints: dict = json_utils.get_value(objects, [group_id, "keypoints"], default={})
                keypoints.update({int(label): points.tolist()[0]})
                objects = json_utils.set_value(objects, key=[group_id, "keypoints"], value=keypoints)
            elif label in class_name:
                contours = points
                contours[:, 0] = np.clip(contours[:, 0], 0, w - 1)
                contours[:, 1] = np.clip(contours[:, 1], 0, h - 1)
                boxes = image_utils.polygons2boxes([contours])
                if group_id in objects:
                    objects[group_id].update({"labels": label, "boxes": boxes[0], "segs": contours})
                else:
                    objects[group_id] = {"labels": label, "boxes": boxes[0], "segs": contours}
        return objects

    def get_instance_object(self, annotation: list, w, h, class_name=[]):
        """
        获得labelme实例分割/检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :return:
        """
        objects = {}
        for i, anno in enumerate(annotation):
            label = anno["label"]
            points = np.asarray(anno["points"], dtype=np.int32)
            group_id = i
            if file_utils.is_int(label):
                continue
            elif class_name is None or len(class_name) == 0 or label in class_name:
                segs = points
                segs[:, 0] = np.clip(segs[:, 0], 0, w - 1)
                segs[:, 1] = np.clip(segs[:, 1], 0, h - 1)
                box = image_utils.polygons2boxes([segs])[0]
                objects = json_utils.set_value(objects, key=[group_id],
                                               value={"labels": label, "boxes": box, "segs": segs})
        return objects

    @staticmethod
    def load_annotations(anno_file: str):
        try:
            with open(anno_file, "r") as f:
                annotation: dict = json.load(f)
            annos = annotation.get("shapes", [])
            width = annotation.get('imageWidth', -1)
            height = annotation.get('imageHeight', -1)
        except:
            print("illegal annotation:{}".format(anno_file))
            annos = []
            width = -1
            height = -1
        return annos, width, height

    @staticmethod
    def get_match_targets(data_info: dict, targets, keys=['points', 'boxes', 'labels', 'groups', 'names', 'keypoints']):
        names = data_info["names"]
        out = {}
        for i in range(len(names)):
            matches = text_utils.find_match_texts(texts=[names[i]], pattern=targets, org=True)
            if len(matches) > 0:
                for k in keys:
                    if k in data_info: out[k] = out.get(k, []) + [data_info[k][i]]
        return out

    @staticmethod
    def get_sub2superclass(data_info, superclass, subclass, scale=[], square=True, vis=False):
        """
        通过IOU的方式，将子类属性分配给父类中
        :param data_info:
        :param superclass: 父类
        :param subclass: 子类
        :param scale: 对superclass进行缩放
        :param square: 对superclass
        :param vis:
        :return:
        """
        image, file = data_info["image"], data_info["image_file"]
        target_info = LabelMeDataset.get_match_targets(data_info, targets=superclass)
        attibu_info = LabelMeDataset.get_match_targets(data_info, targets=subclass)
        item_list = []
        for i in range(len(target_info.get("boxes", []))):
            tbbox = [target_info["boxes"][i]]
            tname = [target_info["names"][i]]
            if square:
                tbbox = image_utils.get_square_boxes(tbbox, use_max=True, baseline=-1)
            if scale:
                tbbox = image_utils.extend_xyxy(tbbox, scale=scale)
            item = {}  #
            for j in range(len(attibu_info.get("boxes", []))):
                abox = attibu_info["boxes"][j]
                iou = image_utils.get_box_iom(tbbox[0], abox)
                if iou > 0:
                    item["ious"] = item.get("ious", []) + [iou]
                    for k, v in attibu_info.items():
                        item[k] = item.get(k, []) + [v[j]]
            item_list.append(dict(file=file, box=tbbox[0], name=tname[0], attribute=item))
            if vis:
                image = image_utils.draw_image_boxes_texts(image, tbbox, tname, thickness=2, fontScale=1.0,
                                                           drawType="ch", color=image_utils.color_table[i + 1])
                texts = [f"{n} {s:3.2f}" for n, s in zip(item.get("names", []), item.get("ious", []))]
                image = image_utils.draw_image_boxes_texts(image,
                                                           item.get("boxes", []),
                                                           texts,
                                                           thickness=2,
                                                           fontScale=1.0,
                                                           color=image_utils.color_table[i + 1],
                                                           drawType="ch")
                image_utils.cv_show_image("instance", image)
        return item_list


def LabelMeDatasets(filename=None,
                    data_root=None,
                    anno_dir=None,
                    image_dir=None,
                    class_name=None,
                    use_rgb=False,
                    shuffle=False,
                    check=False,
                    min_points=-1,
                    **kwargs):
    """
    :param filename:
    :param data_root:
    :param anno_dir:
    :param image_dir:
    :param class_name:
    :param use_rgb:
    :param shuffle:
    :param check:
    :param min_points:
    :param kwargs:
    :return:
    """
    if data_root and isinstance(data_root, str): data_root = [data_root]
    if image_dir and isinstance(image_dir, str): image_dir = [image_dir]
    if anno_dir and isinstance(anno_dir, str): anno_dir = [anno_dir]
    n = max([len(n) for n in [data_root, image_dir, anno_dir] if n])
    if data_root is None: data_root = [None] * n
    if image_dir is None: image_dir = [None] * n
    if anno_dir is None: anno_dir = [None] * n
    datasets = []
    for image, anno, root in zip(image_dir, anno_dir, data_root):
        data = LabelMeDataset(filename=None,
                              data_root=root,
                              anno_dir=anno,
                              image_dir=image,
                              class_name=class_name,
                              use_rgb=use_rgb,
                              shuffle=shuffle,
                              check=check,
                              min_points=min_points,
                              **kwargs)
        datasets.append(data)
    datasets = ConcatDataset(datasets, shuffle=shuffle)
    return datasets


def parser_labelme(anno_file, class_dict={}, shape=None):
    """
    :param annotation:  labelme标注的数据
    :param class_dict:  label映射
    :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
    :return:
    """
    annotation, width, height = LabelMeDataset.load_annotations(anno_file)
    info = LabelMeDataset.parser_annotation(annotation, class_dict, shape)
    return info


def draw_keypoints_image(image, boxes=[], keypoints=[], thickness=1, vis_id=False):
    """绘制keypoints"""
    h, w = image.shape[:2]
    if len(keypoints) == 0: return image
    if len(boxes) == 0: boxes = [(0, 0, w, h)] * len(keypoints)
    from pybaseutils.pose import bones_utils
    target_bones = bones_utils.get_target_bones("coco_person")
    image = image_utils.draw_key_point_in_image(image, keypoints, pointline=target_bones["skeleton"],
                                                colors=target_bones["colors"], thickness=thickness,
                                                boxes=boxes, vis_id=vis_id)

    if vis_id:
        for kpts in keypoints:
            if len(kpts) == 0: continue
            print(kpts)
            kpts = np.asarray(kpts)
            point = kpts[:, 0:2]
            texts = [f"{t:3.2f}" for t in kpts[:, 2]]
            image = image_utils.draw_texts(image, points=point, texts=texts, fontScale=0.8, thickness=2)
    return image


def show_target_image(image, boxes, labels, points, keypoints=[], color=(), thickness=2):
    # image = image_utils.draw_image_bboxes_text(image, boxes, labels, color=(255, 0, 0),
    #                                            thickness=2, fontScale=1.2, drawType="chinese")
    image = image_utils.draw_image_contours(image, points, labels, color=color, thickness=thickness)
    image = draw_keypoints_image(image, boxes, keypoints, thickness=thickness, vis_id=True)
    image_utils.cv_show_image("det", image)
    return image


if __name__ == "__main__":
    from pybaseutils.converter import build_labelme

    anno_dir = "/home/PKing/Downloads/sample/images"
    # anno_dir = "/home/PKing/Downloads/冲击试验/sample/images"
    names = None
    dataset = LabelMeDatasets(filename=None,
                              data_root=None,
                              anno_dir=anno_dir,
                              image_dir=None,
                              class_name=names,
                              check=False,
                              phase="val",
                              shuffle=True)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, points, boxes, labels = data["image"], data["points"], data["boxes"], data["labels"]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        anno_file = os.path.join("masker", "{}.json".format(os.path.basename(image_file).split(".")[0]))
        show_target_image(image, boxes, labels, points, keypoints=data["keypoints"])
