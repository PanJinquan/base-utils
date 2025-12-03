# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
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
                 kpt_names=None,
                 kpt_shape=None,
                 use_kpt=False,
                 use_rgb=False,
                 shuffle=False,
                 check=False,
                 check_kpt=True,
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
        :param check_kpt: True会检查关键点的完整性,图像中仅当所有目标都标注和对应的关键点(无漏标注),才返回数据
                           False不会检查关键点的完整性,,图像中只要标注的目标和对应的关键点(漏标注),会返回数据
        :param min_points: 当标注的轮廓点的个数小于min_points，会被剔除；负数不剔除
        :param kwargs: read_image: 是否读取图片，否则image=None
        """
        self.tag = self.__class__.__name__
        super(LabelMeDataset, self).__init__()
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.use_kpt = use_kpt
        self.min_points = min_points
        self.check_kpt = check_kpt
        self.kwargs = kwargs
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.class_name, self.class_dict = self.parser_classes(class_name)
        self.kpt_names, self.kpt_dict, self.kpt_shape, self.total_names = self.parser_kpt_names(kpt_names, kpt_shape)
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
        self.log("{:15s} data_root     :{}".format(self.tag, self.data_root))
        self.log("{:15s} anno_dir      :{}".format(self.tag, self.anno_dir))
        self.log("{:15s} image_dir     :{}".format(self.tag, self.image_dir))
        self.log("{:15s} class_name    :{}".format(self.tag, self.class_name))
        self.log("{:15s} class_dict    :{}".format(self.tag, self.class_dict))
        self.log("{:15s} kpt_info      :shape={},name={}".format(self.tag, self.kpt_shape, self.kpt_dict))
        self.log("{:15s} num images    :{}".format(self.tag, len(self.image_ids)))
        # self.log("{:15s} num_classes   :{}".format(self.tag,self.num_classes))
        self.log("------" * 10)

    def parser_kpt_names(self, kpt_names, kpt_shape):
        """
        v=0未标注点; v=1标注了但是图像中不可见（例如遮挡）;v=2标注了并图像可见
        :param kpt_names: 关键点名称列表，
                          当kpt_names.ndim=1时，要求所有类别且所有目标的关键点名称必须一致
                          当kpt_names.ndim=2时，仅要求相同类别目标的关键点名称必须一致，且len(kpt_names)==len(self.class_name)
        :param kpt_shape: 关键点的维度，默认是(17, 3)，对于coco-person,有17个关键点，3表示(x,y,v)
        :return:
        """
        if not self.use_kpt: return [], {}, tuple(), self.class_dict
        if kpt_names and not kpt_shape: kpt_shape = (len(kpt_names), 3)
        if not kpt_shape: kpt_shape = (17, 3)
        if np.asarray(kpt_names).ndim == 1:
            kpt_names = [kpt_names] * len(self.class_name)
        kpt_dict, total_names = {}, self.class_dict.copy()
        for cls, name in enumerate(kpt_names):
            kpt_dict.update({f"{self.class_name[cls]}#{n}": i for i, n in enumerate(name)})
            total_names.update({n: len(self.class_dict) + cls + i for i, n in enumerate(name)})
        for i, names in enumerate(kpt_names):
            assert len(names) == kpt_shape[0], (f"维度不一致,"
                                                f"class={self.class_name[i]},kpt_names={names},kpt_shape={kpt_shape}")
        return kpt_names, kpt_dict, kpt_shape, total_names

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
        image_id, img_postfix = file_utils.split_postfix(image_file)
        anno_file = os.path.join(anno_dir, "{}.json".format(image_id))
        return image_file, anno_file, image_name

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        dst_ids = []
        class_name = []
        for image_id in tqdm(image_ids, desc="check data"):
            image_file, anno_file, image_id = self.get_image_anno_file(image_id)
            if not os.path.exists(anno_file):
                continue
            if not os.path.exists(image_file):
                continue
            annotation, width, height = self.load_annotations(anno_file)
            data_info = self.parser_annotation(annotation, self.total_names, size=(width, height),
                                               min_points=self.min_points,
                                               unique=self.unique)
            if self.use_kpt:
                data_info = self.get_kpts_info(data_info, anno_file=anno_file, check_kpt=self.check_kpt, disp=True)
            labels = data_info["labels"]
            if len(labels) == 0:
                continue
            dst_ids.append(image_id)
            class_name += labels
        if self.class_name is None:
            class_name = sorted(list(set(class_name)))
            self.class_name, self.class_dict = self.parser_classes(class_name)
        self.log("{:15s} have nums image:{},legal image:{}".format(self.tag, len(image_ids), len(dst_ids)))
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
            image_ids = self.get_file_list(image_dir, postfix=file_utils.IMG_POSTFIX, sub=True, basename=False)
            if not anno_dir: anno_dir = image_dir
        elif anno_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=file_utils.IMG_POSTFIX, sub=True, basename=False)
            if not image_dir: image_dir = anno_dir
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
            height, width = image.shape[:2]
            size = (width, height)
        else:
            image, size = None, (width, height)
        data_info = self.parser_annotation(annotation, self.total_names, size=size, min_points=self.min_points,
                                           unique=self.unique)
        if self.use_kpt:
            data_info = self.get_kpts_info(data_info, anno_file=anno_file, check_kpt=self.check_kpt)
        # TODO dict(boxes, labels, points, groups, names, keypoints)
        data_info.update({"image": image, "image_file": image_file, "anno_file": anno_file,
                          "size": tuple(size)})
        return data_info

    def get_kpts_info(self, data_info, anno_file="", check_kpt=False, disp=False):
        """
        获得目标和关键点信息
        :param data_info:
        :param anno_file:
        :param check_kpt: True会检查关键点的完整性,图像中仅当所有目标都标注和对应的关键点(无漏标注),才返回数据
                           False不会检查关键点的完整性,,图像中只要标注的目标和对应的关键点(漏标注),会返回数据
        :param disp:
        :return:
        """
        keys = list(data_info.keys())  # keys = ['boxes', 'labels', 'points', 'groups', 'names', 'keypoints']
        groups = data_info["groups"]
        objects = {}
        for i, gid in enumerate(groups):
            info = objects.get(gid, {})
            for key in keys:
                info[key] = info[key] + [data_info[key][i]] if key in info else [data_info[key][i]]
            info["keypoints"] = []
            objects[gid] = info
        out_info = {n: [] for n in keys}
        for gid, info in objects.items():
            c_index = {i: n for i, n in enumerate(info["names"]) if n in self.class_dict}  # 实例index
            if len(c_index) != 1: continue  # TODO 同一组仅有一个实例框
            c_name = list(c_index.values())[0]
            k_index = {i: f"{c_name}#{n}" for i, n in enumerate(info["names"]) if
                       f"{c_name}#{n}" in self.kpt_dict}  # 关键点index
            if not k_index: continue  # 如果目标框存在,但关键点不存在
            for key in keys:
                if key == "keypoints":
                    kpts = np.zeros(shape=tuple(self.kpt_shape), dtype=np.float32)
                    poin = {n: info['points'][i] for i, n in k_index.items()}
                    poin = {self.kpt_dict[n]: v for n, v in poin.items()}
                    k = np.array(list(poin.keys()), dtype=np.int32)
                    p = np.array(list(poin.values()), dtype=np.float32).reshape(-1, 2)
                    if self.kpt_shape[1] == 3:
                        m = np.where((p[..., 0] < 0) | (p[..., 1] < 0), 0.0, 2.0).astype(np.float32)
                        p = np.concatenate([p, m[..., None]], axis=-1)  # (nl, nkpt, 3)
                    kpts[k] = p
                    data = [kpts]
                else:
                    data = [info[key][i] for i, n in c_index.items()]
                out_info[key] = out_info[key] + data if key in out_info else data
        # 如果存在目标没有标注关键点,则将该目标的所有信息设置为空
        c_names = [n for n in data_info['names'] if n in self.class_dict]  # 有效实例
        valid_kpts = [np.sum(kpts) for kpts in out_info["keypoints"]]
        if any(v < 1 for v in valid_kpts) or (check_kpt and len(valid_kpts) != len(c_names)):
            if disp: print("标注文件存在错误:{}".format(anno_file))
            out_info = {n: [] for n in keys}
        return out_info

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

    @staticmethod
    def parser_annotation(annotation: dict, class_dict={}, size=(), min_points=-1, unique=False):
        """
        :param annotation:  labelme标注的数据
        :param class_dict:  label映射,如{"person":0,"car":1}
        :param size: 图片(W,H),可进行坐标点的维度检查，避免越界
        :param min_points: 当标注的轮廓点的个数小于等于min_points，会被剔除；负数不剔除
        :return:
        """
        bboxes, labels, points, groups, names, keypoints = [], [], [], [], [], []
        gid_index = 100000  # TODO bug 若gid_index=0,当某个实例未标注group_id,会导致关键点分组异常
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
            gid = anno.get("group_id", gid_index) or gid_index
            kpt = anno.get("keypoints", [])
            if size:
                pts[:, 0] = np.clip(pts[:, 0], 0, size[0] - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, size[1] - 1)
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

    def get_keypoint_object(self, annotation: list, w, h, class_name=[], kpt_names=[]):
        """
        获得labelme关键点检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :return:
        """
        if not kpt_names: kpt_names = self.kpt_names
        objects = {}
        gid_index = 100000  # TODO bug 若gid_index=0,当某个实例未标注group_id,会导致关键点分组异常
        for i, anno in enumerate(annotation):
            label = anno["label"]
            pts = np.asarray(anno["points"], dtype=np.int32)
            gid = anno.get("group_id", gid_index) or gid_index
            if label in kpt_names:
                keypoints: dict = json_utils.get_value(objects, [gid, "keypoints"], default={})
                keypoints.update({label: pts.tolist()[0]})
                objects = json_utils.set_value(objects, key=[gid, "keypoints"], value=keypoints)
            elif label in class_name:
                contours = pts
                contours[:, 0] = np.clip(contours[:, 0], 0, w - 1)
                contours[:, 1] = np.clip(contours[:, 1], 0, h - 1)
                boxes = image_utils.polygons2boxes([contours])
                if gid in objects:
                    objects[gid].update({"labels": label, "boxes": boxes[0], "segs": contours})
                else:
                    objects[gid] = {"labels": label, "boxes": boxes[0], "segs": contours}
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
            pts = np.asarray(anno["points"], dtype=np.int32)
            gid = i
            if class_name is None or len(class_name) == 0 or label in class_name:
                segs = pts
                segs[:, 0] = np.clip(segs[:, 0], 0, w - 1)
                segs[:, 1] = np.clip(segs[:, 1], 0, h - 1)
                box = image_utils.polygons2boxes([segs])[0]
                objects = json_utils.set_value(objects, key=[gid],
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
        except Exception as e:
            # print(e,"illegal annotation:{}".format(anno_file))
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
    datasets = ConcatDataset(datasets, shuffle=shuffle, **kwargs)
    return datasets


def parser_labelme(anno_file, class_dict={}, size=()):
    """
    :param annotation:  labelme标注的数据
    :param class_dict:  label映射
    :param size: 图片shape(W,H),可进行坐标点的维度检查，避免越界
    :return:
    """
    annotation, width, height = LabelMeDataset.load_annotations(anno_file)
    data_info = LabelMeDataset.parser_annotation(annotation, class_dict, size=size)
    return data_info


def draw_keypoints_image(image, boxes=[], kpts=[], bones_type="coco_person", thickness=1, vis_id=False):
    """绘制keypoints"""
    h, w = image.shape[:2]
    if len(kpts) == 0: return image
    if len(boxes) == 0: boxes = [(0, 0, w, h)] * len(kpts)
    from pybaseutils.pose import bones_utils
    bones_info = bones_utils.get_target_bones(bones_type, kpts=kpts)
    image = image_utils.draw_key_point_in_image(image, kpts, pointline=bones_info["skeleton"],
                                                colors=bones_info["colors"], thickness=thickness,
                                                boxes=boxes, vis_id=vis_id)
    return image


def show_target_image(image, boxes, names, points, kpts=[], bones_type="coco_person", color=(), thickness=2):
    # image = image_utils.draw_image_bboxes_text(image, boxes, names, color=(255, 0, 0),
    #                                            thickness=2, fontScale=1.2, drawType="chinese")
    image = image_utils.draw_image_contours(image, points, names, color=color, thickness=thickness)
    image = draw_keypoints_image(image, boxes, kpts, bones_type=bones_type, thickness=thickness, vis_id=True)
    image_utils.cv_show_image("det", image)
    return image


def example_for_keypoints():
    anno_dir = "/home/PKing/nasdata/tmp/tmp/pressure_meter/dataset-v2/val/images"
    class_name = ['pointer', 'range_start', 'range_end']
    kpt_names = [['p0', 'p1', "p2", "p3"], ['p1', 'p2', "p0", "p3"], ['p1', 'p2', "p3", "p0"]]
    # anno_dir = "/home/PKing/nasdata/tmp/tmp/pressure_meter/dataset-v1/val/images"
    # class_name = ['pressure_meter']
    # kpt_names = ['pointer_start', 'pointer_end', 'range_start', 'range_end']
    # class_name = ['person', 'car']
    # class_name = ['car']
    # kpt_names = ["p0", "p1", "p2", "p3", "p4"]
    # class_name = ['person']
    # anno_dir = "../../data/labelme/images"
    kpt_shape = (4, 3)
    # class_name = ['range_start']
    dataset = LabelMeDatasets(filename=None,
                              data_root=None,
                              anno_dir=anno_dir,
                              image_dir=None,
                              class_name=class_name,
                              use_kpt=True,
                              kpt_names=kpt_names,
                              kpt_shape=kpt_shape,
                              check_kpt=False,
                              check=False,
                              phase="val",
                              shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        # i = 3
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, points, boxes, names = data["image"], data["points"], data["boxes"], data["names"]
        image_file = data["image_file"]
        kpts = data["keypoints"]
        print(image_file)
        show_target_image(image, boxes, names, points, bones_type="", kpts=kpts)


def example_for_segment():
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/train-v2/01-核相操作/dataset-v01/images"
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
        image_file = data["image_file"]
        kpts = data["keypoints"]
        print(image_file)
        show_target_image(image, boxes, labels, points, kpts=kpts)


if __name__ == "__main__":
    # example_for_segment()
    example_for_keypoints()
