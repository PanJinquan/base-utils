import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_colormap(data_type="custom"):
    if data_type == 'pascal':
        num_classes = 21
        colormap = get_pascal_colormap()
    elif data_type == 'coco':
        num_classes = 21
        colormap = get_coco_colormap()
    elif data_type == 'custom':
        colormap = get_custom_colormap()
        num_classes = len(colormap)
    elif data_type == 'cityscapes':
        num_classes = 19
        colormap = get_cityscapes_colormap()
    else:
        raise NotImplementedError
    return colormap, num_classes


def decode_seg_map_sequence(label_masks, data_type='custom'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, data_type)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, data_type="custom", plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    colormap, n_classes = get_colormap(data_type)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = colormap[ll, 0]
        g[label_mask == ll] = colormap[ll, 1]
        b[label_mask == ll] = colormap[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    return rgb


def decode_color_mask(mask, data_type="custom", plot=False) -> np.ndarray:
    """Decode segmentation class labels into a color image
    Args:
        mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    colormap, n_classes = get_colormap(data_type)
    omask = mask.copy()
    if len(omask.shape) == 2:
        omask = cv2.cvtColor(omask, cv2.COLOR_GRAY2BGR)
    m = cv2.cvtColor(omask, cv2.COLOR_BGR2GRAY)
    for ll in range(0, n_classes):
        omask[m == ll] = colormap[ll]
    if plot:
        plt.imshow(omask)
        plt.show()
    return omask


def decode_color_image_mask(image, mask, data_type='custom'):
    """
    :param image: BGR原始图像
    :param mask: 对应的Mask
    :param data_type: 数据类型
    :return:
    """
    color_mask = decode_color_mask(mask, data_type=data_type, plot=False)
    alpha = mask.copy()
    alpha[mask > 0] = 255
    if len(alpha.shape) == 2:
        # alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        alpha = alpha[:, :, np.newaxis]
    alpha = np.asarray(alpha / 255.0, dtype=np.float32)
    color_image = np.asarray(image / 2 + color_mask / 2, dtype=np.uint8)
    color_image = color_image * alpha + image * (1 - alpha)
    color_image = np.asarray(np.clip(color_image, 0, 255), dtype=np.uint8)
    return color_image, color_mask


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_colormap()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_colormap():
    return np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                     [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                     [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


def get_pascal_colormap():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


get_coco_colormap = get_pascal_colormap


def get_custom_colormap():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    color_map = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (128, 0, 0), (0, 128, 0), (128, 128, 0),
                 (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                 (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                 (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)] * 10
    return np.asarray(color_map)
