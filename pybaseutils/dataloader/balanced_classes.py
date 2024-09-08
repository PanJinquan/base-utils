# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-18 14:08:46
"""
import numpy as np
import math


def get_torch_sample(weights,
                     num_samples: int,
                     replacement: bool = True,
                     generator=None):
    '''
    https://blog.csdn.net/caihuanqia/article/details/113258690
    :param weights:weights参数对应的是“样本”的权重而不是“类别的权重”,权重越大，采样次数更多
    :param num_samples:
    :param replacement:
    :param generator:
    :return:
    '''
    import torch.utils.data as torch_utils
    sampler = torch_utils.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler


def class_weight_to_sample_weight(labels_list: list, class_weight: dict):
    '''
    :param labels_list:lable必须从0开始的,连续的int类型
    :param class_weight:
    :return:
    '''
    sample_weight = [0] * len(labels_list)
    for idx, name in enumerate(labels_list):
        sample_weight[idx] = class_weight[name]
    return sample_weight


def count_class_samples_nums(labels_list):
    '''
    classes_dict = {cls_id0: nums_of_id0,cls_id1: nums_of_id1,...,}
    classes_dict = {0: 5,1: 5, 2: 2, 3: 2, 4: 4}
    =========
    # nclasses = len(set(labels_list)) # fix a BUG
    nclasses = max(labels_list) + 1
    count = [0] * nclasses
    for name in labels_list:
        count[name] += 1  # item is (img-data, label-id)
    classes_dict = dict(enumerate(count))
    =========
    :param labels_list:lable必须从0开始的,连续的int类型
    :return:
    '''
    count_class = {}
    for name in labels_list:
        try:
            count_class[name] += 1
        except Exception as e:
            count_class[name] = 1
    return count_class


def create_class_sample_weight_sklearn(labels_list: list, balanced='balanced', weight_type="class_weight"):
    '''
    balanced : dict, 'balanced' or None
    If 'balanced', class weights will be given by
    ``n_samples / (n_classes * np.bincount(lt_steps))``.
    If a dictionary is given, keys are classes and values are corresponding class weights.
    If None
    :param labels_list: lable必须从0开始的,连续的int类型
    :param balanced:dict, 'balanced' or None
            If 'balanced', class weights will be given by
            ``n_samples / (n_classes * np.bincount(lt_steps))``.
            If a dictionary is given, keys are classes and values
            are corresponding class weights.
            If None is given, the class weights will be uniform.
    :param weight_type: class_weight or sample_weight
    :return:
    '''
    import sklearn
    classes = np.unique(labels_list)
    weight_list = sklearn.utils.class_weight.compute_class_weight(balanced,
                                                                  classes,
                                                                  labels_list)
    if weight_type == "class_weight":
        class_weight = dict(zip([x for x in classes], weight_list))
        return class_weight
    elif weight_type == "sample_weight":
        class_weight = dict(zip([x for x in classes], weight_list))
        sample_weight = class_weight_to_sample_weight(labels_list, class_weight)
        return sample_weight
    else:
        return weight_list


def create_class_sample_weight_custom(labels_list, balanced="balanced", weight_type="class_weight"):
    '''

    :param labels_list:lable必须从0开始的,连续的int类型
    :param balanced:
    :param weight_type: class_weight:返回每个classs的权重
                        sample_weight:返回每个labels_list对应的权重
    :param mu:
    :return:
    '''
    count_class = count_class_samples_nums(labels_list)
    n_samples = np.sum(list(count_class.values()))
    classes = count_class.keys()
    n_classes = len(classes)
    class_weight = dict()
    weight = sum(class_weight.values())
    # 计算每个类别的权重：样本越少，权重越大
    for cls in classes:
        cls_num = float(count_class[cls])
        if balanced == "log_balanced":
            mu = 0.15
            score = math.log(mu * n_samples / cls_num)
            class_weight[cls] = score if score > 1.0 else 1.0
        elif balanced == "balanced":
            # score = n_samples / (n_classes * np.bincount(lt_steps))
            score = n_samples / (n_classes * cls_num)
            class_weight[cls] = score
        elif balanced == "auto":
            # N / float(count[i])
            score = n_samples / cls_num
            class_weight[cls] = score
        else:
            raise Exception("Error:{}".format(balanced))
    # loss_weight = {k: v / weight for k, v in class_weight.items()}
    if weight_type == "class_weight":
        return class_weight
    elif weight_type == "sample_weight":
        sample_weight = class_weight_to_sample_weight(labels_list, class_weight)
        return sample_weight
    else:
        raise Exception("Error:{}".format(weight_type))


def create_sample_weight_torch(labels_list, nclasses=None):
    '''
    Make a vector of weights for each image in the dataset, based
    on class frequency. The returned vector of weights can be used
    to create a WeightedRandomSampler for a DataLoader to have
    class balancing when sampling for a training batch.
        images - torchvisionDataset.imgs
        nclasses - len(torchvisionDataset.classes)
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    labels_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    weight       =[3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 9.0, 9.0, 9.0, 9.0, 4.5, 4.5, 4.5, 4.5]
    '''
    if not nclasses:
        nclasses = len(set(labels_list))
    classes_dict = count_class_samples_nums(labels_list)
    count = list(classes_dict.values())
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels_list)
    for idx, name in enumerate(labels_list):
        weight[idx] = weight_per_class[name]

    return weight


def keras_example(class_weight, sample_weight):
    '''
    在keras,fit函数中调用class_weight，可以通过字典设置每个类别输入权重，比如：cw = {0: 1, 1: 25}，
    类别序列可以使用.class_indices获取。
    :param model:
    :param class_weight: Optional dictionary mapping class indices (integers)
        to a weight (float) value, used for weighting the loss function
        (during training only).
        This can be useful to tell the model to
        "pay more attention" to samples from
        an under-represented class.
    :param sample_weight: type->numpy array，用于在训练时调整损失函数（仅用于训练）。
    可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，
    传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。
    这种情况下请确定在compile时添加了sample_weight_mode=‘temporal’。
        1.sample_weight会覆盖class_weight，所以二者用其一；
        2.如果仅仅是类不平衡，则使用class_weight，sample_weights则是类内样本之间还不平衡的时候使用。
        3.keras已经在新版本中加入了 class_weight = ‘auto’。
        设置了这个参数后，keras会自动设置class weight让每类的sample对损失的贡献相等。
        4.在设置类别权重的时候，类别序列可以使用train_generator.class_indices获取。
    :return:
    '''
    import tensorflow as tf
    train_dataset = ...
    steps_per_epoch = ...
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=None, name="fc1")
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              sample_weight=sample_weight,
              class_weight='auto')


if __name__ == "__main__":
    # random labels_dict
    # labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}
    labels_list = [4, 1, 2, 3, 2, 3, 3, 4, 4, 4, 0]
    print("create_class_sample_weight_custom-------------------------")
    w1 = create_class_sample_weight_custom(labels_list, balanced="balanced", weight_type="class_weight")
    w2 = create_class_sample_weight_custom(labels_list, balanced="balanced", weight_type="sample_weight")
    print("class_weight :{}".format(w1))
    print("labels_list  :{}".format(labels_list))
    print("sample_weight:{}".format(w2))
    print("create_class_sample_weight_sklearn-------------------------")
    w2 = create_class_sample_weight_sklearn(labels_list, balanced="balanced", weight_type="class_weight")
    w3 = create_class_sample_weight_sklearn(labels_list, balanced="balanced", weight_type="sample_weight")
    print("class_weight :{}".format(w2))
    print("labels_list  :{}".format(labels_list))
    print("sample_weight:{}".format(w3))
    # print(class_weight_to_sample_weight(labels_list, w2))
    # w3 = create_sample_weight_torch(labels_list, nclasses=5)
    # print(w3)
