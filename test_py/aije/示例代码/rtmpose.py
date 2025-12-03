# -*- coding: utf-8 -*-
from mmpose.apis import MMPoseInferencer


class PoseEstimator:
    def __init__(self, model_file):
        self.model = MMPoseInferencer(model=model_file)

    def predict(self, image_file):
        # 对单张图像进行推理
        result = self.model(image_file, show=False)
        return result


if __name__ == '__main__':
    model_file = "weights/rtmpose-tiny.pth"
    image_file = "images/image1.jpg"
    infer = PoseEstimator(model_file)
    result = infer.predict(image_file)
