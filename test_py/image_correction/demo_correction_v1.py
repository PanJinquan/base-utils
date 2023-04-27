# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-24 22:13:25
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import geometry_tools, image_utils, file_utils


class ImageCorrection(object):
    """图像矫正程序"""

    @staticmethod
    def get_hough_lines(img: np.ndarray, rho=1, theta=np.pi / 180, threshold=100, max_angle=35, max_lines=50,
                        thickness=2, vis=False):
        """
        参考：https://blog.csdn.net/on2way/article/details/47028969
        :param img: 输入图像
        :param rho: 线段以像素为单位的距离精度,double类型的,推荐用1.0
        :param theta: 线段以弧度为单位的角度精度,推荐用numpy.pi/180
        :param threshold: : 累加平面的阈值参数,int类型,超过设定阈值才被检测出线段,
                            值越大,意味着检出的线段越长,检出的线段个数越少。根据情况推荐先用100试试
        :return:
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
        else:
            gray = img.copy()
        gray = image_utils.get_image_mask(gray, inv=True)
        edge = cv2.Canny(gray, threshold1=0, threshold2=255, apertureSize=3)
        # lines is (num_lines,1,2)==>(r,θ)==>(距离rho,角度theta)
        lines = cv2.HoughLines(edge, rho=rho, theta=theta, threshold=threshold)
        lines = [] if lines is None else lines[:, 0, :]
        lines = lines[0:min(len(lines), max_lines)]
        angles = []
        for i in range(len(lines)):
            rho, theta = lines[i]  # 其中theta是与Y轴的夹角
            angle = 90 - theta * (180 / np.pi)
            # print(rho, theta, angle)
            if abs(angle) < max_angle:  # 水平直线
                angles.append(angle)
                if vis:
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (edge.shape[1], int((rho - edge.shape[1] * np.cos(theta)) / np.sin(theta)))
                    # 绘制一条直线
                    cv2.line(img, pt1, pt2, (0, 0, 255), thickness=thickness)
            else:  # 垂直直线
                if vis:
                    #  (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)) 垂直直线(<45°,>135°)
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - edge.shape[0] * np.sin(theta)) / np.cos(theta)), edge.shape[0])
                    # 绘制一条白线
                    cv2.line(img, pt1, pt2, (255, 0, 0), thickness=thickness)
        angle = 0 if len(angles) < 1 else ImageCorrection.get_lines_mean_angle(angles)
        return angle, img

    @staticmethod
    def get_hough_lines_p(img: np.ndarray, rho=1, theta=np.pi / 180, threshold=100, max_angle=45,
                          max_lines=200, minLineLength=100, maxLineGap=10, thickness=2, vis=False):
        """
        https://blog.csdn.net/on2way/article/details/47028969
        :param img: 输入图像
        :param rho: 线段以像素为单位的距离精度,double类型的,推荐用1.0
        :param theta: 线段以弧度为单位的角度精度,推荐用numpy.pi/180
        :param threshold: : 累加平面的阈值参数,int类型,超过设定阈值才被检测出线段,
                            值越大,意味着检出的线段越长,检出的线段个数越少。根据情况推荐先用100试试
        :param  minLineLength 用来控制「接受直线的最小长度」的值，默认值为 0。
        :param  maxLineGap 用来控制接受共线线段之间的最小间隔，即在一条线中两点的最大间隔。
        :return:
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
        else:
            gray = img.copy()
        gray = image_utils.get_image_mask(gray, inv=True)
        edge = cv2.Canny(gray, threshold1=0, threshold2=255, apertureSize=3)
        lines = cv2.HoughLinesP(edge, rho=rho, theta=theta,
                                threshold=threshold,
                                minLineLength=minLineLength,
                                maxLineGap=maxLineGap)
        lines = [] if lines is None else lines[:, 0, :]
        lines = lines[0:min(len(lines), max_lines)]
        angles = []
        for x1, y1, x2, y2 in lines[:]:
            pt1, pt2 = (x1, y1), (x2, y2)  # P12 = point2-point1
            angle = geometry_tools.compute_horizontal_angle(pt1, pt2, minangle=False)
            # print(pt1, pt2, angle)
            if abs(angle) < max_angle:  # 水平直线
                angles.append(angle)
                if vis: cv2.line(img, pt1, pt2, color=(0, 0, 255), thickness=thickness)
            else:  # 垂直直线
                if vis: cv2.line(img, pt1, pt2, color=(255, 0, 0), thickness=thickness)
        angle = 0 if len(angles) < 1 else ImageCorrection.get_lines_mean_angle(angles)
        return angle, img

    @staticmethod
    def get_lines_mean_angle(angles):
        """求直线簇的平均角度"""
        angles = sorted(angles)
        r = len(angles) // 2
        ar = (r - r // 2, r + r // 2 + 1)
        angles = angles[ar[0]:ar[1]]
        angle = np.mean(angles)
        return angle

    @staticmethod
    def rotation(image, angle):
        """实现图像旋转"""
        h, w = image.shape[:2]
        center = (w / 2., h / 2.)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        return image

    @staticmethod
    def correct(image, max_angle=35, vis=False):
        """
        图像矫正
        :param image: 输入RGB或BGR图像
        :param max_angle: 图像最大的倾斜角度,超过该角度的无法矫正,默认不超过35°
        :param vis: 是否可视化图像矫正结果
        :return: image返回矫正后的图像
        :return: angle返回原始图像倾斜的角度
        """
        # angle, image_line = ImageCorrection.get_hough_lines(image, max_angle=max_angle,vis=vis)
        angle, image_line = ImageCorrection.get_hough_lines_p(image, max_angle=max_angle, vis=vis)
        image = ImageCorrection.rotation(image, angle=-angle)  # 9ms
        if vis:
            print(angle)
            image_line = image_utils.resize_image(image_line, size=(None, image.shape[0]))
            image_line = np.hstack((image_line, image))
            image_utils.cv_show_image("Origin-Alignment", image_line, delay=0, use_rgb=False)
        return image, angle


def image_correction_demo(image_dir):
    """
    :param image_dir:
    :return:
    """
    image_list = file_utils.get_files_lists(image_dir)
    alignment = ImageCorrection()
    for image_file in image_list:
        print(image_file)
        image = cv2.imread(image_file)
        image, angle = alignment.correct(image, vis=True)
        print("倾斜角度：{}".format(angle))
        print("--" * 10)


if __name__ == "__main__":
    image_dir = "data/image1"  # 测试图片
    image_correction_demo(image_dir)
