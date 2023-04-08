#!/usr/bin/env python
# coding: UTF-8 
# author: Cmoon
# date: 2023/3/11 下午2:20

from typing import *
import numpy as np
import cv2
import time


class Utils:
    @staticmethod
    def timer(f):
        """计时函数运行时间"""

        def timeit(*args, **kwargs):
            print("---------开始计时---------")
            start = time.time()
            ret = f(*args, **kwargs)
            print(f"----运行时间：{(time.time() - start):.5f}秒----")
            return ret

        return timeit

    @staticmethod
    def xyxy2cnt(xyxy: List[int]) -> List[int]:
        """
        xyxy坐标转中心点坐标

        :param xyxy: [x,y,x,y]
        :return: [x,y]
        """
        center = [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
        return center

    @staticmethod
    def show_image(name, img):
        try:
            cv2.imshow(name, img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    @staticmethod
    def show_stream(name, img):
        try:
            cv2.imshow(name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    @staticmethod
    def plot_masks(img0, mask):
        img = img0.copy()
        if type(mask) is list:
            for result in mask:
                msk1 = result.mask.astype(np.bool)
                color = np.array([0, 0, 255], dtype=np.uint8)
                img[msk1] = img[msk1] * 0.7 + color * 0.3

        else:
            msk1 = mask.astype(np.bool)
            color = np.array([0, 0, 255], dtype=np.uint8)
            img[msk1] = img[msk1] * 0.7 + color * 0.3

        return img

    @staticmethod
    def calc_roi(size, roi):
        size = [pix for pix in size for _ in range(2)]
        roi_range = [pixel * propotion for pixel, propotion in zip(size, roi)]
        roi_point = [[int(roi_range[0]), int(roi_range[2])], [int(roi_range[1]), int(roi_range[3])]]
        return roi_range, roi_point

    @staticmethod
    def in_roi(point: List[int], size: List[int], roi: Sequence[float]) -> bool:
        """
        判断物品中心点是否在设定的画面范围内

        :param xs: 一张图像上检测到的物品中心坐标
        :param range: 画面中心多大范围内的检测结果被采用
        :return:bool
        """
        roi_range, _ = Utils.calc_roi(size, roi)
        return roi_range[0] <= point[0] <= roi_range[1] and roi_range[2] <= point[1] <= roi_range[3]


class DetResult:
    """
    目标检测检测结果数据类型

    :param name: 标签名称
    :param box: 矩形框左上右下xyxy坐标
    :param center: 矩形框中心点坐标
    :param conf: 置信度
    :param img0: 原图
    :param id: 第几个物体（track时启用）
    """

    def __init__(self, name: str, box: List[int], center: List[int], conf: float, img0: np.ndarray, id: int = None):
        self.name = name
        self.box = box
        self.center = center
        self.conf = conf
        self.img0 = img0
        self.id = id

    def __str__(self):
        return f'name:{self.name}; box:{self.box}; conf:{self.conf:.2f}; id:{self.id}' if self.id else f'name:{self.name}; box:{self.box}; conf:{self.conf:.2f}'

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class SegResult:
    """
       实例分割检测结果数据类型

       :param name: 标签名称
       :param box: 矩形框左上右下xyxy坐标
       :param center: 矩形框中心点坐标
       :param conf: 置信度
       :param img0: 原图
       :param mask: 分割结果mask
       :param id: 第几个物体（track时启用）
    """

    def __init__(self, name: str, box: List[int], center: List[int], conf: float, img0: np.ndarray,
                 mask: np.ndarray, id: int = None):
        self.name = name
        self.box = box
        self.center = center
        self.conf = conf
        self.img0 = img0
        self.mask = mask
        self.img_mask = Utils.plot_masks(self.img0, self.mask)
        self.id = id

    def __str__(self):
        return f'name:{self.name}; box:{self.box}; conf:{self.conf:.2f}; id:{self.id}' if self.id else f'name:{self.name}; box:{self.box}; conf:{self.conf:.2f}'

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class ClsResult:
    """
       图像分类检测结果数据类型

       :param probs: top5 results[name,conf]
       :param img0: 原图
       """

    def __init__(self, probs: list, img0: np.ndarray):
        self.probs = probs
        self.name = self.probs[0][0]
        self.img0 = img0

    def __str__(self):
        return f'name:{self.name}; top3:{self.probs}'

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class PoseResult:
    """
       图像分类检测结果数据类型

       :param box： 矩形框左上右下xyxy坐标
       :param joints: 关节点
       :param img0: 原图
       """

    def __init__(self, box: List[int], joints: List, img0: np.ndarray):
        self.box = box
        self.joints = joints
        self.img0 = img0

    def __str__(self):
        return f'box：{self.box}'

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
