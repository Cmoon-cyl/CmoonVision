#!/usr/bin/env python
# coding: UTF-8 
# author: Cmoon
# date: 2023/3/27 下午8:50

from pathlib import Path
from typing import *
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from cmoon_utils import DetResult, SegResult, ClsResult, PoseResult, Utils
from abc import ABCMeta, abstractmethod

import pykinect_azure as pykinect


class YoloBase(metaclass=ABCMeta):
    """Yolo抽象基类"""

    def __init__(self, weights: str, imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.7):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = YOLO(str(self.weights))
        self.names = self.model.names

        self.results = []
        self.find = None
        self.noshow = False
        self.roi = [0.0, 1.0, 0.0, 1.0]

    @abstractmethod
    def process_results(self, results):
        raise NotImplementedError

    def judge_results(self, results) -> bool:
        items = {item.name: item.center for item in results}
        size = results[0].img0.shape[1::-1]
        flag = self.find in items and Utils.in_roi(items[self.find], size, self.roi) if self.find else cv2.waitKey(
            1) & 0xFF == ord('q')
        return flag

    @abstractmethod
    def predict(self, source: Any) -> list:
        raise NotImplementedError


# YoloDet 子类
class YoloDet(YoloBase):
    def __init__(self, weights: str = Path(Path.cwd(), "weights", 'yolov8s.pt'), imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.7):
        super().__init__(weights, imgsz, conf_thres, iou_thres)
        self.track = False

    def process(self, results):
        for result in results:
            self.results = []
            img0 = result.orig_img
            size = img0.shape[1::-1]
            if len(result):
                for data in result.boxes.data.tolist():
                    name = result.names[int(data[-1])]
                    box = list(map(int, data[:4]))
                    center = Utils.xyxy2cnt(box)
                    conf = data[-2]
                    id = int(data[-3]) if self.track else None
                    if Utils.in_roi(center, size, self.roi):
                        self.results.append(DetResult(name, box, center, conf, img0, id))
                        if not self.noshow:
                            cv2.circle(img0, center, 3, (0, 0, 255), -1)
                        print(self.results[-1])
                        print('------------------------------------------')
            if not self.noshow:
                img_plot = result.plot()
                if self.roi != [0.0, 1.0, 0.0, 1.0]:
                    _, roi_point = Utils.calc_roi(size, self.roi)
                    cv2.rectangle(img_plot, roi_point[0], roi_point[1], (255, 0, 0), 2)
                Utils.show_stream('result', img_plot)

    def process_results(self, results):
        try:
            for result in results:
                self.results = []
                img0 = result.orig_img
                size = img0.shape[1::-1]
                if len(result):
                    for data in result.boxes.data.tolist():
                        name = result.names[int(data[-1])]
                        box = list(map(int, data[:4]))
                        center = Utils.xyxy2cnt(box)
                        conf = data[-2]
                        id = int(data[-3]) if self.track else None
                        if Utils.in_roi(center, size, self.roi):
                            self.results.append(DetResult(name, box, center, conf, img0, id))
                            if not self.noshow:
                                cv2.circle(img0, center, 3, (0, 0, 255), -1)
                            print(self.results[-1])
                            print('------------------------------------------')
                if not self.noshow:
                    img_plot = result.plot()
                    if self.roi != [0.0, 1.0, 0.0, 1.0]:
                        _, roi_point = Utils.calc_roi(size, self.roi)
                        cv2.rectangle(img_plot, roi_point[0], roi_point[1], (255, 0, 0), 2)
                    cv2.imshow('result', img_plot)
                if len(self.results):
                    if self.judge_results(self.results):
                        break
        except KeyboardInterrupt:
            print('Stop Manually!')
        finally:
            cv2.destroyAllWindows()

    def predict(self, source: Any, classes: str = None, find: str = None, roi: Sequence = (0.0, 1.0, 0.0, 1.0),
                nosave: bool = False, noshow: bool = False, track: bool = False) -> list:
        """
            模型推理

            :param source: 图片来源:cv2读取的图片(np.ndarray)/电脑摄像头('cam','0',0)/本地文件(str)
            :param classes: 哪些物体可以被检测到,字符串名称间用','分割('bottle,person')
            :param find: 需要寻找的物体名称('bottle')
            :param roi: 画面中心多大范围内的检测结果被采用([0.0,0.5,0.0,1.0])
            :param nosave 不保存结果到runs文件夹
            :param noshow 不显示结果
            :param track 使用track计数
            :return: list 根据任务类型返回对应自定义消息类型
        """
        self.find = find
        self.roi = roi
        self.track = track
        self.noshow = noshow
        if classes is not None:
            classes = [list(self.names.values()).index(name) for name in classes.split(',')]
        if type(source) == np.ndarray:
            if self.track:
                results = self.model.track(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)

        elif source in ['kinect', 'k4a', 'stream']:
            pykinect.initialize_libraries()
            # Modify camera configuration
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            # print(device_config)
            # Start device
            device = pykinect.start_device(config=device_config)
            try:
                while True:
                    # Get capture
                    capture = device.update()

                    # Get the color image from the capture
                    ret, color_image = capture.get_color_image()
                    if not ret:
                        continue
                    if self.track:
                        results = self.model.track(source=color_image, stream=True, imgsz=self.imgsz,
                                                   conf=self.conf_thres,
                                                   iou=self.iou_thres, classes=classes, save=not nosave, show=False)
                    else:
                        results = self.model(source=color_image, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                             iou=self.iou_thres, classes=classes, save=not nosave, show=False)
                    self.process(results)
                    if len(self.results):
                        if self.judge_results(self.results):
                            break
            except KeyboardInterrupt:
                print("Stop Manually")
            finally:
                cv2.destroyAllWindows()
                device.stop_cameras()
                device.close()
                return self.results

        elif source in ['cam', '0', 0]:
            if self.track:
                results = self.model.track(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['avi', 'mp4', 'gif', 'mkv', 'mov', 'webm']:
            if self.track:
                results = self.model.track(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        else:
            raise NotImplementedError('Unsupported Source! ')
        self.process_results(results)
        return self.results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# YoloSeg 子类
class YoloSeg(YoloBase):
    def __init__(self, weights: str = Path(Path.cwd(), "weights", 'yolov8s-seg.pt'), imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.7):
        super().__init__(weights, imgsz, conf_thres, iou_thres)
        self.track = False

    def process_results(self, results):
        try:
            for result in results:
                self.results = []
                img0 = result.orig_img
                size = img0.shape[1::-1]
                if len(result):
                    for data, _mask in zip(result.boxes.data.tolist(), result.masks.data.cpu().numpy()):
                        name = result.names[int(data[-1])]
                        box = list(map(int, data[:4]))
                        center = Utils.xyxy2cnt(box)
                        conf = data[-2]
                        maxValue = _mask.max()
                        _mask = _mask * 255 / maxValue
                        _mask = np.uint8(_mask)
                        _mask = cv2.resize(_mask, [img0.shape[1], img0.shape[0]])
                        ret, mask = cv2.threshold(_mask, 127, 255, cv2.THRESH_BINARY)
                        id = int(data[-3]) if self.track else None
                        if Utils.in_roi(center, size, self.roi):
                            self.results.append(SegResult(name, box, center, conf, img0, mask, id))
                            if not self.noshow:
                                cv2.circle(img0, center, 3, (0, 0, 255), -1)
                            print(self.results[-1])
                            print('------------------------------------------')
                if not self.noshow:
                    img_plot = result.plot()
                    if self.roi != [0.0, 1.0, 0.0, 1.0]:
                        _, roi_point = Utils.calc_roi(size, self.roi)
                        cv2.rectangle(img_plot, roi_point[0], roi_point[1], (255, 0, 0), 2)
                    cv2.imshow('result', img_plot)
                if len(self.results):
                    if self.judge_results(self.results):
                        break
        except KeyboardInterrupt:
            print('Stop Manually!')
        finally:
            cv2.destroyAllWindows()

    def predict(self, source: Any, classes: str = None, find: str = None, roi: Sequence = (0.0, 1.0, 0.0, 1.0),
                nosave: bool = False, noshow: bool = False, track: bool = False) -> list:
        self.find = find
        self.roi = roi
        self.track = track
        self.noshow = noshow
        if classes is not None:
            classes = [list(self.names.values()).index(name) for name in classes.split(',')]
        if type(source) == np.ndarray:
            if self.track:
                results = self.model.track(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source in ['cam', '0', 0]:
            if self.track:
                results = self.model.track(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['avi', 'mp4', 'gif', 'mkv', 'mov', 'webm']:
            if self.track:
                results = self.model.track(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                           iou=self.iou_thres, classes=classes, save=not nosave, show=False)
            else:
                results = self.model(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                     iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        else:
            raise NotImplementedError('Unsupported Source! ')
        self.process_results(results)
        print('Detection Finished')
        return self.results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# YoloCls 子类
class YoloCls(YoloBase):
    def __init__(self, weights: str = Path(Path.cwd(), "weights", 'yolov8s-cls.pt'), imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.7):
        super().__init__(weights, imgsz, conf_thres, iou_thres)

    def judge_results(self, results) -> bool:
        items = [item.name for item in results]
        flag = self.find in items if self.find else cv2.waitKey(1) & 0xFF == ord('q')
        return flag

    def process_results(self, results):
        try:
            for result in results:
                self.results = []
                if result is not None:
                    probs, indices = torch.topk(result.probs, 3)
                    probs = probs.tolist()
                    indices = indices.tolist()
                    names = [result.names[index] for index in indices]
                    img0 = result.orig_img
                    self.results.append(ClsResult(list(zip(names, probs)), img0))
                    print(self.results[-1])
                    print('-------------------------')
                if not self.noshow:
                    img_plot = result.plot()
                    Utils.show_stream('result', img_plot)
                if self.judge_results(self.results):
                    break
        except KeyboardInterrupt:
            print('Stop Manually!')
        finally:
            cv2.destroyAllWindows()

    def predict(self, source: Any, classes: str = None, find: str = None, nosave: bool = False,
                noshow: bool = False) -> list:
        self.find = find
        self.noshow = noshow
        if classes is not None:
            classes = [list(self.names.values()).index(name) for name in classes.split(',')]
        if type(source) == np.ndarray:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source in ['cam', '0', 0]:
            results = self.model(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['avi', 'mp4', 'gif', 'mkv', 'mov', 'webm']:
            results = self.model(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        else:
            raise NotImplementedError('Unsupported Source! ')
        self.process_results(results)
        print('Detection Finished')
        return self.results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class YoloPose(YoloBase):
    def __init__(self, weights: str = Path(Path.cwd(), "weights", 'yolov8n-pose.pt'), imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.7):
        super().__init__(weights, imgsz, conf_thres, iou_thres)

    def process_results(self, results):
        try:
            for result in results:
                self.results = []
                img0 = result.orig_img
                joints = result.keypoints.tolist()
                if len(result):
                    for data in result.boxes.data.tolist():
                        box = list(map(int, data[:4]))
                        self.results.append(PoseResult(box, joints, img0))
                        print(self.results[-1])
                if not self.noshow:
                    img_plot = result.plot()
                    Utils.show_stream('result', img_plot)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print('Stop Manually!')
        finally:
            cv2.destroyAllWindows()

    def predict(self, source: Any, classes: str = None, find: str = None, nosave: bool = False,
                noshow: bool = False) -> list:
        self.find = find
        self.noshow = noshow
        if classes is not None:
            classes = [list(self.names.values()).index(name) for name in classes.split(',')]
        if type(source) == np.ndarray:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source in ['cam', '0', 0]:
            results = self.model(source=0, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['avi', 'mp4', 'gif', 'mkv', 'mov', 'webm']:
            results = self.model(source=source, stream=True, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        elif source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            results = self.model(source=source, stream=False, imgsz=self.imgsz, conf=self.conf_thres,
                                 iou=self.iou_thres, classes=classes, save=not nosave, show=False)
        else:
            raise NotImplementedError('Unsupported Source! ')
        self.process_results(results)
        print('Detection Finished')
        return self.results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class CmoonVision:
    """
        CmoonVision Based on YoloV8

        :param task: 任务类型
        :param weights: .pt模型路径
        :param imgsz: 图片大小
        :param conf: 置信度阈值
        :param iou: iou阈值
    """

    def __init__(self, task, *args, **kwargs):
        model = {'detect': YoloDet, 'segment': YoloSeg, 'classify': YoloCls, 'pose': YoloPose}
        self.task = task
        self.model = model[task](*args, **kwargs)

    def predict(self, source: Any, classes: str = None, find: str = None, roi: Sequence = (0.0, 1.0, 0.0, 1.0),
                nosave: bool = False, noshow: bool = False, track: bool = False, *args, **kwargs):
        if self.task == 'detect':
            return self.model(source, classes, find, roi, nosave, noshow, track)
        elif self.task == 'segment':
            return self.model(source, classes, find, roi, nosave, noshow, track)
        elif self.task == 'classify':
            return self.model(source, classes, find, nosave, noshow)
        elif self.task == 'pose':
            return self.model(source, classes, find, nosave, noshow)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detect', help='task type')
    parser.add_argument('--weights', type=str, default=Path(Path.cwd(), "weights", 'yolov8s.pt'),
                        help='.pt model path')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--source', type=str, default='0', help='image source')
    parser.add_argument('--find', type=str, default=None, help='find object')
    parser.add_argument('--classes', type=str, default=None, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--roi', type=float, nargs='+', default=[0.0, 1.0, 0.0, 1.0], help='roi region x x y y')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--noshow', action='store_true', help='do not show result')
    parser.add_argument('--track', action='store_true', help='use track')
    opt = parser.parse_args()

    detector = CmoonVision(task=opt.task, weights=opt.weights, imgsz=opt.imgsz, conf_thres=opt.conf, iou_thres=opt.iou)
    results = detector.predict(source=opt.source, find=opt.find, classes=opt.classes, roi=opt.roi, nosave=opt.nosave,
                               noshow=opt.noshow, track=opt.track)
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
