#!/usr/bin/env python
# coding: UTF-8 
# author: Cmoon
# date: 2023/4/8 下午4:44

from ultralytics import YOLO
from pathlib import Path
import argparse


class Trainer:
    def __init__(self, pt: str, yaml: str = None):
        self.pt = pt
        self.yaml = yaml
        if self.yaml is not None:
            self.model = YOLO(str(self.yaml)).load(str(self.pt))
        else:
            self.model = YOLO(str(self.pt))

    def train(self, data, epochs: int = 100, batch: int = 16, imgsz: int = 640, resume: bool = False):
        self.model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz, resume=resume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=Path(Path.cwd(), "weights", 'yolov8s.pt'),
                        help='.pt model path')
    parser.add_argument('--yaml', type=str, default=None, help='weights yaml path')
    parser.add_argument('--data', type=str, default='coco.yaml', help='data yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='inference size (pixels)')
    parser.add_argument('--batch', type=int, default=16, help='inference size (pixels)')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--pretrained', type=bool, default=True, help='use track')
    parser.add_argument('--resume', action='store_true', help='use track')

    opt = parser.parse_args()
    trainer = Trainer(pt=opt.weights, yaml=opt.yaml)
    trainer.train(data=opt.data, epochs=opt.epochs, batch=opt.batch, imgsz=opt.imgsz, resume=opt.resume)


if __name__ == '__main__':
    main()
