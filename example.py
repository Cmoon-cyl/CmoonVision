#!/usr/bin/env python
# coding: UTF-8 
# author: Cmoon
# date: 2023/3/29 下午4:04

from CmoonVision import CmoonVision
from pathlib import Path


def main():
    weights = Path(Path.cwd(), 'weights', 'yolov8s.pt')
    detector = CmoonVision(task='detect', weights=weights)
    results = detector.predict(source=0)
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
