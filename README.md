# CmoonVision Based on YoloV8

## 安装

不使用GPU忽略第三行

```bash
conda create -n cmoonvision python=3.8
conda activate cmoonvision
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install -r requirements.txt
```

修改默认路径

```
cd .config/Ultralytics
gedit settings.yaml
```

将三个目录替换为你对应的绝对路径

```yaml
datasets_dir: /home/cmoon/PythonCodes/CmoonVision/datasets
weights_dir: /home/cmoon/PythonCodes/CmoonVision/weights
runs_dir: /home/cmoon/PythonCodes/CmoonVision/runs
uuid: 3488c788a3faf60bdd221418b00faf776599a96db2fc55d2f02e7d80cbba0de8
sync: true
api_key: ''
settings_version: 0.0.3
```

## 使用

### CLI

#### 目标检测

```bash
#默认参数（按q停止）
python CmoonVision.py
#设定模型参数
python CmoonVision.py --task detect --weights yolov8s.pt --imgsz 640 --conf 0.8 --iou 0.7
#只检测指定物体(例:只检测瓶子和人)
python CmoonVision.py --task detect --classes=bottle,person
#寻找物体（检测到指定物体停止）
python CmoonVision.py --task detect --find bottle
#只关注出现在roi区域中的物体(例:只检测出现在画面左半边的物体(0.0 0.5表示宽度范围,0.0 1.0表示高度范围))
python CmoonVision.py --task detect --find bottle --roi 0.0 0.5 0.0 1.0
#使用电脑摄像头(source=cam或0)
python CmoonVision.py --task detect --source cam
#使用本地图片或视频(source为图片或视频路径)
python CmoonVision.py --task detect --source sources/image1.jpg
#使用track计数
python CmoonVision.py --task detect --source cam --track
#不保存或不显示
python CmoonVision.py --nosave --noshow
```

#### 实例分割

```bash
#默认参数（按q停止）
python CmoonVision.py --task segment --weights yolov8s-seg.pt
#设定模型参数
python CmoonVision.py --task segment --weights yolov8s-seg.pt --imgsz 640 --conf 0.8 --iou 0.7
#只检测指定物体(例:只检测瓶子和人)
python CmoonVision.py --task segment --weights yolov8s-seg.pt --classes=bottle,person
#寻找物体（检测到指定物体停止）
python CmoonVision.py --task segment --weights yolov8s-seg.pt --find bottle
#只关注出现在roi区域中的物体(例:只检测出现在画面左半边的物体(0.0 0.5表示宽度范围,0.0 1.0表示高度范围))
python CmoonVision.py --task segment --weights yolov8s-seg.pt --find bottle --roi 0.0 0.5 0.0 1.0
#使用电脑摄像头(source=cam或0或不传)
python CmoonVision.py --task segment --weights yolov8s-seg.pt --source cam
#使用本地图片或视频(source为图片或视频路径)
python CmoonVision.py --task segment --weights yolov8s-seg.pt --source sources/image1.jpg
#使用track计数
python CmoonVision.py --task segment --weights yolov8s-seg.pt --source cam --track
```

#### 图片分类

```bash
#默认参数（按q停止）
python CmoonVision.py --task classify --weights yolov8s-cls.pt
#寻找物体（检测到概率最高是指定类别停止）
python CmoonVision.py --task classify --weights yolov8s-cls.pt --find water_bottle
#使用电脑摄像头(source=cam或0)
python CmoonVision.py --task classify --weights yolov8s-cls.pt --source cam
#使用本地图片或视频(source为图片或视频路径)
python CmoonVision.py --task classify --weights yolov8s-cls.pt --source sources/image1.jpg
```

#### 人体关键点检测

```bash
#默认参数（按q停止）
python CmoonVision.py --task pose --weights yolov8n-pose.pt
#使用电脑摄像头(source=cam或0)
python CmoonVision.py --task pose --weights yolov8n-pose.pt --source cam
#使用本地图片或视频(source为图片或视频路径)
python CmoonVision.py --task pose --weights yolov8n-pose.pt --source sources/image1.jpg
```

#### 所有可选参数

```bash
--task 任务类型(detect,segment,classify)
--weights .pt模型路径
--imgsz	图片大小(默认640)
--conf 置信度阈值(默认0.25)
--iou iou阈值(默认0.7)
--source 图像来源(电脑摄像头:cam,0;本地图片或视频:路径)
--find 需要寻找的物体名称,找到就停止检测(例:--find bottle)
--classes: 哪些物体可以被检测到,字符串名称间用','分割(例:--classes bottle,person)
--roi 画面中心多大范围内的检测结果被采用(左宽度 右宽度 上高度 下高度) 例: --roi 0.0 0.5 0.0 1.0
--nosave 不保存图片
--noshow 不显示结果
--track 使用track计数（只支持detect和segment）
```

### 代码调用

#### example1

```python
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
```

#### example2

```python
from CmoonVision import YoloDet,YoloSeg,YoloCls,YoloPose
from pathlib import Path

def main():
    detector = YoloDet()
    results = detector(source=0)
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
```

#### 自定义检测器（更改或增加功能）

```python
from CmoonVision import YoloDet

class MyDet(YoloDet):
    def __init__(self):
        super().__init__()
        
    def new_feature(self):
        """增加新功能"""
        pass
    
    def predict(self):
        """修改原功能"""
        pass

def main():
    detector = MyDet()
    results = detector(source=0)
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
```

### 参数说明

#### CmoonVision

##### Parameters

|  Name   | Type  | Description | Default  |
| :-----: | :---: | :---------: | :------: |
|  task   |  str  |  任务类型   | Required |
| weights |  str  | .pt模型路径 | Required |
|  imgsz  |  int  |  图片大小   |   640    |
|  conf   | float | 置信度阈值  |   0.25   |
|   iou   | float |   iou阈值   |   0.7    |

#### predict 模型推理函数,返回根据不同任务自定义的数据类型list 

##### Parameters

|  Name   |   Type   |                         Description                          |      Default      |
| :-----: | :------: | :----------------------------------------------------------: | :---------------: |
| source  |   any    | 图片来源:cv2读取的图片(np.ndarray)/电脑摄像头('cam','0',0)/本地文件(str) |     Required      |
| classes |   str    | 哪些物体可以被检测到,字符串名称间用','分割('bottle,person')  |       None        |
|  find   |   str    |                 需要寻找的物体名称('bottle')                 |       None        |
|   roi   | sequence |    画面中心多大范围内的检测结果被采用([0.0,0.5,0.0,1.0])     | (0.0,1.0,0.0,1.0) |
| nosave  |   bool   |                    不保存结果到runs文件夹                    |       False       |
| noshow  |   bool   |                          不显示结果                          |       False       |
|  track  |   bool   |          是否使用track计数（只支持detect和segment）          |       False       |

#### 目标检测结果数据类型DetResult

##### Parameters:

|  Name  |    Type     |      Description       | Default  |
| :----: | :---------: | :--------------------: | :------: |
|  name  |     str     |      物体标签名称      | Required |
|  box   |  List[int]  | 矩形框左上右下xyxy坐标 | Required |
| center |  List[int]  |    矩形框中心点坐标    | Required |
|  conf  |    float    |         置信度         | Required |
|  img0  | np.ndarrary |          原图          | Required |
|   id   |     int     |       第几个物体       |   None   |

##### Attributes

|  Name  |    Type     |      Description       |
| :----: | :---------: | :--------------------: |
|  name  |     str     |      物体标签名称      |
|  box   |  List[int]  | 矩形框左上右下xyxy坐标 |
| center |  List[int]  |    矩形框中心点坐标    |
|  conf  |    float    |         置信度         |
|  img0  | np.ndarrary |          原图          |
|   id   |     int     |       第几个物体       |

#### 实例分割结果数据类型SegResult

##### Parameters:

|  Name  |    Type    |      Description       | Default  |
| :----: | :--------: | :--------------------: | :------: |
|  name  |    str     |      物体标签名称      | Required |
|  box   | List[int]  | 矩形框左上右下xyxy坐标 | Required |
| center | List[int]  |    矩形框中心点坐标    | Required |
|  conf  |   float    |         置信度         | Required |
|  img0  | np.ndarray |          原图          | Required |
|  mask  | np.ndarray |      分割结果mask      |   None   |
|   id   |    int     |       第几个物体       |   None   |

##### Attributes

|   Name   |    Type    |      Description       |
| :------: | :--------: | :--------------------: |
|   name   |    str     |      物体标签名称      |
|   box    | List[int]  | 矩形框左上右下xyxy坐标 |
|  center  | List[int]  |    矩形框中心点坐标    |
|   conf   |   float    |         置信度         |
|   img0   | np.ndarray |          原图          |
|   mask   | np.ndarray |      分割结果mask      |
| img_mask | np.ndarray |   画上分割结果的图片   |
|    id    |    int     |       第几个物体       |

#### 图像分类结果数据类型ClsResult

##### Parameters:

| Name  |    Type    |       Description        | Default  |
| :---: | :--------: | :----------------------: | :------: |
| probs |    List    | 概率前五结果 [name,conf] | Required |
| img0  | np.ndarray |           原图           | Required |

##### Attributes

| Name  |    Type    |       Description        |
| :---: | :--------: | :----------------------: |
| probs |    List    | 概率前五结果 [name,conf] |
| img0  | np.ndarray |           原图           |
| name  |    str     |    概率最高的结果名称    |

#### 人体关键点检测结果数据类型PoseResult

##### Parameters:

|  Name  |    Type    |       Description       | Default  |
| :----: | :--------: | :---------------------: | :------: |
|  box   |    List    | 矩形框左上右下xyxy坐标  | Required |
| ioints |    List    | 关节点list [[x,y,prob]] | Required |
|  img0  | np.ndarray |          原图           | Required |

##### Attributes

|  Name  |    Type    |        Description         |
| :----: | :--------: | :------------------------: |
|  box   |    List    | 矩形框左上右下**xyxy**坐标 |
| ioints |    List    |       关节点x,y,prob       |
|  img0  | np.ndarray |            原图            |



