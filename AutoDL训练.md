# AutoDL 训练

## 准备数据

### 推荐使用Roboflow标注数据

#### 在上传图片后可分配多人共同标注

#### 可快速划分训练集验证集和测试集比例

#### 可对图像进行预处理和数据增强

#### 在Export中选择YOLOv8格式下载

![robo1](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/4c3221a1c3ac0975b16254bcbc3958f0-robo1-b9d7af.png)

#### 下载后文件结构

![image-20221010165132896](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2022/10/10/60cf532e0bd83bab2ff8c99a6220bc23-image-20221010165132896-21dcec.png)

## 租用实例

### 在“我的实例”中选择需要的GPU型号

![image-20221010120824235](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2022/10/10/6f7503046b44a6d7b92b10d8ff9f8f0b-image-20221010120824235-d691a5.png)

### 下方选择“基础镜像”

![robo2](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/18f2495a255f45351e10409ca031eaa6-robo2-969fbf.png)

### 点击“立即创建”

## 训练准备

### 在“我的实例”中选择租用的主机开机，点击右侧JupyterLab进入

![image-20221010140054916](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2022/10/10/1ecc3d4dabea4be10233ff390de1d87d-image-20221010140054916-9e808a.png)

### 换源,安装Ultralytics

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install ultralytics
```

### 输入*yolo checks*检查是否安装正常

![train2](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/3666a8c391905c744aad0b5f7f21510d-train2-c9d6d6.png)

### 上传文件

![train3](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/773f5d0b61a096edc020995619870535-train3-6bfbe8.png)

#### 上传roboflow下载的dataset

#### 在终端运行解压命令

```bash
unzip filename.zip -d datasets
```

#### 刷新后可看到解压出的datasets文件夹

#### 上传/.config/Ultralytics目录下的Arial.ttf

![train4](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/e34c44dac4eff04e18efdff3ff712468-train4-cb99bc.png)

#### 将字体文件复制到/root/.config/Ultralytics

```bash
sudo cp Arrial.ttf /root/.config/Ultralytics
```

#### 上传需要用到的权重

![train5](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/4cf6940f5352f865357b43468cd075ab-train5-4f0048.png)

## 开始训练(两种方法)

### 从预训练模型微调

```
yolo detect train data=datasets/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 从头训练

#### 上传模型和预训练模型对应的yaml文件![train6](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/d6a2b2c669dd18c83928aa6106cda693-train6-8b4d65.png)

#### 修改yaml中的nc和data.yaml相同

![train9](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/f6d5c57f953ac3d144ebde3dd0c75b86-train9-d190d5.png)

```
yolo detect train data=datasets/data.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
```

### 若报错CUDA out of memory则减小batch

```bash
yolo detect train data=datasets/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8
```

### 若训练中断可传参数 resume=True,并将model设为最后一次训练结果,断点续训

```bash
yolo detect train data=datasets/data.yaml model=runs/detect/train4/weights/last.pt epochs=100 resume=True
```

### 训练好后进入/runs/detect/train/weights目录，下载best.pt

![train10](https://cdn.jsdelivr.net/gh/Cmoon-cyl/Image-Uploader@latest/ubuntu20.04/2023/04/08/f344f2523ee7ce23b76c547073450b37-train10-f3444d.png)



