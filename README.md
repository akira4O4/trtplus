# TRTPlus

---

## Introduction

- Easy to Use
- Fast Inference
- Automatic Model Decoding

---

## Support

**Only Testing On Linux.**

| Task                  | Work | 
|-----------------------|:----:|
| segmentation          |  ✔️  | 
| yolov8 classification |  ✔️  | 
| yolov8 detection      |  ✔️  | 
| yolov8 segmentation   |  ❔   | 
| yolov8 obb detection  |  ❔   | 
| yolov10 detection     |  ✔️  | 
| yolov11 detection     |  ✔️  | 

## Install

```bash
cd <your work dir>
git clone https://github.com/akira4O4/tensorrt-lab.git
cd tensorrt-lab
```

---

## How to use

### Step1: Write your label.txt

If your project task is a```segmentation```,the ```label.txt``` first line must be ```0_background_```.

E.g. ```classification_label.txt```

```txt
classes1
classes2
...
classes7
classes8
```

E.g. ```segmentation_label.txt```

```txt
0_background_
1_xxx
2_xxx
...
5_xxx
6_xxx
```

### Step2: Modify Your ```main.cpp```

Refer to the ```cpp``` file in the ```examples``` folder to write your code

```bash
examples/
├── yolov10_det.cpp
├── yolov8_cls.cpp
├── yolov8_det.cpp
└── yolov8_seg.cpp
```

---

## Compile & Rund

**Attention**: Configure your CmakeLists.txt

```bash
mkdir build && cd build
cmake ..
sudo make -j8

./main.bin
```