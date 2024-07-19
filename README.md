# TensorRT-Lab

**Support Tensorrt Version 8.2.x**

---

## Introdution

This Repo is a TensorRT **automated inference** framework, where the framework will **automatically analyze** the model and **allocate memory** to
infer
images.

---

## Feat

- Only Support **Linux** for now
- Auto-allocation CPU and GPU memory
- Auto-analysis TensorRT Model
- Auto-support **Dynamic** dmoel

---

## Install

```bash
cd <your work dir>
git clone https://github.com/akira4O4/tensorrt-lab.git
cd tensorrt-lab
```

---

## How to use

### Step1:Convert your model : ONNX->Engine

File Path: ```tensorrt-lab/onnx2engine/export_static(dynamic)_engine.sh```

Static Args:

```shell
TRTEXEC_PATH  = # tensorrt trtexec path
MODEL_NAME    = # onnx model name without suffix
INPUT_PATH    = # model path
OUTPUT_PATH   = # engine output path (default OUTPUT_PATH=INPUT_PATH)
WORKSPACE_SIZE= # default 1024
```

Dynamic Args:

```shell
TRTEXEC_PATH  = # tensorrt trtexec path
MODEL_NAME    = # onnx model name without suffix
INPUT_PATH    = # model path
OUTPUT_PATH   = # engine output path (default OUTPUT_PATH=INPUT_PATH)
WORKSPACE_SIZE= # default 1024
INPUT_NAME    = # model input node name 
C             = # input image channel
H             = # input image height 
W             = # input image width
MIN_BATCH     = # min input batch 
OPT_BATCH     = # opt input batch 
MAX_BATCH     = # max input batch 
```

Run:

```bash
cd /onnx2engine
sudo chmod +x export_static(dynamica)_engine.sh
./export_static(dynamica)_engine.sh
```

### Step2: Write your label.txt

If your project task is a```segmentation```,the ```label.txt``` first line must be ```0_background_```.

E.g. ```classification_label.txt```

```txt
0_tench
1_goldfish
...
998_ear
999_toilet_tissue
```

E.g. ```segmentation_label.txt```

```txt
0_background_
1_xxx
2_xxx
```

### Step3: Modify ```main.cpp``` Args

```c++
auto task        = "";                  #   Project task i.e. classication or segmentation
auto model_path  = "";                  #   Engine model path
auto images_dir  = "";                  #   Test image dir
auto output_dir  = "";                  #   Output image dir
auto device      = kDefaultDevice;      #   GPU idx i.e. 0,1,2,
auto batch       = 1;                   #   Ignore if your model is static
auto thr         = std::vector<float>{};#   Model thr list
auto labels_file = "";                  #   Label.txt path
auto mode        = kDefaultMode;        #   FP32 or FP16
```

---

## Compile

**Attention**: Configure your CmakeLists.txt

```bash
mkdir build && cd build
cmake ..
sudo make -j8
```