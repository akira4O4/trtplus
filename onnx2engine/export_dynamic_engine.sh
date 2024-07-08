#!/bin/bash
TRTEXEC_PATH=trtexec
#-----------------------------------------------------------------------
MODEL_NAME=yolov8n-cls-dynamic
#-----------------------------------------------------------------------
INPUT_PATH=/home/seeking/llf/code/tensorRT-lab/temp/models/yolov8
OUTPUT_PATH=${INPUT_PATH}
#OUTPUT_PATH=/home/seeking/llf/code/tensorRT-lab/temp/models/yolov8
#-----------------------------------------------------------------------
INPUT_NAME=images
C=3
H=224
W=224
MIN_BATCH=1
OPT_BATCH=6
MAX_BATCH=6
#-----------------------------------------------------------------------
WORKSPACE_SIZE=1024
#-----------------------------------------------------------------------
${TRTEXEC_PATH} \
  --onnx=${INPUT_PATH}/${MODEL_NAME}.onnx \
  --minShapes=${INPUT_NAME}:${MIN_BATCH}x${C}x${H}x${W} \
  --optShapes=${INPUT_NAME}:${OPT_BATCH}x${C}x${H}x${W} \
  --maxShapes=${INPUT_NAME}:${MAX_BATCH}x${C}x${H}x${W} \
  --workspace=${WORKSPACE_SIZE} \
  --saveEngine=${OUTPUT_PATH}/${MODEL_NAME}.engine
#-----------------------------------------------------------------------

