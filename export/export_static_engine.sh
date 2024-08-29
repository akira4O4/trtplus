#!/bin/bash
TRTEXEC_PATH=trtexec
#-----------------------------------------------------------------------
MODEL_NAME=yolov8n-cls-1x3x224x224
#-----------------------------------------------------------------------
INPUT_PATH=/home/seeking/llf/code/trtplus/assets/yolo
OUTPUT_PATH=${INPUT_PATH}
#-----------------------------------------------------------------------
WORKSPACE_SIZE=1024
#-----------------------------------------------------------------------
${TRTEXEC_PATH} \
  --onnx=${INPUT_PATH}/${MODEL_NAME}.onnx \
  --workspace=${WORKSPACE_SIZE} \
  --saveEngine=${OUTPUT_PATH}/${MODEL_NAME}.engine \
  --fp16
#  --int8
#  --best

