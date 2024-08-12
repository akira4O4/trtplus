#!/bin/bash
TRTEXEC_PATH=trtexec
#-----------------------------------------------------------------------
MODEL_NAME=yolov8n-cls-bs6
#-----------------------------------------------------------------------
INPUT_PATH=/home/seeking/llf/code/tensorRT-lab/temp/models/yolov8
OUTPUT_PATH=${INPUT_PATH}
#OUTPUT_PATH=/home/seeking/llf/code/trtplus/temp/models/yolov8
#-----------------------------------------------------------------------
WORKSPACE_SIZE=1024
#-----------------------------------------------------------------------
${TRTEXEC_PATH} \
  --onnx=${INPUT_PATH}/${MODEL_NAME}.onnx \
  --workspace=${WORKSPACE_SIZE} \
  --saveEngine=${OUTPUT_PATH}/${MODEL_NAME}.engine
#-----------------------------------------------------------------------

